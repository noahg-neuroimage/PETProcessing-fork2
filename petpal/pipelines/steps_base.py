import inspect
from typing import Callable


class ArgsDict(dict):
    """
    A specialized subclass of Python's built-in `dict` that provides a customized string representation.

    Attributes:
        None, since ArgsDict inherits directly from dict.

    Methods:
        __str__(): Returns a formatted string representation of the dictionary contents.

    """
    def __str__(self):
        """
        Returns a formatted string representation of the dictionary.
        
        The string output will list each key-value pair on a new line with indentation to improve readability.
        
        Returns:
            str: A string containing the formatted key-value pairs, indented for clarity.
        """
        rep_str = [f'    {arg}={val}' for arg, val in self.items()]
        return ',\n'.join(rep_str)


class StepsAPI:
    """
    StepsAPI provides an interface for defining steps in a processing pipeline.

    This class outlines methods that allow input and output management between different steps,
    and perform inference of output files based on input data and given parameters.

    Methods
    -------
    set_input_as_output_from(sending_step):
        Sets the input of the current step as the output from a specified sending step.

    infer_outputs_from_inputs(out_dir, der_type, suffix=None, ext=None, **extra_desc):
        Infers output files from input data based on the specified output directory,
        derivative type, optional suffix and extension, plus any extra descriptions.
    """
    
    def set_input_as_output_from(self, sending_step):
        """
        Sets the input of the current step as the output from a specified sending step.

        Args:
            sending_step: The previous step from which the output will be used as input for the current step.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type: str, suffix: str = None, ext: str = None, **extra_desc):
        """
        Infers output files from input data based on the specified output directory,
        derivative type, optional suffix and extension, plus any extra descriptions.

        Args:
            out_dir (str): The directory where the output files will be saved.
            der_type (str): The type of derivative being produced.
            suffix (str, optional): An optional suffix for the output files.
            ext (str, optional): An optional extension for the output files.
            **extra_desc: Additional keyword arguments for extra descriptions to be included.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError


class FunctionBasedStep(StepsAPI):
    def __init__(self, name: str, function: Callable, *args, **kwargs) -> None:
        self.name = name
        self.function = function
        self._func_name = function.__name__
        self.args = args
        self.kwargs = ArgsDict(kwargs)
        self.func_sig = inspect.signature(self.function)
        self.validate_kwargs_for_non_default_have_been_set()
    
    def get_function_args_not_set_in_kwargs(self) -> ArgsDict:
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        arg_names = list(func_params)
        for arg_name in arg_names[len(self.args):]:
            if arg_name not in self.kwargs and arg_name != 'kwargs':
                unset_args_dict[arg_name] = func_params[arg_name].default
        return unset_args_dict
    
    def get_empty_default_kwargs(self) -> list:
        unset_args_dict = self.get_function_args_not_set_in_kwargs()
        empty_kwargs = []
        for arg_name, arg_val in unset_args_dict.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in self.kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def validate_kwargs_for_non_default_have_been_set(self) -> None:
        empty_kwargs = self.get_empty_default_kwargs()
        if empty_kwargs:
            unset_args = '\n'.join(empty_kwargs)
            raise RuntimeError(f"For {self._func_name}, the following arguments must be set:\n{unset_args}")
    
    def execute(self):
        print(f"(Info): Executing {self.name}")
        self.function(*self.args, **self.kwargs)
        print(f"(Info): Finished {self.name}")
    
    def generate_kwargs_from_args(self) -> ArgsDict:
        args_to_kwargs_dict = ArgsDict()
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            args_to_kwargs_dict[arg_name] = arg_val
        return args_to_kwargs_dict
    
    def __str__(self):
        args_to_kwargs_dict = self.generate_kwargs_from_args()
        info_str = [f'({type(self).__name__} Info):',
                    f'Step Name: {self.name}',
                    f'Function Name: {self._func_name}',
                    f'Arguments Passed:',
                    f'{args_to_kwargs_dict if args_to_kwargs_dict else "N/A"}',
                    'Keyword-Arguments Set:',
                    f'{self.kwargs if self.kwargs else "N/A"}',
                    'Default Arguments:',
                    f'{self.get_function_args_not_set_in_kwargs()}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        cls_name = type(self).__name__
        full_func_name = f'{self.function.__module__}.{self._func_name}'
        info_str = [f'{cls_name}(', f'name={repr(self.name)},', f'function={full_func_name},']
        
        init_params = inspect.signature(self.__init__).parameters
        for arg_name in list(init_params)[2:-2]:
            info_str.append(f'{arg_name}={repr(getattr(self, arg_name))},')
        
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            info_str.append(f'{arg_name}={repr(arg_val)}', )
        
        for arg_name, arg_val in self.kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def all_args_non_empty_strings(self):
        for arg in self.args:
            if arg == '':
                return False
        return True
    
    def all_kwargs_non_empty_strings(self):
        for arg_name, arg in self.kwargs.items():
            if arg == '':
                return False
        return True
    
    def can_potentially_run(self):
        return self.all_args_non_empty_strings() and self.all_kwargs_non_empty_strings()


class ObjectBasedStep(StepsAPI):
    def __init__(self, name: str, class_type: type, init_kwargs: dict, call_kwargs: dict) -> None:
        self.name: str = name
        self.class_type: type = class_type
        self.init_kwargs: ArgsDict = ArgsDict(init_kwargs)
        self.call_kwargs: ArgsDict = ArgsDict(call_kwargs)
        self.init_sig: inspect.Signature = inspect.signature(self.class_type.__init__)
        self.call_sig: inspect.Signature = inspect.signature(self.class_type.__call__)
        self.validate_kwargs()
    
    def validate_kwargs(self):
        empty_init_kwargs = self.get_empty_default_kwargs(self.init_sig, self.init_kwargs)
        empty_call_kwargs = self.get_empty_default_kwargs(self.call_sig, self.call_kwargs)
        
        if empty_init_kwargs or empty_call_kwargs:
            err_msg = [f"For {self.class_type.__name__}, the following arguments must be set:"]
            if empty_init_kwargs:
                err_msg.append("Initialization:")
                err_msg.append(f"{empty_init_kwargs}")
            if empty_call_kwargs:
                err_msg.append("Calling:")
                err_msg.append(f"{empty_call_kwargs}")
            raise RuntimeError("\n".join(err_msg))
    
    @staticmethod
    def get_args_not_set_in_kwargs(sig: inspect.Signature, kwargs: dict) -> dict:
        unset_args_dict = ArgsDict()
        for arg_name, arg_val in sig.parameters.items():
            if arg_name not in kwargs and arg_name != 'self':
                unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def get_empty_default_kwargs(self, sig: inspect.Signature, set_kwargs: dict) -> list:
        unset_kwargs = self.get_args_not_set_in_kwargs(sig=sig, kwargs=set_kwargs)
        empty_kwargs = []
        for arg_name, arg_val in unset_kwargs.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in set_kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def execute(self) -> None:
        print(f"(Info): Executing {self.name}")
        obj_instance = self.class_type(**self.init_kwargs)
        obj_instance(**self.call_kwargs)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        unset_init_args = self.get_args_not_set_in_kwargs(self.init_sig, self.init_kwargs)
        unset_call_args = self.get_args_not_set_in_kwargs(self.call_sig, self.call_kwargs)
        
        info_str = [f'({type(self).__name__} Info):', f'Step Name: {self.name}',
                    f'Class Name: {self.class_type.__name__}', 'Initialization Arguments:', f'{self.init_kwargs}',
                    'Default Initialization Arguments:', f'{unset_init_args if unset_init_args else "N/A"}',
                    'Call Arguments:', f'{self.call_kwargs if self.call_kwargs else "N/A"}', 'Default Call Arguments:',
                    f'{unset_call_args if unset_call_args else "N/A"}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        cls_name = type(self).__name__
        full_func_name = f'{self.class_type.__module__}.{self.class_type.__name__}'
        info_str = [f'{cls_name}(', f'name={repr(self.name)},', f'class_type={full_func_name},']
        
        if self.init_kwargs:
            info_str.append('init_kwargs={')
            for arg_name, arg_val in self.init_kwargs.items():
                info_str.append(f'    {arg_name}={repr(arg_val)},')
            info_str[-1] = f'{info_str[-1]}' + '}'
        
        if self.call_kwargs:
            info_str.append('call_kwargs={')
            for arg_name, arg_val in self.call_kwargs.items():
                info_str.append(f'    {arg_name}={repr(arg_val)},')
            info_str[-1] = f'{info_str[-1]}' + '}'
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def all_init_kwargs_non_empty_strings(self):
        for arg_name, arg_val in self.init_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def all_call_kwargs_non_empty_strings(self):
        for arg_name, arg_val in self.call_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def can_potentially_run(self):
        return self.all_init_kwargs_non_empty_strings() and self.all_call_kwargs_non_empty_strings()

