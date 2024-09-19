from typing import Callable
import inspect

class ArgsDict(dict):
    def __str__(self):
        rep_str = [f'    {arg}={val}' for arg, val in self.items()]
        return ',\n'.join(rep_str)


class GenericStep:
    def __init__(self, name: str, function: Callable, **kwargs) -> None:
        self.name = name
        self.function = function
        self._func_name = function.__name__
        self.kwargs = ArgsDict(kwargs)
        self.func_sig = inspect.signature(self.function)
        self.validate_kwargs_for_non_default_have_been_set()
        
    def get_non_unset_args_from_function(self):
        """
        This class gets all the function arguments as kwargs so we check through them all.
        Returns:

        """
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        for arg_name, arg_val in func_params.items():
            if arg_name not in self.kwargs:
                    unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def get_empty_default_kwargs(self):
        unset_args_dict = self.get_non_unset_args_from_function()
        empty_kwargs = []
        for arg_name, arg_val in unset_args_dict.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in self.kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
        
    
    def validate_kwargs_for_non_default_have_been_set(self):
        empty_kwargs = self.get_empty_default_kwargs()
        if empty_kwargs:
            unset_args = '\n'.join(empty_kwargs)
            raise RuntimeError(f"For {self._func_name}, the following arguments must be set:\n{unset_args}")
        
    
    def execute(self) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(**self.kwargs)
        print(f"(Info): Finished {self.name}")
        
        
    def __str__(self):
        info_str = [f'({type(self).__name__} Info):',
                    f'Step Name: {self.name}',
                    f'Function Name: {self._func_name}',
                    'Arguments Set:',
                    f'{self.kwargs}',
                    'Default Arguments:',
                    f'{self.get_non_unset_args_from_function()}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        repr_kwargs = ArgsDict({'name'    : self.name,
                                'function': f'{self.function.__module__}.{self.function.__name__}',
                                })
        obj_signature = inspect.signature(self.__init__).parameters
        for arg_name in list(obj_signature)[2:-1]:
            repr_kwargs[arg_name] = self.__dict__[arg_name]
        repr_kwargs = ArgsDict({**repr_kwargs, **self.kwargs})
        
        return f'{type(self).__name__}(\n{repr_kwargs}\n)'


class ImageToImageStep(GenericStep):
    def __init__(self, name: str, function: Callable,
                 input_image_path: str, output_image_path: str,
                 **kwargs) -> None:
        super().__init__(name, function, **kwargs)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        
    def execute(self) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path, self.output_image_path, **self.kwargs)
        print(f"(Info): Finished {self.name}")
        
    def get_non_unset_args_from_function(self):
        """
        Since this class expects to have functions with the signature f(some_input_path, some_output_path, **kwargs),
        we want to skip the first two arguments
        Returns:

        """
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        for arg_name, arg_val in list(func_params.items())[2:]:
            if (arg_name not in self.kwargs) and ('kwargs' not in arg_name):
                    unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def __str__(self):
        io_dict = ArgsDict({'input_image_path': self.input_image_path,
                            'output_image_path': self.output_image_path})
        sup_str_list = super().__str__().split('\n')
        args_ind = sup_str_list.index("Arguments Set:")+1
        sup_str_list.insert(args_ind, f"{io_dict}")
        
        return "\n".join(sup_str_list)
    
        
class GenericPipeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self.steps = []
        
    def add_step(self, step: GenericStep) -> None:
        self.steps.append(step)
        
    def list_step_details(self) -> None:
        print(f"(Pipeline Info):")
        for step_id, a_step in enumerate(self.steps):
            print('-' * 80)
            print(f"Step Number {step_id+1}")
            print(a_step)
            print('-' * 80)
        
    def list_step_names(self) -> None:
        print(f"(Pipeline Info):")
        print('-' * 80)
        for step_id, a_step in enumerate(self.steps):
            print(f"Step Number {step_id+1}: {a_step.name}")
        print('-' * 80)
    
    def run_steps(self) -> None:
        for step in self.steps:
            step.execute()


class ProcessingPipeline(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.preproc = GenericPipeline('preproc')
        self.kinetic_modeling = GenericPipeline('kinetic_modeling')
    
    def run_preproc(self):
        self.preproc.run_steps()
    
    def run_kinetic_modeling(self):
        self.kinetic_modeling.run_steps()
    
    def add_preproc_step(self, step: GenericStep) -> None:
        self.preproc.add_step(step)
    
    def add_kinetic_modeling_step(self, step: GenericStep) -> None:
        self.kinetic_modeling.add_step(step)
    
    def list_preproc_stes(self) -> None:
        self.preproc.list_step_names()
    
    def list_kinetic_modeling_stes(self) -> None:
        self.kinetic_modeling.list_step_names()
    
    def list_preproc_step_details(self) -> None:
        self.preproc.list_step_details()
    
    def list_kinetic_modeling_step_details(self) -> None:
        self.kinetic_modeling.list_step_details()
