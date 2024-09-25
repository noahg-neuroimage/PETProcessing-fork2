from ..utils.image_io import safe_copy_meta
from typing import Callable
import inspect
from ..preproc import preproc
from ..input_function import blood_input
from ..kinetic_modeling import parametric_images

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
        
    def get_function_args_not_set_in_kwargs(self):
        """
        This class gets all the function arguments as kwargs so we check through them all. Note that we ignore
        kwargs for the function.
        Returns:

        """
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        for arg_name, arg_val in func_params.items():
            if arg_name not in self.kwargs:
                    unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def get_empty_default_kwargs(self):
        unset_args_dict = self.get_function_args_not_set_in_kwargs()
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
                    f'{self.get_function_args_not_set_in_kwargs()}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        repr_kwargs = ArgsDict({'name'    : self.name,
                                'function': f'{self.function.__module__}.{self._func_name}',
                                })
        obj_signature = inspect.signature(self.__init__).parameters
        for arg_name in list(obj_signature)[2:-1]:
            repr_kwargs[arg_name] = getattr(self, arg_name)
        repr_kwargs = ArgsDict({**repr_kwargs, **self.kwargs})
        
        return f'{type(self).__name__}(\n{repr_kwargs}\n)'


class ImageToImageStep(GenericStep):
    def __init__(self, name: str, function: Callable,
                 input_image_path: str, output_image_path: str,
                 **kwargs) -> None:
        super().__init__(name, function, **kwargs)
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        
    def execute(self, copy_meta_file: bool = True) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path, self.output_image_path, **self.kwargs)
        if copy_meta_file:
            safe_copy_meta(input_image_path=self.input_image_path,
                           out_image_path=self.output_image_path)
        print(f"(Info): Finished {self.name}")
        
    def get_function_args_not_set_in_kwargs(self):
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
    
    def set_input_as_output_from(self, sending_step: GenericStep) -> None:
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            raise TypeError(
                    f"The provided step {sending_step} is not an instance of ImageToImageStep. "
                    f"It is of type {type(sending_step)}.")


class ObjectBasedStep:
    def __init__(self,
                 name: str,
                 class_type: type,
                 init_kwargs: dict,
                 call_kwargs: dict):
        self.name : str = name
        self.class_type: type = class_type
        self.init_kwargs: dict = ArgsDict(init_kwargs)
        self.call_kwargs: dict = ArgsDict(call_kwargs)
        self.init_sig: inspect.Signature = inspect.signature(self.class_type.__init__)
        self.call_sig: inspect.Signature = inspect.signature(self.class_type.__call__)
        self.validate_kwargs()
        self.instance: type = self.class_type(**self.init_kwargs)
    
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
        self.instance(**self.call_kwargs)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        unset_init_args = self.get_args_not_set_in_kwargs(self.init_sig, self.init_kwargs)
        unset_call_args = self.get_args_not_set_in_kwargs(self.call_sig, self.call_kwargs)
        
        info_str = [f'({type(self).__name__} Info):',
                    f'Step Name: {self.name}',
                    f'Class Name: {self.class_type.__name__}',
                    'Initialization Arguments:',
                    f'{self.init_kwargs}',
                    'Default Initialization Arguments:',
                    f'{unset_init_args}',
                    'Call Arguments:',
                    f'{self.call_kwargs}',
                    'Default Call Arguments:',
                    f'{unset_call_args}']
        return '\n'.join(info_str)


TEMPLATE_STEPS = {
    'thresh_crop': ImageToImageStep(name='crop',
                                    function=preproc.image_operations_4d.SimpleAutoImageCropper,
                                    input_image_path='',
                                    output_image_path=''
                                   ),
    'moco_frames_above_mean': ImageToImageStep(name='moco',
                                              function=preproc.motion_corr.motion_corr_frames_above_mean_value,
                                              input_image_path='',
                                              output_image_path='',
                                              motion_target_option='mean_image',
                                              verbose=True),
    'register_pet_to_t1' : ImageToImageStep(name='reg',
                                           function=preproc.register.register_pet,
                                           input_image_path='',
                                           output_image_path='',
                                           reference_image_path='',
                                           motion_target_option='weighted_sum_series',
                                           verbose=True),
    'resample_blood' : GenericStep(name='resample_bTAC',
                            function=blood_input.resample_blood_data_on_scanner_times,
                            pet4d_path='',
                            raw_blood_tac='',
                            out_tac_path='',
                            lin_fit_thresh_in_mins=30.0),
    'parametric_patlak' : ObjectBasedStep(name='parametric_patlak',
                              class_type=parametric_images.GraphicalAnalysisParametricImage,
                              init_kwargs=dict(input_tac_path='',
                                               pet4D_img_path='',
                                               output_directory='',
                                               output_filename_prefix=''
                                              ),
                              call_kwargs=dict(method_name='patlak',
                                               t_thresh_in_mins=30.0))
    }


class GenericPipeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self.steps = {}
        
    def add_step(self, step: GenericStep) -> None:
        self.steps[step.name] = step
        
    def list_step_details(self) -> None:
        if not self.steps:
            return None
        print("*"*90)
        print(f"({self.name} Pipeline Info):")
        for step_id, (step_name, a_step) in enumerate(self.steps.items()):
            print('-' * 80)
            print(f"Step Number {step_id+1}")
            print(a_step)
            print('-' * 80)
        print("*" * 90)
        
    def list_step_names(self) -> None:
        if not self.steps:
            return None
        print(f"({self.name} pipeline info):")
        print('-' * 80)
        for step_id, (step_name, a_step) in enumerate(self.steps.items()):
            print(f"Step Number {step_id+1}: {a_step.name}")
        print('-' * 80)
    
    def get_step_names(self):
        if not self.steps:
            return None
        step_names = [a_name for a_name, a_step in self.steps.items()]
        return step_names
    
    def run_steps(self) -> None:
        for step_id, (step_name, a_step) in enumerate(self.steps.items()):
            a_step.execute()
            
    def __getitem__(self, key: str) -> GenericStep:
        """
        Allows accessing steps by name.

        Args:
            key: str, step name

        Returns:
            Corresponding GenericStep object

        Raises:
            KeyError: If the name does not exist
        """
        try:
            return self.steps[key]
        except KeyError:
            raise KeyError(f"No step found with name: {key}")


class ProcessingPipeline(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.preproc = GenericPipeline('preproc')
        self.kinetic_modeling = GenericPipeline('kinetic_modeling')
    
    def run(self):
        self.preproc.run_steps()
        self.kinetic_modeling.run_steps()
    
    def run_preproc(self):
        self.preproc.run_steps()
    
    def run_kinetic_modeling(self):
        self.kinetic_modeling.run_steps()
    
    def add_preproc_step(self, step: GenericStep, receives_output_from_previous_step_as_input: bool = False) -> None:
        self.preproc.add_step(step)
        if receives_output_from_previous_step_as_input:
            step_names = self.preproc.get_step_names()
            this_step_name = step_names[-1]
            last_step_name = step_names[-2]
            self.chain_outputs_as_inputs_between_steps(out_pipe_name='preproc',
                                                       out_step_name=last_step_name,
                                                       in_pipe_name='preproc',
                                                       in_step_name=this_step_name)
    
    def add_kinetic_modeling_step(self, step: GenericStep) -> None:
        self.kinetic_modeling.add_step(step)
    
    def list_preproc_steps(self) -> None:
        self.preproc.list_step_names()
    
    def list_kinetic_modeling_steps(self) -> None:
        self.kinetic_modeling.list_step_names()
    
    def list_preproc_step_details(self) -> None:
        self.preproc.list_step_details()
    
    def list_kinetic_modeling_step_details(self) -> None:
        self.kinetic_modeling.list_step_details()
    
    def chain_outputs_as_inputs_between_steps(self,
                                              out_pipe_name: str,
                                              out_step_name: str,
                                              in_pipe_name : str,
                                              in_step_name: str) -> None:
        
        try:
            out_pipeline : ImageToImageStep = getattr(self, out_pipe_name)
            in_pipeline : ImageToImageStep = getattr(self, in_pipe_name)
            in_pipeline[in_step_name].set_input_as_output_from(out_pipeline[out_step_name])
        except AttributeError as e:
            raise RuntimeError(f"Error setting chaining outputs and inputs for steps: {e}")
        
