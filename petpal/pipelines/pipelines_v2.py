import copy
import os.path
import networkx as nx
from ..utils.image_io import safe_copy_meta
from typing import Callable, Union
import inspect
from ..preproc import preproc
from ..input_function import blood_input
from ..kinetic_modeling import parametric_images
from ..kinetic_modeling import tac_fitting
from ..kinetic_modeling import reference_tissue_models as pet_rtms
from ..kinetic_modeling import graphical_analysis as pet_grph
from .pipelines import ArgsDict


class FunctionBasedStep():
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
    
    
class ObjectBasedStep():
    def __init__(self,
                 name: str,
                 class_type: type,
                 init_kwargs: dict,
                 call_kwargs: dict) -> None:
        self.name: str = name
        self.class_type: type = class_type
        self.init_kwargs: ArgsDict = ArgsDict(init_kwargs)
        self.call_kwargs: ArgsDict = ArgsDict(call_kwargs)
        self.init_sig: inspect.Signature = inspect.signature(self.class_type.__init__)
        self.call_sig: inspect.Signature = inspect.signature(self.class_type.__call__)
        self.validate_kwargs()
        self.instance: type = self.class_type(**self.init_kwargs)
        
    def remake_instance(self):
        self.instance = self.class_type(**self.init_kwargs)
    
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
        
    def execute(self, remake_obj: bool = True) -> None:
        if remake_obj:
            self.remake_instance()
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
                    f'{unset_init_args if unset_init_args else "N/A"}',
                    'Call Arguments:',
                    f'{self.call_kwargs if self.call_kwargs else "N/A"}',
                    'Default Call Arguments:',
                    f'{unset_call_args if unset_call_args else "N/A"}']
        return '\n'.join(info_str)

    
class ImageToImageStep(FunctionBasedStep):
    def __init__(self,
                 name: str,
                 function: Callable,
                 input_image_path: str,
                 output_image_path: str,
                 *args,
                 **kwargs) -> None:
        super().__init__(name, function, *(input_image_path, output_image_path, *args), **kwargs)
        self.input_image_path = self.args[0]
        self.output_image_path = self.args[1]
        self.args = self.args[2:]
    
    def execute(self, copy_meta_file: bool = True) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path, self.output_image_path, self.args, **self.kwargs)
        if copy_meta_file:
            safe_copy_meta(input_image_path=self.input_image_path,
                           out_image_path=self.output_image_path)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        io_dict = ArgsDict({'input_image_path': self.input_image_path,
                            'output_image_path': self.output_image_path
                           })
        sup_str_list = super().__str__().split('\n')
        args_ind = sup_str_list.index("Arguments Passed:")
        sup_str_list.insert(args_ind, f"Input & Output Paths:\n{io_dict}")
        def_args_ind = sup_str_list.index("Default Arguments:")
        sup_str_list.pop(def_args_ind + 1)
        sup_str_list.pop(def_args_ind + 1)
        
        return "\n".join(sup_str_list)
    
    def set_input_as_output_from(self, sending_step: FunctionBasedStep) -> None:
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            raise TypeError(
                    f"The provided step: {sending_step}\n is not an instance of ImageToImageStep. "
                    f"It is of type {type(sending_step)}.")
        

class GraphicalAnalysisStep(ObjectBasedStep):
    def __init__(self,
                 input_tac_path: str,
                 roi_tac_path: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0,):
        super().__init__(name=f'{method}-analysis',
                         class_type=pet_grph.GraphicalAnalysis,
                         init_kwargs=dict(input_tac_path=input_tac_path,
                                          roi_tac_path=roi_tac_path,
                                          output_directory=output_directory,
                                          output_filename_prefix=output_prefix),
                         call_kwargs=dict(method_name=method, t_thresh_in_mins=fit_threshold_in_mins))
        
        
def get_template_steps():
    
    out_dict = dict(
            thresh_crop_step = ImageToImageStep(name='thresh_crop',
                                                function=preproc.image_operations_4d.SimpleAutoImageCropper,
                                                input_image_path='',
                                                output_image_path='',),
            moco_frames_above_mean = ImageToImageStep(name='moco_frames_above_mean',
                                                      function=preproc.motion_corr.motion_corr_frames_above_mean_value,
                                                      input_image_path='',
                                                      output_image_path='',
                                                      motion_target_option='mean_image',
                                                      verbose=True),
            register_pet_to_t1 = ImageToImageStep(name='register_pet_to_t1',
                                                  function=preproc.register.register_pet,
                                                  input_image_path='',
                                                  output_image_path='',
                                                  reference_image_path='',
                                                  motion_target_option='weighted_series_sum',
                                                  half_life='',
                                                  verbose=True),
            write_roi_tacs = FunctionBasedStep(name='write_roi_tacs',
                                               function=preproc.image_operations_4d.write_tacs,
                                               input_image_path='',
                                               label_map_path='',
                                               segmentation_image_path='',
                                               out_tac_dir='',
                                               verbose=True,
                                               time_frame_keyword='FrameReferenceTime',
                                               out_tac_prefix=''),
            resample_blood = FunctionBasedStep(name='resample_bTAC',
                                               function=blood_input.resample_blood_data_on_scanner_times,
                                               pet4d_path='',
                                               raw_blood_tac='',
                                               out_tac_path=',',
                                               lin_fit_thresh_in_mins=30.0),
            
    
            )
    
    return out_dict

