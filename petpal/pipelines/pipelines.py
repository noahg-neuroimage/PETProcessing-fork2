import copy
import pathlib
import warnings
import os
import networkx as nx
from ..utils.image_io import safe_copy_meta, get_half_life_from_nifty
from typing import Callable, Union
import inspect
from ..preproc.image_operations_4d import SimpleAutoImageCropper, write_tacs
from ..preproc.register import register_pet
from ..preproc.motion_corr import motion_corr_frames_above_mean_value
from ..input_function import blood_input
from ..kinetic_modeling import parametric_images
from ..kinetic_modeling import tac_fitting
from ..kinetic_modeling import rtm_analysis as pet_rtms
from ..kinetic_modeling import graphical_analysis as pet_grph
from ..utils.bids_utils import parse_path_to_get_subject_and_session_id, snake_to_camel_case, gen_bids_like_dir_path, gen_bids_like_filepath


class ArgsDict(dict):
    def __str__(self):
        rep_str = [f'    {arg}={val}' for arg, val in self.items()]
        return ',\n'.join(rep_str)


class StepsAPI:
    def set_input_as_output_from(self, sending_step):
        raise NotImplementedError
    
    def infer_outputs_from_inputs(self, out_dir, der_type, suffix=None, ext=None, **extra_desc):
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
   
   
class TACsFromSegmentationStep(FunctionBasedStep):
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path:str,
                 segmentation_label_map_path: str,
                 out_tacs_dir:str,
                 out_tacs_prefix:str,
                 time_keyword='FrameReferenceTime',
                 verbose=False) -> None:
        super().__init__(name='write_roi_tacs',
                         function=write_tacs,
                         input_image_path=input_image_path,
                         segmentation_image_path=segmentation_image_path,
                         label_map_path=segmentation_label_map_path,
                         out_tac_dir=out_tacs_dir,
                         out_tac_prefix=out_tacs_prefix,
                         time_frame_keyword=time_keyword,
                         verbose=verbose,
                         )
        self._input_image = input_image_path
        self._segmentation_image = segmentation_image_path
        self._segmentation_label_map = segmentation_label_map_path
        self._out_tacs_path = out_tacs_dir
        self._out_tacs_prefix = out_tacs_prefix
        
    @property
    def segmentation_image_path(self):
        return self._segmentation_image
    
    @segmentation_image_path.setter
    def segmentation_image_path(self, segmentation_image_path):
        self._segmentation_image = segmentation_image_path
        self.kwargs['segmentation_image_path'] = segmentation_image_path
        
    @property
    def segmentation_label_map_path(self):
        return self._segmentation_label_map
    
    @segmentation_label_map_path.setter
    def segmentation_label_map_path(self, segmentation_label_map_path):
        self._segmentation_label_map = segmentation_label_map_path
        self.kwargs['label_map_path'] = segmentation_label_map_path
        
    @property
    def out_tacs_dir(self):
        return self._out_tacs_path
    
    @out_tacs_dir.setter
    def out_tacs_dir(self, out_tacs_path: str):
        self.kwargs['out_tac_dir'] = out_tacs_path
        self._out_tacs_path = out_tacs_path
    
    @property
    def out_tacs_prefix(self):
        return self._out_tacs_prefix
    
    @out_tacs_prefix.setter
    def out_tacs_prefix(self, out_tacs_prefix: str):
        self.kwargs['out_tac_prefix'] = out_tacs_prefix
        self._out_tacs_prefix = out_tacs_prefix
    
    
    @property
    def out_path_and_prefix(self):
        return self._out_tacs_path, self._out_tacs_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix):
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.out_tacs_dir = out_dir
            self.out_tacs_prefix = out_prefix
        
    @property
    def input_image_path(self):
        return self._input_image
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        self.kwargs['input_image_path'] = input_image_path
        self._input_image = input_image_path
        
    def set_input_as_output_from(self, sending_step):
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
            
    @classmethod
    def default_write_tacs_from_segmentation_rois(cls):
        return cls(input_image_path='',
                   segmentation_image_path='',
                   segmentation_label_map_path='',
                   out_tacs_dir='',
                   out_tacs_prefix='',
                   time_keyword='FrameReferenceTime',
                   verbose=False)
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type, suffix=None, ext=None, **extra_desc):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        outpath = gen_bids_like_dir_path(sub_id=sub_id,
                                         ses_id=ses_id,
                                         sup_dir=out_dir,
                                         modality='tacs')
        self.out_tacs_dir = outpath
        step_name_in_camel_case = snake_to_camel_case(self.name)
        self.out_tacs_prefix = f'sub-{sub_id}_ses-{ses_id}_desc-{step_name_in_camel_case}'
        
    
class ResampleBloodTACStep(FunctionBasedStep):
    def __init__(self,
                 input_raw_blood_tac_path: str,
                 input_image_path: str,
                 out_tac_path:str,
                 lin_fit_thresh_in_mins=30.0
                 ):
        super().__init__(name='resample_PTAC_on_scanner',
                         function=blood_input.resample_blood_data_on_scanner_times,
                         raw_blood_tac=input_raw_blood_tac_path,
                         pet4d_path=input_image_path,
                         out_tac_path=out_tac_path,
                         lin_fit_thresh_in_mins=lin_fit_thresh_in_mins)
        self._raw_blood_tac_path = input_raw_blood_tac_path
        self._input_image_path = input_image_path
        self._resampled_tac_path = out_tac_path
        
    @property
    def raw_blood_tac_path(self):
        return self._raw_blood_tac_path
    
    @raw_blood_tac_path.setter
    def raw_blood_tac_path(self, raw_blood_tac_path):
        self.kwargs['raw_blood_tac'] = raw_blood_tac_path
        self._raw_blood_tac_path = raw_blood_tac_path
        
    @property
    def input_image_path(self):
        return self._input_image_path
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        self.kwargs['pet4d_path'] = input_image_path
        self._input_image_path = input_image_path
        
    @property
    def resampled_tac_path(self):
        return self._resampled_tac_path
    
    @resampled_tac_path.setter
    def resampled_tac_path(self, resampled_tac_path):
        self.kwargs['out_tac_path'] = resampled_tac_path
        self._resampled_tac_path = resampled_tac_path
        
    def set_input_as_output_from(self, sending_step):
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
            
    @classmethod
    def default_resample_blood_tac_on_scanner_times(cls):
        return cls(input_raw_blood_tac_path='',
                   input_image_path='',
                   out_tac_path='',
                   lin_fit_thresh_in_mins=30.0)
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type, suffix='blood', ext='.tsv', **extra_desc):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.raw_blood_tac_path)
        filepath = gen_bids_like_filepath(sub_id=sub_id,
                                          ses_id=ses_id,
                                          bids_dir=out_dir,
                                          modality='preproc',
                                          suffix=suffix,
                                          ext=ext,
                                          desc='OnScannerFrameTimes'
                                          )
        self.resampled_tac_path = filepath
   
        
class ObjectBasedStep(StepsAPI):
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
    

class ImageToImageStep(FunctionBasedStep):
    def __init__(self,
                 name: str,
                 function: Callable,
                 input_image_path: str,
                 output_image_path: str,
                 *args,
                 **kwargs) -> None:
        super().__init__(name, function, *(input_image_path, output_image_path, *args), **kwargs)
        self.input_image_path = copy.copy(self.args[0])
        self.output_image_path = copy.copy(self.args[1])
        self.args = self.args[2:]
    
    def execute(self, copy_meta_file: bool = True) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image_path, self.output_image_path, *self.args, **self.kwargs)
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
            super().set_input_as_output_from(sending_step)
        
    def can_potentially_run(self):
        input_img_non_empty_str = False if self.input_image_path == '' else True
        output_img_non_empty_str = False if self.output_image_path == '' else True
        return super().can_potentially_run() and input_img_non_empty_str and output_img_non_empty_str
    
    def infer_outputs_from_inputs(self,
                                  out_dir: str ,
                                  der_type='preproc',
                                  suffix: str = 'pet',
                                  ext: str = '.nii.gz',
                                  **extra_desc):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        step_name_in_camel_case = snake_to_camel_case(self.name)
        filepath = gen_bids_like_filepath(sub_id=sub_id,
                                          ses_id=ses_id,
                                          suffix=suffix,
                                          bids_dir=out_dir,
                                          modality=der_type,
                                          ext=ext,
                                          desc=step_name_in_camel_case,
                                          **extra_desc)
        self.output_image_path = filepath
    
    @classmethod
    def default_threshold_cropping(cls, **overrides):
        defaults = dict(name='thresh_crop',
                        function=SimpleAutoImageCropper,
                        input_image_path='',
                        output_image_path='',)
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
        
    @classmethod
    def default_moco_frames_above_mean(cls, verbose=False, **overrides):
        defaults = dict(name='moco_frames_above_mean',
                        function=motion_corr_frames_above_mean_value,
                        input_image_path='',
                        output_image_path='',
                        motion_target_option='mean_image',
                        verbose=verbose,
                        half_life=None,)
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
    
    @classmethod
    def default_register_pet_to_t1(cls, reference_image_path='', half_life='', verbose=False, **overrides):
        defaults = dict(name='register_pet_to_t1',
                        function=register_pet,
                        input_image_path='',
                        output_image_path='',
                        reference_image_path=reference_image_path,
                        motion_target_option='weighted_series_sum',
                        verbose=verbose,
                        half_life=half_life, )
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
        

PreprocSteps = Union[TACsFromSegmentationStep, ResampleBloodTACStep, ImageToImageStep]


class TACAnalysisStepMixin(StepsAPI):
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 is_ref_tac_based_model:bool,
                 **kwargs):
        common_init_kwargs = dict(roi_tacs_dir=roi_tacs_dir,
                                  output_directory=output_directory,
                                  output_filename_prefix=output_prefix,)
        if is_ref_tac_based_model:
            self.init_kwargs = dict(ref_tac_path=input_tac_path,
                                    **common_init_kwargs,
                                    **kwargs)
        else:
            self.init_kwargs = dict(input_tac_path=input_tac_path,
                                    **common_init_kwargs,
                                    **kwargs)
        self._tacs_dir = roi_tacs_dir
        self._input_tac_path = input_tac_path
        self._output_directory = output_directory
        self._output_prefix = output_prefix
        
    @property
    def input_tac_path(self) -> str:
        return self._input_tac_path
    
    @input_tac_path.setter
    def input_tac_path(self, input_tac_path: str):
        self._input_tac_path = input_tac_path
        self.init_kwargs['input_tac_path'] = input_tac_path
        
    @property
    def reference_tac_path(self) -> str:
        return self.input_tac_path
    
    @reference_tac_path.setter
    def reference_tac_path(self, ref_tac_path: str):
        self._input_tac_path = ref_tac_path
        self.init_kwargs['ref_tac_path'] = ref_tac_path
        
    @property
    def tacs_dir(self) -> str:
        return self._tacs_dir
    
    @tacs_dir.setter
    def tacs_dir(self, tacs_dir: str):
        self._tacs_dir = tacs_dir
    
    @property
    def roi_tacs_dir(self) -> str:
        return self.tacs_dir
    
    @roi_tacs_dir.setter
    def roi_tacs_dir(self, roi_tacs_dir: str):
        self.tacs_dir = roi_tacs_dir
        self.init_kwargs['roi_tacs_dir'] = roi_tacs_dir
    
    @property
    def output_directory(self) -> str:
        return self._output_directory
    
    @output_directory.setter
    def output_directory(self, output_directory: str):
        self._output_directory = output_directory
        self.init_kwargs['output_directory'] = output_directory
    
    @property
    def output_prefix(self) -> str:
        return self._output_prefix
    
    @output_prefix.setter
    def output_prefix(self, output_prefix: str):
        self._output_prefix = output_prefix
        self.init_kwargs['output_filename_prefix'] = output_prefix
    
    @property
    def out_path_and_prefix(self):
        return self._output_directory, self._output_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix):
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.output_directory = out_dir
            self.output_prefix = out_prefix
            
    def infer_prefix_from_input_tac_path(self):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_tac_path)
        self.output_prefix = f'sub-{sub_id}_ses-{ses_id}'
        
    def infer_output_directory_from_input_tac_path(self, out_dir:str, der_type:str='km'):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_tac_path)
        outpath = gen_bids_like_dir_path(sub_id=sub_id,
                                         ses_id=ses_id,
                                         modality=der_type,
                                         sup_dir=out_dir)
        self.output_directory = outpath
        
    def infer_outputs_from_inputs(self, out_dir: str, der_type, suffix=None, ext=None, **extra_desc):
        self.infer_prefix_from_input_tac_path()
        self.infer_output_directory_from_input_tac_path(out_dir=out_dir, der_type=der_type)
        
    def set_input_as_output_from(self, sending_step: PreprocSteps) -> None:
        if isinstance(sending_step, TACsFromSegmentationStep):
            self.roi_tacs_dir = sending_step.out_tacs_dir
        elif isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        else:
            super().set_input_as_output_from(sending_step)


class GraphicalAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0,
                 image_rescale: float = 1.0/37000.0):
        TACAnalysisStepMixin.__init__(self,
                                      input_tac_path=input_tac_path,
                                      roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=False,
                                      method=method,
                                      fit_thresh_in_mins = fit_threshold_in_mins,
                                      )
        ObjectBasedStep.__init__(self, name=f'roi_{method}_fit',
                                 class_type=pet_grph.MultiTACGraphicalAnalysis,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict())
    
    @classmethod
    def default_patlak(cls):
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', method='patlak', )
    
    @classmethod
    def default_logan(cls):
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', method='logan', )
    
    @classmethod
    def default_alt_logan(cls):
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='', output_prefix='', method='alt_logan', )
        
        
class TCMFittingAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 input_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 compartment_model='2tcm-k4zer0',
                 **kwargs):
        TACAnalysisStepMixin.__init__(self,
                                      input_tac_path=input_tac_path,
                                      roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=False,
                                      compartment_model=compartment_model,
                                      **kwargs)
        
        ObjectBasedStep.__init__(self, name=f'roi_{compartment_model}_fit',
                                 class_type=tac_fitting.FitTCMToManyTACs,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict())
        
        
    @classmethod
    def default_1tcm(cls, **kwargs):
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='',
                   output_prefix='', compartment_model='1tcm', **kwargs)
    
    @classmethod
    def default_serial2tcm(cls, **kwargs):
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='',
                   output_prefix='', compartment_model='serial-2tcm', **kwargs)
    
    @classmethod
    def default_irreversible_2tcm(cls, **kwargs):
        return cls(input_tac_path='', roi_tacs_dir='', output_directory='',
                   output_prefix='', compartment_model='2tcm-k4zero', **kwargs)
        

class RTMFittingAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 ref_tac_path: str,
                 roi_tacs_dir: str,
                 output_directory: str,
                 output_prefix: str,
                 rtm_model: str,
                 bounds = None,
                 k2_prime = None,
                 fit_threshold_in_mins: float = 30.0,):
        
        TACAnalysisStepMixin.__init__(self,
                                      input_tac_path=ref_tac_path,
                                      roi_tacs_dir=roi_tacs_dir,
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=True,
                                      method=rtm_model, )
        ObjectBasedStep.__init__(self, name=f'roi_{rtm_model}_fit',
                                 class_type=pet_rtms.MultiTACRTMAnalysis,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict(bounds=bounds,
                                                  t_thresh_in_mins=fit_threshold_in_mins,
                                                  k2_prime=k2_prime))


class ParametricGraphicalAnalysisStep(ObjectBasedStep, TACAnalysisStepMixin):
    def __init__(self,
                 input_tac_path: str,
                 input_image_path: str,
                 output_directory: str,
                 output_prefix: str,
                 method: str,
                 fit_threshold_in_mins: float = 30.0, ):
        TACAnalysisStepMixin.__init__(self, input_tac_path=input_tac_path,
                                      pet4D_img_path=input_image_path,
                                      roi_tacs_dir='',
                                      output_directory=output_directory,
                                      output_prefix=output_prefix,
                                      is_ref_tac_based_model=False,
                                      )
        del self.init_kwargs['roi_tacs_dir']
        
        
        ObjectBasedStep.__init__(self, name=f'parametric_{method}_fit',
                                 class_type=parametric_images.GraphicalAnalysisParametricImage,
                                 init_kwargs=self.init_kwargs,
                                 call_kwargs=dict(method_name=method,
                                                  t_thresh_in_mins=fit_threshold_in_mins,))
        self._input_image_path = input_image_path
    
    
    @property
    def input_image_path(self) -> str:
        return self._input_image_path
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        self._input_image_path = input_image_path
        self.init_kwargs['pet4D_img_path'] = input_image_path
    
    def set_input_as_output_from(self, sending_step: PreprocSteps) -> None:
        if isinstance(sending_step, ResampleBloodTACStep):
            self.input_tac_path = sending_step.resampled_tac_path
        elif isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
            
    @classmethod
    def default_patlak(cls):
        return cls(input_tac_path='', input_image_path='', output_directory='', output_prefix='', method='patlak')
    
    @classmethod
    def default_logan(cls):
        return cls(input_tac_path='', input_image_path='', output_directory='', output_prefix='', method='logan')
    
    @classmethod
    def default_alt_logan(cls):
        return cls(input_tac_path='', input_image_path='', output_directory='', output_prefix='', method='alt_logan')


KMStepType = Union[GraphicalAnalysisStep,
                   TCMFittingAnalysisStep,
                   ParametricGraphicalAnalysisStep,
                   RTMFittingAnalysisStep]

StepType = Union[FunctionBasedStep, ObjectBasedStep,
                 PreprocSteps, KMStepType]


class StepsContainer:
    def __init__(self, name: str, *steps: StepType):
        self.name = name
        self.step_objs: list[StepType] = []
        self.step_names: list[str] = []
        for step in steps:
            self.add_step(step)
        
    def add_step(self, step: StepType):
        if not isinstance(step, StepType.__args__):
            raise TypeError("Step must be of type StepType")
        
        if step.name not in self.step_names:
            self.step_objs.append(copy.deepcopy(step))
            self.step_names.append(self.step_objs[-1].name)
        else:
            raise KeyError("A step with this name already exists.")
        
    def remove_step(self, step: Union[int, str]):
        if isinstance(step, int):
            try:
                del self.step_objs[step]
                del self.step_names[step]
            except IndexError:
                raise IndexError(f"Step number {step} does not exist.")
        elif isinstance(step, str):
            if step not in self.step_names:
                raise KeyError(f"Step name {step} does not exist.")
            try:
                step_index = self.step_names.index(step)
                del self.step_objs[step_index]
                del self.step_names[step_index]
            except Exception:
                raise Exception
                
    def print_step_details(self):
        if not self.step_objs:
            print("No steps in container.")
        else:
            print(f"({self.name} Pipeline Info):")
            for step_id, a_step in enumerate(self.step_objs):
                print('-' * 80)
                print(f"Step Number {step_id + 1}")
                print(a_step)
                print('-' * 80)
            print("*" * 90)
            
    def print_step_names(self) -> None:
        if not self.step_objs:
            print("No steps in container.")
        else:
            print(f"({self.name} pipeline info):")
            print('-' * 80)
            for step_id, step_name in enumerate(self.step_names):
                print(f"Step Number {step_id+1}: {step_name}")
            print('-' * 80)
            
    def __call__(self):
        for step_id, (step_name, a_step) in enumerate(zip(self.step_names, self.step_objs)):
            a_step.execute()
            
    def __getitem__(self, step: Union[int, str]):
        if isinstance(step, int):
            try:
                return self.step_objs[step]
            except IndexError:
                raise IndexError(f"Step number {step} does not exist.")
        elif isinstance(step, str):
            if step not in self.step_names:
                raise KeyError(f"Step name {step} does not exist.")
            try:
                step_index = self.step_names.index(step)
                return self.step_objs[step_index]
            except KeyError:
                raise KeyError(f"Step name {step} does not exist.")
        else:
            raise TypeError(f"Key must be an integer or a string. Got {type(step)}")
    
    def __add__(self, other: 'StepsContainer') -> 'StepsContainer':
        if isinstance(other, StepsContainer):
            new_container_name = f"{self.name}-{other.name}"
            new_container = StepsContainer(new_container_name)
            for step in self.step_objs:
                new_container.add_step(step)
            for step in other.step_objs:
                new_container.add_step(step)
            return new_container
        else:
            raise TypeError("Can only add another StepsContainer instance")
        
    @classmethod
    def default_preprocess_steps(cls, name:str='preproc'):
        obj = cls(name=name)
        obj.add_step(ImageToImageStep.default_threshold_cropping())
        obj.add_step(ImageToImageStep.default_moco_frames_above_mean())
        obj.add_step(ImageToImageStep.default_register_pet_to_t1())
        obj.add_step(TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois())
        obj.add_step(ResampleBloodTACStep.default_resample_blood_tac_on_scanner_times())
        return obj
    
    @classmethod
    def default_graphical_analysis_steps(cls, name: str='km_graphical_analysis'):
        obj = cls(name=name)
        obj.add_step(GraphicalAnalysisStep.default_patlak())
        obj.add_step(GraphicalAnalysisStep.default_logan())
        obj.add_step(GraphicalAnalysisStep.default_alt_logan())
        return obj
    
    @classmethod
    def default_parametric_graphical_analysis_steps(cls, name: str='km_parametric_graphical_analysis'):
        obj = cls(name=name)
        obj.add_step(ParametricGraphicalAnalysisStep.default_patlak())
        obj.add_step(ParametricGraphicalAnalysisStep.default_logan())
        obj.add_step(ParametricGraphicalAnalysisStep.default_alt_logan())
        return obj
    
    @classmethod
    def default_tcm_analysis_steps(cls, name: str='km_tcm_analysis'):
        obj = cls(name=name)
        obj.add_step(TCMFittingAnalysisStep.default_1tcm())
        obj.add_step(TCMFittingAnalysisStep.default_serial2tcm())
        obj.add_step(TCMFittingAnalysisStep.default_irreversible_2tcm())
        return obj
    
    @classmethod
    def default_kinetic_analysis_steps(cls, name: str='km'):
        parametric_graphical_analysis_steps = cls.default_parametric_graphical_analysis_steps()
        graphical_analysis_steps = cls.default_graphical_analysis_steps()
        tcm_analysis_steps = cls.default_tcm_analysis_steps()
        
        obj = parametric_graphical_analysis_steps + graphical_analysis_steps + tcm_analysis_steps
        obj.name = name
        return obj
    
    
class StepsPipeline:
    def __init__(self, name: str, *step_containers):
        self.name: str = name
        self.step_containers: dict[str, StepsContainer] = {}
        self.dependency_graph = nx.DiGraph()
        
        for container in step_containers:
            self.add_container(container)
        
    def add_container(self, step_container: StepsContainer):
        if not isinstance(step_container, StepsContainer):
            raise TypeError("`step_container` must be an instance of StepsContainer")
        
        container_name = step_container.name
        if container_name in self.step_containers.keys():
            raise KeyError(f"Container name {container_name} already exists.")
        
        self.step_containers[container_name] = copy.deepcopy(step_container)
        
        step_objs = self.step_containers[container_name].step_objs
        for step in step_objs:
            self.dependency_graph.add_node(f"{step.name}", grp=container_name)
        
    def __call__(self):
        for name, container in self.step_containers.items():
            container()
    
    def add_step(self, container_name: str, step: StepType):
        if container_name not in self.step_containers.keys():
            raise KeyError(f"Container name {container_name} does not exist.")
        if step.name in self.dependency_graph:
            raise KeyError(f"Step name {step.name} already exists. Steps must have unique names.")
        
        self.step_containers[container_name].add_step(step)
        self.dependency_graph.add_node(f"{step.name}", grp=container_name)
    
    def remove_step(self, step: str):
        node_names = list(self.dependency_graph.nodes)
        if step not in node_names:
            raise KeyError(f"Step name {step} does not exist.")
        graph_nodes = self.dependency_graph.nodes(data=True)
        container_name = graph_nodes[step]['grp']
        self.dependency_graph.remove_node(step)
        self.step_containers[container_name].remove_step(step)
    
    def print_steps_names(self, container_name: Union[str, None] = None):
        if container_name is None:
            for name, container in self.step_containers.items():
                container.print_step_names()
        elif container_name in self.step_containers.keys():
            self.step_containers[container_name].print_step_names()
        else:
            raise KeyError(f"Container name {container_name} does not exist. ")
        
    def print_steps_details(self, container_name: Union[str, None] = None):
        if container_name is None:
            for name, container in self.step_containers.items():
                container.print_step_details()
        elif container_name in self.step_containers.keys():
            self.step_containers[container_name].print_step_details()
        else:
            raise KeyError(f"Container name {container_name} does not exist. ")
        
    def add_dependency(self, sending: str, receiving: str):
        node_names = list(self.dependency_graph.nodes)
        if sending not in node_names:
            raise KeyError(f"Step name {sending} does not exist.")
        if receiving not in node_names:
            raise KeyError(f"Step name {receiving} does not exist.")
        
        self.dependency_graph.add_edge(sending, receiving)
        
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            raise RuntimeError(f"Adding dependency {sending} -> {receiving} creates a cycle!")
        
    def get_step_from_node_label(self, node_label: str):
        if node_label not in self.dependency_graph.nodes:
            raise KeyError(f"Step name {node_label} does not exist.")
        graph_nodes = self.dependency_graph.nodes(data=True)
        container_name = graph_nodes[node_label]['grp']
        return self.step_containers[container_name][node_label]
    
    def update_dependencies_for(self, step_name: str, verbose=False):
        sending_step = self.get_step_from_node_label(step_name)
        for an_edge in self.dependency_graph[step_name]:
            receiving_step = self.get_step_from_node_label(an_edge)
            try:
                receiving_step.set_input_as_output_from(sending_step)
            except NotImplementedError:
                if verbose:
                    print(f"Step `{receiving_step.name}` of type `{type(receiving_step).__name__}` does not have a "
                              f"set_input_as_output_from method implemented.\nSkipping.")
            else:
                if verbose:
                    print(f"Updated input-output dependency between {sending_step.name} and {receiving_step.name}")
    
    def update_dependencies(self, verbose=False):
        for step_name in nx.topological_sort(self.dependency_graph):
            self.update_dependencies_for(step_name=step_name, verbose=verbose)

    def get_steps_potential_run_state(self) -> dict:
        step_maybe_runnable = {}
        for node_name in nx.topological_sort(self.dependency_graph):
            step = self.get_step_from_node_label(node_name)
            step_maybe_runnable[node_name] = step.can_potentially_run()
        return step_maybe_runnable

    def can_steps_potentially_run(self) -> bool:
        steps_run_state = self.get_steps_potential_run_state()
        for step_name, run_state in steps_run_state.items():
            if not run_state:
                return False
        return True
    
    @classmethod
    def default_steps_pipeline(cls, name='PET-MR_analysis'):
        obj = cls(name=name)
        obj.add_container(StepsContainer.default_preprocess_steps(name='preproc'))
        obj.add_container(StepsContainer.default_kinetic_analysis_steps(name='km'))
        
        obj.add_dependency(sending='thresh_crop', receiving='moco_frames_above_mean')
        obj.add_dependency(sending='moco_frames_above_mean', receiving='register_pet_to_t1')
        obj.add_dependency(sending='register_pet_to_t1', receiving='write_roi_tacs')
        obj.add_dependency(sending='register_pet_to_t1', receiving='resample_PTAC_on_scanner')
        
        for method in ['patlak', 'logan', 'alt_logan']:
            obj.add_dependency(sending='register_pet_to_t1', receiving=f'parametric_{method}_fit')
            obj.add_dependency(sending='resample_PTAC_on_scanner', receiving=f'parametric_{method}_fit')
        
        for fit_model in ['1tcm', '2tcm-k4zero', 'serial-2tcm', 'patlak', 'logan', 'alt_logan']:
            obj.add_dependency(sending='write_roi_tacs', receiving=f"roi_{fit_model}_fit")
            obj.add_dependency(sending='resample_PTAC_on_scanner', receiving=f"roi_{fit_model}_fit")
        
        return obj
        

class BIDSyPathsForRawData:
    def __init__(self,
                 sub_id: str,
                 ses_id: str,
                 bids_root_dir: str = None,
                 derivatives_dir: str = None,
                 raw_pet_img_path: str = None,
                 raw_anat_img_path: str = None,
                 segmentation_img_path: str = None,
                 segmentation_label_table_path: str = None,
                 raw_blood_tac_path: str = None,):
        self.sub_id = sub_id
        self.ses_id = ses_id
        self._bids_dir = bids_root_dir
        self._derivatives_dir = derivatives_dir
        self._raw_pet_path = raw_pet_img_path
        self._raw_anat_path = raw_anat_img_path
        self._segmentation_img_path = segmentation_img_path
        self._segmentation_label_table_path = segmentation_label_table_path
        self._raw_blood_tac_path = raw_blood_tac_path
        
        self.bids_dir = self._bids_dir
        self.derivatives_dir = self._derivatives_dir
        self.pet_path = self._raw_pet_path
        self.anat_path = self._raw_anat_path
        self.seg_img = self._segmentation_img_path
        self.seg_table = self._segmentation_label_table_path
        self.blood_tac = self._raw_blood_tac_path
    
    @property
    def bids_dir(self):
        return self._bids_dir
    
    @bids_dir.setter
    def bids_dir(self, value):
        if value is None:
            self._bids_dir = os.path.abspath('../')
        else:
            val_path = pathlib.Path(value)
            if val_path.is_dir():
                self._bids_dir = os.path.abspath(value)
            else:
                raise ValueError("Given BIDS path is not a directory.")
            
    @property
    def derivatives_dir(self):
        return self._derivatives_dir
    
    @derivatives_dir.setter
    def derivatives_dir(self, value):
        if value is None:
            self._derivatives_dir = os.path.abspath(os.path.join(self.bids_dir, 'derivatives'))
        else:
            val_path = pathlib.Path(value).absolute()
            if val_path.is_relative_to(self.bids_dir):
                self._derivatives_dir = os.path.abspath(value)
            else:
                raise ValueError(f"Given derivatives path is not a sub-directory of BIDS path."
                                 f"\nBIDS:       {self.bids_dir}"
                                 f"\nDerivatives:{os.path.abspath(value)}")
    
    
    @property
    def pet_path(self):
        return self._raw_pet_path
    
    @pet_path.setter
    def pet_path(self, value: str):
        if value is None:
            filepath = gen_bids_like_filepath(sub_id=self.sub_id,
                                              ses_id=self.ses_id,
                                              modality='pet',
                                              bids_dir=self.bids_dir,
                                              ext='.nii.gz')
            self._raw_pet_path = filepath
        else:
            val_path = pathlib.Path(value)
            val_suff = "".join(val_path.suffixes)
            if val_path.is_file() and val_suff == '.nii.gz':
                self._raw_pet_path = value
            else:
                raise FileNotFoundError(f"File does not exist: {value}")
            
    @property
    def anat_path(self):
        return self._raw_anat_path
    
    @anat_path.setter
    def anat_path(self, value: str):
        if value is None:
            filepath = gen_bids_like_filepath(sub_id=self.sub_id, ses_id=self.ses_id,
                                              modality='anat', suffix='MPRAGE',
                                              bids_dir=self.bids_dir, ext='.nii.gz')
            self._raw_anat_path =  filepath
        else:
            val_path = pathlib.Path(value)
            val_suff = "".join(val_path.suffixes)
            if val_path.is_file() and val_suff == '.nii.gz':
                self._raw_anat_path = value
            else:
                raise FileNotFoundError(f"File does not exist: {value}")
            
    @property
    def seg_img(self):
        return self._segmentation_img_path
    
    @seg_img.setter
    def seg_img(self, value: str):
        if value is None:
            seg_dir = os.path.join(self.derivatives_dir, 'ROI_mask')
            filepath = gen_bids_like_filepath(sub_id=self.sub_id, ses_id=self.ses_id,
                                              modality='anat', bids_dir=seg_dir,
                                              suffix='ROImask',
                                              ext='.nii.gz',
                                              space='MPRAGE',
                                              desc='lesionsincluded')
            self._segmentation_img_path = filepath
        else:
            if os.path.isfile(value):
                self._segmentation_img_path = value
            else:
                raise FileNotFoundError(f"File does not exist: {value}")
            
    @property
    def seg_table(self):
        return self._segmentation_label_table_path
    
    @seg_table.setter
    def seg_table(self, value: str):
        if value is None:
            seg_dir = os.path.join(self.derivatives_dir, 'ROI_mask')
            filename = 'dseg_forCMMS.tsv'
            self._segmentation_label_table_path = os.path.join(seg_dir, filename)
        else:
            val_path = pathlib.Path(value)
            if val_path.is_file() and (val_path.suffix == '.tsv'):
                self._segmentation_label_table_path = value
            else:
                raise FileNotFoundError(f"File does not exist: {value}")
            
    @property
    def blood_tac(self):
        return self._raw_blood_tac_path
    
    @blood_tac.setter
    def blood_tac(self, value: str):
        if value is None:
            filepath = gen_bids_like_filepath(sub_id=self.sub_id,
                                              ses_id=self.ses_id,
                                              bids_dir=self.bids_dir,
                                              modality='pet',
                                              suffix='blood',
                                              ext='.tsv',
                                              desc='decaycorrected'
                                              )
            self._raw_blood_tac_path = filepath
        else:
            val_path = pathlib.Path(value)
            if val_path.is_file() and (val_path.suffix == '.tsv'):
                self._raw_blood_tac = value
            else:
                raise FileNotFoundError(f"File does not exist: {value}")
                
                
class BIDSyPathsForPipelines(BIDSyPathsForRawData):
    def __init__(self,
                 sub_id: str,
                 ses_id: str,
                 pipeline_name: str ='generic_pipeline',
                 list_of_analysis_dir_names: Union[None, list[str]] = None,
                 bids_root_dir: str = None,
                 derivatives_dir: str = None,
                 raw_pet_img_path: str = None,
                 raw_anat_img_path: str = None,
                 segmentation_img_path: str = None,
                 segmentation_label_table_path: str = None,
                 raw_blood_tac_path: str = None):
        super().__init__(sub_id=sub_id,
                         ses_id=ses_id,
                         bids_root_dir=bids_root_dir,
                         derivatives_dir=derivatives_dir,
                         raw_pet_img_path=raw_pet_img_path,
                         raw_anat_img_path=raw_anat_img_path,
                         segmentation_img_path=segmentation_img_path,
                         segmentation_label_table_path=segmentation_label_table_path,
                         raw_blood_tac_path=raw_blood_tac_path)
        
        self._pipeline_dir = None
        self.pipeline_name = pipeline_name
        self.pipeline_dir = self._pipeline_dir
        self.analysis_dirs = self.generate_analysis_dirs(list_of_dir_names=list_of_analysis_dir_names)
        self.make_analysis_dirs()
        
        
    @property
    def pipeline_dir(self):
        return self._pipeline_dir
    
    @pipeline_dir.setter
    def pipeline_dir(self, value: str):
        if value is None:
            default_path = os.path.join(self.derivatives_dir, 'petpal', 'pipelines', self.pipeline_name)
            self._pipeline_dir = os.path.abspath(default_path)
        else:
            pipe_path = pathlib.Path(value).absolute()
            if pipe_path.is_relative_to(self.derivatives_dir):
                self._pipeline_dir = os.path.abspath(value)
            else:
                raise ValueError("Pipeline directory is not relative to the derivatives directory")
    
    def generate_analysis_dirs(self, list_of_dir_names: Union[None, list[str]] = None) -> dict:
        if list_of_dir_names is None:
            list_of_dir_names = ['preproc', 'km', 'tacs']
        path_gen = lambda name: gen_bids_like_dir_path(sub_id=self.sub_id,
                                                       ses_id=self.ses_id,
                                                       modality=name,
                                                       sup_dir=self.pipeline_dir)
        analysis_dirs = {name:path_gen(name) for name in list_of_dir_names}
        return analysis_dirs
    
    def make_analysis_dirs(self):
        for a_name, a_dir in self.analysis_dirs.items():
            os.makedirs(a_dir, exist_ok=True)
            
    
class BIDS_Pipeline(BIDSyPathsForPipelines, StepsPipeline):
    def __init__(self,
                 sub_id: str,
                 ses_id: str,
                 pipeline_name: str = 'generic_pipeline',
                 list_of_analysis_dir_names: Union[None, list[str]] = None,
                 bids_root_dir: str = None,
                 derivatives_dir: str = None,
                 raw_pet_img_path: str = None,
                 raw_anat_img_path: str = None,
                 segmentation_img_path: str = None,
                 segmentation_label_table_path: str = None,
                 raw_blood_tac_path: str = None):
        BIDSyPathsForPipelines.__init__(self,
                                        sub_id=sub_id,
                                        ses_id=ses_id,
                                        pipeline_name=pipeline_name,
                                        list_of_analysis_dir_names=list_of_analysis_dir_names,
                                        bids_root_dir=bids_root_dir,
                                        derivatives_dir=derivatives_dir,
                                        raw_pet_img_path=raw_pet_img_path,
                                        raw_anat_img_path=raw_anat_img_path,
                                        segmentation_img_path=segmentation_img_path,
                                        segmentation_label_table_path=segmentation_label_table_path,
                                        raw_blood_tac_path=raw_blood_tac_path)
        StepsPipeline.__init__(self, name=pipeline_name)
        
    def update_dependencies_for(self, step_name, verbose=False):
        sending_step = self.get_step_from_node_label(step_name)
        sending_step_grp_name = self.dependency_graph.nodes(data=True)[step_name]['grp']
        sending_step.infer_outputs_from_inputs(out_dir=self.pipeline_dir,
                                               der_type=sending_step_grp_name,)
        for an_edge in self.dependency_graph[step_name]:
            receiving_step = self.get_step_from_node_label(an_edge)
            try:
                receiving_step.set_input_as_output_from(sending_step)
            except NotImplementedError:
                if verbose:
                    print(f"Step `{receiving_step.name}` of type `{type(receiving_step).__name__}` does not have a "
                          f"set_input_as_output_from method implemented.\nSkipping.")
                else:
                    if verbose:
                        print(f"Updated input-output dependency between {sending_step.name} and {receiving_step.name}")
    
    
    @classmethod
    def default_bids_pipeline(cls,
                              sub_id: str,
                              ses_id: str,
                              pipeline_name: str = 'generic_pipeline',
                              list_of_analysis_dir_names: Union[None, list[str]] = None,
                              bids_root_dir: str = None,
                              derivatives_dir: str = None,
                              raw_pet_img_path: str = None,
                              raw_anat_img_path: str = None,
                              segmentation_img_path: str = None,
                              segmentation_label_table_path: str = None,
                              raw_blood_tac_path: str = None):
        obj = cls(sub_id=sub_id,
                  ses_id=ses_id,
                  pipeline_name=pipeline_name,
                  list_of_analysis_dir_names=list_of_analysis_dir_names,
                  bids_root_dir=bids_root_dir,
                  derivatives_dir=derivatives_dir,
                  raw_pet_img_path=raw_pet_img_path,
                  raw_anat_img_path=raw_anat_img_path,
                  segmentation_img_path=segmentation_img_path,
                  segmentation_label_table_path=segmentation_label_table_path,
                  raw_blood_tac_path=raw_blood_tac_path)
        
        temp_pipeline = StepsPipeline.default_steps_pipeline()
        obj.step_containers = copy.deepcopy(temp_pipeline.step_containers)
        obj.dependency_graph = copy.deepcopy(temp_pipeline.dependency_graph)
        
        containers = obj.step_containers
        
        containers["preproc"][0].input_image_path = obj.pet_path
        containers["preproc"][1].kwargs['half_life'] = get_half_life_from_nifty(obj.pet_path)
        containers["preproc"][2].kwargs['reference_image_path'] = obj.anat_path
        containers["preproc"][2].kwargs['half_life'] = get_half_life_from_nifty(obj.pet_path)
        containers["preproc"][3].segmentation_label_map_path = obj.seg_table
        containers["preproc"][3].segmentation_image_path = obj.seg_img
        containers["preproc"][4].raw_blood_tac_path = obj.blood_tac
        
        obj.update_dependencies(verbose=False)
        return obj
