import warnings
import copy
from typing import Union
from .steps_base import *
from ..preproc.image_operations_4d import SimpleAutoImageCropper, write_tacs
from ..preproc.register import register_pet
from ..preproc.motion_corr import motion_corr_frames_above_mean_value
from ..input_function import blood_input
from ..utils.bids_utils import parse_path_to_get_subject_and_session_id, snake_to_camel_case, gen_bids_like_dir_path, gen_bids_like_filepath
from ..utils.image_io import safe_copy_meta

class TACsFromSegmentationStep(FunctionBasedStep):
    """
    A step in a processing pipeline for generating Time Activity Curves (TACs) from segmented images.

    This class is specialized for handling the input and output paths related to TAC generation,
    extending the :class:`FunctionBasedStep<petpal.pipelines.steps_base.FunctionBasedStep>` with specific properties
    and methods for TACs. The class uses :func:`write_tacs<petpal.preproc.image_operations_4d.write_tacs>` which
    uses segmentation information to generate ROI TACs, and write them to disk.

    Attributes:
        input_image_path (str): Path to the input image.
        segmentation_image_path (str): Path to the segmentation image.
        segmentation_label_map_path (str): Path to the segmentation label map.
        out_tacs_dir (str): Directory where the output TACs will be saved.
        out_tacs_prefix (str): Prefix for the output TACs.
        time_keyword (str): Keyword for the time frame, default is 'FrameReferenceTime'.
        verbose (bool): Verbosity flag, default is False.

    Methods:
        - set_input_as_output_from(sending_step): Sets the input image path based on the output from
            a specified sending step.
        - infer_outputs_from_inputs(out_dir, der_type, suffix=None, ext=None, **extra_desc): Infers
            output directory and prefix for TACs based on the input image path.
        - default_write_tacs_from_segmentation_rois(): Provides a class method to create an instance
            with default parameters.
        
    """
    def __init__(self,
                 input_image_path: str,
                 segmentation_image_path: str,
                 segmentation_label_map_path: str,
                 out_tacs_dir: str,
                 out_tacs_prefix: str,
                 time_keyword='FrameReferenceTime',
                 verbose=False) -> None:
        """
        Initializes a TACsFromSegmentationStep with specified parameters.

        Args:
            input_image_path (str): Path to the input image.
            segmentation_image_path (str): Path to the segmentation image.
            segmentation_label_map_path (str): Path to the segmentation label map.
            out_tacs_dir (str): Directory where the output TACs will be saved.
            out_tacs_prefix (str): Prefix for the output TACs.
            time_keyword (str): Keyword for the time frame, default is 'FrameReferenceTime'.
            verbose (bool): Verbosity flag, default is False.
        """
        super().__init__(name='write_roi_tacs', function=write_tacs, input_image_path=input_image_path,
                         segmentation_image_path=segmentation_image_path, label_map_path=segmentation_label_map_path,
                         out_tac_dir=out_tacs_dir, out_tac_prefix=out_tacs_prefix, time_frame_keyword=time_keyword,
                         verbose=verbose, )
        self._input_image = input_image_path
        self._segmentation_image = segmentation_image_path
        self._segmentation_label_map = segmentation_label_map_path
        self._out_tacs_path = out_tacs_dir
        self._out_tacs_prefix = out_tacs_prefix
        self.time_keyword = time_keyword
        self.verbose = verbose
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the TACsFromSegmentationStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(
            dict(input_image_path=self.input_image_path, segmentation_image_path=self.segmentation_image_path,
                 segmentation_label_map_path=self.segmentation_label_map_path, out_tacs_dir=self.out_tacs_dir,
                 out_tacs_prefix=self.out_tacs_prefix, time_keyword=self.time_keyword, verbose=self.verbose))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    @property
    def segmentation_image_path(self):
        """
        Gets the path to the segmentation image.

        Returns:
            str: The path to the segmentation image.
        """
        return self._segmentation_image
    
    @segmentation_image_path.setter
    def segmentation_image_path(self, segmentation_image_path: str):
        """
        Sets the path to the segmentation image and updates the function arguments.

        Args:
            segmentation_image_path (str): The new path to the segmentation image.
        """
        self._segmentation_image = segmentation_image_path
        self.kwargs['segmentation_image_path'] = segmentation_image_path
    
    @property
    def segmentation_label_map_path(self):
        """
        Gets the path to the segmentation label map.

        Returns:
            str: The path to the segmentation label map.
        """
        return self._segmentation_label_map
    
    @segmentation_label_map_path.setter
    def segmentation_label_map_path(self, segmentation_label_map_path: str):
        """
        Sets the path to the segmentation label map and updates the function arguments.

        Args:
            segmentation_label_map_path (str): The new path to the segmentation label map.
        """
        self._segmentation_label_map = segmentation_label_map_path
        self.kwargs['label_map_path'] = segmentation_label_map_path
    
    @property
    def out_tacs_dir(self):
        """
        Gets the directory where the output TACs will be saved.

        Returns:
            str: The output directory path.
        """
        return self._out_tacs_path
    
    @out_tacs_dir.setter
    def out_tacs_dir(self, out_tacs_path: str):
        """
        Sets the directory where the output TACs will be saved and updates the function arguments.

        Args:
            out_tacs_path (str): The new output directory path.
        """
        self.kwargs['out_tac_dir'] = out_tacs_path
        self._out_tacs_path = out_tacs_path
    
    @property
    def out_tacs_prefix(self):
        """
        Gets the prefix for the output TACs.

        Returns:
            str: The prefix for the output TACs.
        """
        return self._out_tacs_prefix
    
    @out_tacs_prefix.setter
    def out_tacs_prefix(self, out_tacs_prefix: str):
        """
        Sets the prefix for the output TACs and updates the function arguments.

        Args:
            out_tacs_prefix (str): The new prefix for the output TACs.
        """
        self.kwargs['out_tac_prefix'] = out_tacs_prefix
        self._out_tacs_prefix = out_tacs_prefix
    
    @property
    def out_path_and_prefix(self):
        """
        Gets the output directory path and prefix as a tuple.

        Returns:
            tuple: A tuple containing the output directory path and prefix.
        """
        return self._out_tacs_path, self._out_tacs_prefix
    
    @out_path_and_prefix.setter
    def out_path_and_prefix(self, out_dir_and_prefix: str):
        """
        Sets both the output directory path and prefix.

        Args:
            out_dir_and_prefix (tuple): A tuple containing the output directory and prefix.

        Raises:
            ValueError: If the provided value is not a tuple with two items.
        """
        try:
            out_dir, out_prefix = out_dir_and_prefix
        except ValueError:
            raise ValueError("Pass a tuple with two items: `(out_dir, out_prefix)`")
        else:
            self.out_tacs_dir = out_dir
            self.out_tacs_prefix = out_prefix
    
    @property
    def input_image_path(self):
        """
        Gets the path to the input image.

        Returns:
            str: The path to the input image.
        """
        return self._input_image
    
    @input_image_path.setter
    def input_image_path(self, input_image_path: str):
        """
        Sets the path to the input image and updates the function arguments.

        Args:
            input_image_path (str): The new path to the input image.
        """
        self.kwargs['input_image_path'] = input_image_path
        self._input_image = input_image_path
    
    def set_input_as_output_from(self, sending_step):
        """
        Sets the input image path based on the output from a specified sending step.

        Args:
            sending_step: The step from which to derive the input image path.
        """
        if isinstance(sending_step, ImageToImageStep):
            self.input_image_path = sending_step.output_image_path
        else:
            super().set_input_as_output_from(sending_step)
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type: str, suffix: str=None, ext: str=None, **extra_desc):
        """
        Infers output directory and prefix for TACs based on the input image path.

        Args:
            out_dir (str): Directory where the outputs will be saved.
            der_type (str): Type of derivatives.
            suffix (str, optional): Suffix for the output files. Defaults to None.
            ext (str, optional): Extension for the output files. Defaults to None.
            **extra_desc: Additional descriptive parameters.
        """
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        outpath = gen_bids_like_dir_path(sub_id=sub_id, ses_id=ses_id, sup_dir=out_dir, modality='tacs')
        self.out_tacs_dir = outpath
        step_name_in_camel_case = snake_to_camel_case(self.name)
        self.out_tacs_prefix = f'sub-{sub_id}_ses-{ses_id}_desc-{step_name_in_camel_case}'
    
    @classmethod
    def default_write_tacs_from_segmentation_rois(cls):
        """
        Provides a class method to create an instance with default parameters. All paths
        are set to empty strings, `time_keyword=FrameReferenceTime`, and `verbose=False`.

        Returns:
            TACsFromSegmentationStep: A new instance with default parameters.
        """
        return cls(input_image_path='',
                   segmentation_image_path='',
                   segmentation_label_map_path='',
                   out_tacs_dir='',
                   out_tacs_prefix='',
                   time_keyword='FrameReferenceTime',
                   verbose=False)
    

class ResampleBloodTACStep(FunctionBasedStep):
    def __init__(self,
                 input_raw_blood_tac_path: str,
                 input_image_path: str,
                 out_tac_path: str,
                 lin_fit_thresh_in_mins=30.0):
        super().__init__(name='resample_PTAC_on_scanner', function=blood_input.resample_blood_data_on_scanner_times,
                         raw_blood_tac=input_raw_blood_tac_path, pet4d_path=input_image_path, out_tac_path=out_tac_path,
                         lin_fit_thresh_in_mins=lin_fit_thresh_in_mins)
        self._raw_blood_tac_path = input_raw_blood_tac_path
        self._input_image_path = input_image_path
        self._resampled_tac_path = out_tac_path
        self.lin_fit_thresh_in_mins = lin_fit_thresh_in_mins
    
    def __repr__(self):
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(']
        
        in_kwargs = ArgsDict(
                dict(input_raw_blood_tac_path=self.raw_blood_tac_path, input_image_path=self.input_image_path,
                     out_tac_path=self.resampled_tac_path, lin_fit_thresh_in_mins=self.lin_fit_thresh_in_mins))
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
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
        return cls(input_raw_blood_tac_path='', input_image_path='', out_tac_path='', lin_fit_thresh_in_mins=30.0)
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type, suffix='blood', ext='.tsv', **extra_desc):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.raw_blood_tac_path)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, bids_dir=out_dir, modality='preproc',
                                          suffix=suffix, ext=ext, desc='OnScannerFrameTimes')
        self.resampled_tac_path = filepath


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
            safe_copy_meta(input_image_path=self.input_image_path, out_image_path=self.output_image_path)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        io_dict = ArgsDict({
                               'input_image_path': self.input_image_path, 'output_image_path': self.output_image_path
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
                                  out_dir: str,
                                  der_type='preproc',
                                  suffix: str = 'pet',
                                  ext: str = '.nii.gz',
                                  **extra_desc):
        sub_id, ses_id = parse_path_to_get_subject_and_session_id(self.input_image_path)
        step_name_in_camel_case = snake_to_camel_case(self.name)
        filepath = gen_bids_like_filepath(sub_id=sub_id, ses_id=ses_id, suffix=suffix, bids_dir=out_dir,
                                          modality=der_type, ext=ext, desc=step_name_in_camel_case, **extra_desc)
        self.output_image_path = filepath
    
    @classmethod
    def default_threshold_cropping(cls, **overrides):
        defaults = dict(name='thresh_crop', function=SimpleAutoImageCropper, input_image_path='',
                        output_image_path='', )
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
    
    @classmethod
    def default_moco_frames_above_mean(cls, verbose=False, **overrides):
        defaults = dict(name='moco_frames_above_mean', function=motion_corr_frames_above_mean_value,
                        input_image_path='', output_image_path='', motion_target_option='mean_image', verbose=verbose,
                        half_life=None, )
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)
    
    @classmethod
    def default_register_pet_to_t1(cls, reference_image_path='', half_life='', verbose=False, **overrides):
        defaults = dict(name='register_pet_to_t1', function=register_pet, input_image_path='', output_image_path='',
                        reference_image_path=reference_image_path, motion_target_option='weighted_series_sum',
                        verbose=verbose, half_life=half_life, )
        override_dict = {**defaults, **overrides}
        try:
            return cls(**override_dict)
        except RuntimeError as err:
            warnings.warn(f"Invalid override: {err}. Using default instance instead.", stacklevel=2)
            return cls(**defaults)


PreprocStepType = Union[TACsFromSegmentationStep, ResampleBloodTACStep, ImageToImageStep]