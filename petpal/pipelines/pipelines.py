import pathlib
import os
import copy
from typing import Union
from .steps_base import *
from .steps_containers import StepsContainer, StepsPipeline
from ..utils.image_io import get_half_life_from_nifty
from ..utils.bids_utils import gen_bids_like_dir_path, gen_bids_like_filepath


class BIDSyPathsForRawData:
    """
    A class to manage and generate paths for raw BIDS data and its derivatives.

    This class handles the generation and validation of paths for different types of raw data
    (such as PET, anatomical images, and blood TAC files) and their derivatives, following the
    BIDS format.

    Attributes:
        sub_id (str): Subject ID.
        ses_id (str): Session ID.
        _bids_dir (Optional[str]): Root directory for BIDS data.
        _derivatives_dir (Optional[str]): Directory for derivative data.
        _raw_pet_path (Optional[str]): Path for raw PET images.
        _raw_anat_path (Optional[str]): Path for raw anatomical images.
        _segmentation_img_path (Optional[str]): Path for segmentation images.
        _segmentation_label_table_path (Optional[str]): Path for segmentation label tables.
        _raw_blood_tac_path (Optional[str]): Path for raw blood TAC files.
    """
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
        """
        Initializes the BIDSyPathsForRawData object with the given subject and session IDs,
        and paths for various types of data.

        Args:
            sub_id (str): Subject ID.
            ses_id (str): Session ID.
            bids_root_dir (Optional[str]): Optional path to the BIDS root directory. Defaults to None.
            derivatives_dir (Optional[str]): Optional path to the derivatives directory. Defaults to None.
            raw_pet_img_path (Optional[str]): Optional path to the raw PET image. Defaults to None.
            raw_anat_img_path (Optional[str]): Optional path to the raw anatomical image. Defaults to None.
            segmentation_img_path (Optional[str]): Optional path to the segmentation image. Defaults to None.
            segmentation_label_table_path (Optional[str]): Optional path to the segmentation label table. Defaults to None.
            raw_blood_tac_path (Optional[str]): Optional path to the raw blood TAC file. Defaults to None.
        """
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
        self.blood_path = self._raw_blood_tac_path
    
    @property
    def bids_dir(self) -> str:
        """
        str: Property for the BIDS root directory.
        """
        return self._bids_dir
    
    @bids_dir.setter
    def bids_dir(self, value):
        """
        Sets the BIDS root directory, validating if it is a directory.

        Args:
            value (Optional[str]): Path to the BIDS root directory.

        Raises:
            ValueError: If the given path is not a directory.
        """
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
        """
        str: Property for the derivatives directory.
        """
        return self._derivatives_dir
    
    @derivatives_dir.setter
    def derivatives_dir(self, value):
        """
        Sets the derivatives directory, validating if it is a sub-directory of the BIDS root directory.

        Args:
            value (Optional[str]): Path to the derivatives directory.

        Raises:
            ValueError: If the given path is not a sub-directory of the BIDS root directory.
        """
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
        """
        str: Property for the raw PET image path.
        """
        return self._raw_pet_path
    
    @pet_path.setter
    def pet_path(self, value: str):
        """
        Sets the raw PET image path, generating a path if None provided, and validating the file.

        Args:
            value (Optional[str]): Path to the raw PET image. If None, a path will be generated.

        Raises:
            FileNotFoundError: If the given file does not exist or does not have a .nii.gz extension.
        """
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
        """
        str: Property for the raw anatomical image path.
        """
        return self._raw_anat_path
    
    @anat_path.setter
    def anat_path(self, value: str):
        """
        Sets the raw anatomical image path, generating a path if None provided, and validating the file.

        Args:
            value (Optional[str]): Path to the raw anatomical image. If None, a path will be generated.

        Raises:
            FileNotFoundError: If the given file does not exist or does not have a .nii.gz extension.
        """
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
        """
        str: Property for the segmentation image path.
        """
        return self._segmentation_img_path
    
    @seg_img.setter
    def seg_img(self, value: str):
        """
        Sets the segmentation image path, generating a path if None provided, and validating the file.

        Args:
            value (Optional[str]): Path to the segmentation image. If None, a path will be generated.

        Raises:
            FileNotFoundError: If the given file does not exist.
        """
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
        """
        str: Property for the segmentation label table path.
        """
        return self._segmentation_label_table_path
    
    @seg_table.setter
    def seg_table(self, value: str):
        """
        Sets the segmentation label table path, generating a path if None provided, and validating the file.

        Args:
            value (Optional[str]): Path to the segmentation label table. If None, a path will be generated.

        Raises:
            FileNotFoundError: If the given file does not exist or does not have a .tsv extension.
        """
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
    def blood_path(self):
        """
        str: Property for the raw blood TAC path.
        """
        return self._raw_blood_tac_path
    
    @blood_path.setter
    def blood_path(self, value: str):
        """
        Sets the raw blood TAC path, generating a path if None provided, and validating the file.

        Args:
            value (Optional[str]): Path to the raw blood TAC file. If None, a path will be generated.

        Raises:
            FileNotFoundError: If the given file does not exist or does not have a .tsv extension.
        """
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
        self.list_of_analysis_dir_names = list_of_analysis_dir_names
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
                 raw_blood_tac_path: str = None,
                 step_containers: list[StepsContainer] = []):
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
        StepsPipeline.__init__(self, name=pipeline_name, step_containers=step_containers)
        

    def __repr__(self):
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(', ]
        
        in_kwargs = ArgsDict(dict(sub_id=self.sub_id,
                                  ses_id=self.ses_id,
                                  pipeline_name = self.name,
                                  list_of_analysis_dir_names = self.list_of_analysis_dir_names,
                                  bids_root_dir = self.bids_dir,
                                  derivatives_dir = self.derivatives_dir,
                                  raw_pet_img_path = self.pet_path,
                                  raw_anat_img_path = self.anat_path,
                                  segmentation_img_path = self.seg_img,
                                  segmentation_label_table_path = self.seg_table,
                                  raw_blood_tac_path = self.blood_path)
                )
        
        for arg_name, arg_val in in_kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append('step_containers=[')
        
        for _, container in self.step_containers.items():
            info_str.append(f'{repr(container)},')
        
        info_str.append(']')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
        
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
        
        temp_pipeline = StepsPipeline.default_steps_pipeline()
        
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
                  raw_blood_tac_path=raw_blood_tac_path,
                  step_containers=list(temp_pipeline.step_containers.values())
                  )
        
        obj.dependency_graph = copy.deepcopy(temp_pipeline.dependency_graph)
        
        del temp_pipeline
        
        containers = obj.step_containers
        
        containers["preproc"][0].input_image_path = obj.pet_path
        containers["preproc"][1].kwargs['half_life'] = get_half_life_from_nifty(obj.pet_path)
        containers["preproc"][2].kwargs['reference_image_path'] = obj.anat_path
        containers["preproc"][2].kwargs['half_life'] = get_half_life_from_nifty(obj.pet_path)
        containers["preproc"][3].segmentation_label_map_path = obj.seg_table
        containers["preproc"][3].segmentation_image_path = obj.seg_img
        containers["preproc"][4].raw_blood_tac_path = obj.blood_path
        
        obj.update_dependencies(verbose=False)
        return obj
