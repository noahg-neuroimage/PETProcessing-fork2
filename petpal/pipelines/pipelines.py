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
                    f"The provided step: {sending_step}\n is not an instance of ImageToImageStep. "
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
                    f'{unset_init_args}',
                    'Call Arguments:',
                    f'{self.call_kwargs}',
                    'Default Call Arguments:',
                    f'{unset_call_args}']
        return '\n'.join(info_str)


StepType = Union[ImageToImageStep, GenericStep, ObjectBasedStep]


class GenericPipeline:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.step_objs: list[StepType] = []
        self.step_names: list[str] = []
        
    def add_step(self, step: StepType) -> None:
        if step.name not in self.step_names:
            self.step_objs.append(step)
            self.step_names.append(step.name)
        else:
            raise KeyError("A step with this name already exists.")
        
    def list_step_details(self) -> None:
        if not self.step_objs:
            return None
        print("*"*90)
        print(f"({self.name} Pipeline Info):")
        for step_id, a_step in enumerate(self.step_objs):
            print('-' * 80)
            print(f"Step Number {step_id+1}")
            print(a_step)
            print('-' * 80)
        print("*" * 90)
        
    def list_step_names(self) -> None:
        if not self.step_objs:
            return None
        print(f"({self.name} pipeline info):")
        print('-' * 80)
        for step_id, step_name in enumerate(self.step_names):
            print(f"Step Number {step_id+1}: {step_name}")
        print('-' * 80)
    
    def get_step_names(self):
        return self.step_names
    
    def run_steps(self) -> None:
        for step_id, (step_name, a_step) in enumerate(zip(self.step_names, self.step_objs)):
            a_step.execute()
            
    def __getitem__(self, step: Union[int, str]) -> StepType:
        """
        Allows accessing steps by name.

        Args:
            step: int|str, step number in the sequence. If a string, we get the corresponding step

        Returns:
            Corresponding Step object

        Raises:
            IndexError: If the index is out of range
            KeyError: If the name does not exist
        """
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
    

class ProcessingPipeline(object):
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.preproc: GenericPipeline = GenericPipeline('preproc')
        self.km: GenericPipeline = GenericPipeline('km')
        self.dependency_graph: nx.DiGraph = nx.DiGraph()
        self.output_to_input_chains: list[tuple[StepType, StepType]] = []
    
    def run(self):
        self.preproc.run_steps()
        self.km.run_steps()
    
    def run_preproc(self):
        self.preproc.run_steps()
    
    def run_kinetic_modeling(self):
        self.km.run_steps()
    
    def add_preproc_step(self, step: StepType, receives_output_from_previous_step_as_input: bool = False) -> None:
        self.preproc.add_step(step)
        self.dependency_graph.add_node(":".join(['preproc', step.name]), proc='preproc')
        if receives_output_from_previous_step_as_input:
            self.add_output_to_input_dependency(out_pipe_name='preproc', in_pipe_name='preproc',
                                                out_step=-2, in_step=-1)
            self.chain_outputs_as_inputs_between_steps(out_pipe_name='preproc',
                                                       in_pipe_name='preproc',
                                                       out_step=-2,
                                                       in_step=-1)
    
    def add_kinetic_modeling_step(self, step: StepType, receives_output_from_previous_step: bool = False) -> None:
        self.km.add_step(step)
        self.dependency_graph.add_node(":".join(['km', step.name]), proc='km')
        if receives_output_from_previous_step:
            self.add_output_to_input_dependency(out_pipe_name='km',
                                                in_pipe_name='km', out_step=-2,
                                                in_step=-1)
            self.chain_outputs_as_inputs_between_steps(out_pipe_name='km',
                                                       in_pipe_name='km', out_step=-2,
                                                       in_step=-1)
    
    def list_preproc_steps(self) -> None:
        self.preproc.list_step_names()
    
    def list_kinetic_modeling_steps(self) -> None:
        self.km.list_step_names()
    
    def list_preproc_step_details(self) -> None:
        self.preproc.list_step_details()
    
    def list_kinetic_modeling_step_details(self) -> None:
        self.km.list_step_details()
        
    def __getitem__(self, pipe_name: str) -> GenericPipeline:
        try:
            return getattr(self, pipe_name)
        except AttributeError:
            raise KeyError(f"Pipeline {pipe_name} does not exist.")
    
    def add_output_to_input_dependency(self,
                                       out_pipe_name: str, in_pipe_name: str,
                                       out_step: Union[str, int], in_step: Union[str, int]) -> None:
        out_pipe: GenericPipeline = self[out_pipe_name]
        in_pipe: GenericPipeline = self[in_pipe_name]
        
        out_node_name = ":".join([out_pipe_name, out_pipe[out_step].name])
        in_node_name  = ":".join([in_pipe_name, in_pipe[in_step].name])
        
        self.dependency_graph.add_edge(out_node_name, in_node_name)
        
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            raise RuntimeError(f"Adding dependency {out_node_name} -> {in_node_name} creates a cycle!")
    
    def update_step_dependencies(self, verbose=True):
        for node_name in nx.topological_sort(self.dependency_graph):
            for an_edge in self.dependency_graph[node_name]:
                out_pipe, out_step = node_name.split(':')
                in_pipe, in_step = an_edge.split(':')
                out_pipe = self[out_pipe]
                in_pipe = self[in_pipe]
                in_pipe[in_step].set_input_as_output_from(out_pipe[out_step])
                if verbose:
                    print(f"(Info): Dependency {node_name} -> {an_edge} updated")
    
    
    def chain_outputs_as_inputs_between_steps(self,
                                              out_pipe_name: str,
                                              in_pipe_name: str,
                                              out_step: Union[str, int],
                                              in_step: Union[str, int]) -> None:
        
        try:
            out_pipeline : GenericPipeline = getattr(self, out_pipe_name)
            in_pipeline : GenericPipeline = getattr(self, in_pipe_name)
            if (out_pipeline[out_step], in_pipeline[in_step]) not in self.output_to_input_chains:
                in_pipeline[in_step].set_input_as_output_from(out_pipeline[out_step])
                self.output_to_input_chains.append((out_pipeline[out_step], in_pipeline[in_step]))
                
        except AttributeError as e:
            raise RuntimeError(f"Error setting chaining outputs and inputs for steps: {e}")

        
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
                                           motion_target_option='weighted_series_sum',
                                            half_life='',
                                           verbose=True),
    'write_roi_tacs' : GenericStep(name='write_roi_tacs',
                                   function=preproc.image_operations_4d.write_tacs,
                                   input_image_4d_path='',
                                   label_map_path='',
                                   segmentation_image_path='',
                                   out_tac_dir='',
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
                                               t_thresh_in_mins=30.0)),
    'parametric_cmrglc' : ImageToImageStep(name='parametric_cmrglc', # Clearly we need a path-to-path step as a more general step
                                           function=parametric_images.generate_cmrglc_parametric_image_from_ki_image,
                                           input_image_path='',
                                           output_image_path='',
                                           plasma_glucose_file_path='',
                                           glucose_rescaling_constant=1.0/18.0,
                                           lumped_constant=0.65,
                                           rescaling_const=100.,
                                           ),
    'roi_2tcm-k4zero_fit' : ObjectBasedStep(name='roi_2tcm-k4zero_fit',
                                           class_type=tac_fitting.FitTCMToTAC,
                                           init_kwargs=dict(input_tac_path='',
                                                            roi_tac_path='',
                                                            output_directory='',
                                                            output_filename_prefix='',
                                                            compartment_model='2tcm-k4zero'),
                                           call_kwargs=dict()),
    }

general_fdg_pipeline = ProcessingPipeline(name='general_fdg_pipeline',)

general_fdg_pipeline.add_preproc_step(TEMPLATE_STEPS['thresh_crop'])
general_fdg_pipeline.add_preproc_step(TEMPLATE_STEPS['moco_frames_above_mean'],
                                      receives_output_from_previous_step_as_input=True)
general_fdg_pipeline.add_preproc_step(TEMPLATE_STEPS['register_pet_to_t1'],
                                      receives_output_from_previous_step_as_input=True)
general_fdg_pipeline.add_preproc_step(TEMPLATE_STEPS['write_roi_tacs'],
                                      receives_output_from_previous_step_as_input=False)
general_fdg_pipeline.add_preproc_step(TEMPLATE_STEPS['resample_blood'])

general_fdg_pipeline.add_kinetic_modeling_step(TEMPLATE_STEPS['parametric_patlak'])
general_fdg_pipeline.add_kinetic_modeling_step(TEMPLATE_STEPS['parametric_cmrglc'],
                                               receives_output_from_previous_step=False)
general_fdg_pipeline.add_kinetic_modeling_step(TEMPLATE_STEPS['roi_2tcm-k4zero_fit'],
                                               receives_output_from_previous_step=False)


class BIDsPipeline():
    def __init__(self,
                 sub_id: str,
                 ses_id: str,
                 bids_dir: str = '../',
                 proc_pipeline: ProcessingPipeline = general_fdg_pipeline):
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.bids_root_dir = os.path.abspath(bids_dir)
        self.processing_pipeline = copy.deepcopy(proc_pipeline)