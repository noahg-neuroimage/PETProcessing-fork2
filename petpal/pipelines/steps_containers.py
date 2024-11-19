import copy
from typing import Union
import networkx as nx
from .steps_base import *
from .preproc_steps import PreprocStepType, ImageToImageStep, TACsFromSegmentationStep, ResampleBloodTACStep
from .kinetic_modeling_steps import KMStepType, GraphicalAnalysisStep, TCMFittingAnalysisStep, ParametricGraphicalAnalysisStep

StepType = Union[FunctionBasedStep, ObjectBasedStep, PreprocStepType, KMStepType]

class StepsContainer:
    def __init__(self, name: str, *steps: StepType):
        self.name = name
        self.step_objs: list[StepType] = []
        self.step_names: list[str] = []
        for step in steps:
            self.add_step(step)
    
    def __repr__(self):
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(', f'{repr(self.name)},']
        
        for step_obj in self.step_objs:
            info_str.append(f'{repr(step_obj)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
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
                print(f"Step Number {step_id + 1}: {step_name}")
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
    def default_preprocess_steps(cls, name: str = 'preproc'):
        obj = cls(name=name)
        obj.add_step(ImageToImageStep.default_threshold_cropping())
        obj.add_step(ImageToImageStep.default_moco_frames_above_mean())
        obj.add_step(ImageToImageStep.default_register_pet_to_t1())
        obj.add_step(TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois())
        obj.add_step(ResampleBloodTACStep.default_resample_blood_tac_on_scanner_times())
        return obj
    
    @classmethod
    def default_graphical_analysis_steps(cls, name: str = 'km_graphical_analysis'):
        obj = cls(name=name)
        obj.add_step(GraphicalAnalysisStep.default_patlak())
        obj.add_step(GraphicalAnalysisStep.default_logan())
        obj.add_step(GraphicalAnalysisStep.default_alt_logan())
        return obj
    
    @classmethod
    def default_parametric_graphical_analysis_steps(cls, name: str = 'km_parametric_graphical_analysis'):
        obj = cls(name=name)
        obj.add_step(ParametricGraphicalAnalysisStep.default_patlak())
        obj.add_step(ParametricGraphicalAnalysisStep.default_logan())
        obj.add_step(ParametricGraphicalAnalysisStep.default_alt_logan())
        return obj
    
    @classmethod
    def default_tcm_analysis_steps(cls, name: str = 'km_tcm_analysis'):
        obj = cls(name=name)
        obj.add_step(TCMFittingAnalysisStep.default_1tcm())
        obj.add_step(TCMFittingAnalysisStep.default_serial2tcm())
        obj.add_step(TCMFittingAnalysisStep.default_irreversible_2tcm())
        return obj
    
    @classmethod
    def default_kinetic_analysis_steps(cls, name: str = 'km'):
        parametric_graphical_analysis_steps = cls.default_parametric_graphical_analysis_steps()
        graphical_analysis_steps = cls.default_graphical_analysis_steps()
        tcm_analysis_steps = cls.default_tcm_analysis_steps()
        
        obj = parametric_graphical_analysis_steps + graphical_analysis_steps + tcm_analysis_steps
        obj.name = name
        return obj


class StepsPipeline:
    def __init__(self, name: str, step_containers: list[StepsContainer]):
        self.name: str = name
        self.step_containers: dict[str, StepsContainer] = {}
        self.dependency_graph = nx.DiGraph()
        
        for container in step_containers:
            self.add_container(container)
    
    def __repr__(self):
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(', f'name={repr(self.name)},' 'step_containers=[']
        
        for _, container_obj in self.step_containers.items():
            info_str.append(f'{repr(container_obj)},')
        info_str.append(']')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
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
        obj = cls(name=name, step_containers=[])
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