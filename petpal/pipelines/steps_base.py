import inspect
from typing import Callable, Union

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