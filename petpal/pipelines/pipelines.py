from typing import Callable
import inspect


class AbstractStep:
    def __init__(self, name: str, function: Callable, **kwargs) -> None:
        self.name = name
        self.function = function
        self.kwargs = kwargs
        self.func_sig = inspect.signature(self.function)
        
    def get_non_unset_args_from_function(self):
        """
        This class gets all the function arguments as kwargs so we check through them all.
        Returns:

        """
        unset_args_dict = dict()
        func_params = self.func_sig.parameters
        for arg_name, arg_val in func_params.items():
            if arg_name not in self.kwargs:
                    unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
        
    
    def execute(self) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(**self.kwargs)
        print(f"(Info): Finished {self.name}")


class ImageToImageStep(AbstractStep):
    def __init__(self, name: str, function: Callable,
                 input_image_path: str, output_image_path: str,
                 **kwargs) -> None:
        super().__init__(name, function, **kwargs)
        self.input_image = input_image_path
        self.output_image = output_image_path
        
        
    def execute(self) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image, self.output_image, **self.kwargs)
        print(f"(Info): Finished {self.name}")
        
    def get_non_unset_args_from_function(self):
        """
        Since this class expects to have functions with the signature f(some_input_path, some_output_path, *args),
        we want to skip the first two arguments
        Returns:

        """
        unset_args_dict = dict()
        func_params = self.func_sig.parameters
        for arg_name, arg_val in list(func_params.items())[2:]:
            if arg_name not in self.kwargs:
                    unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
        
        
class AbstractPipeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self.steps = []
        
    def add_step(self, step: AbstractStep) -> None:
        self.steps.append(step)
        
    def list_steps(self) -> None:
        for step in self.steps:
            print(step.name)
        
    def run_steps(self) -> None:
        for step in self.steps:
            step.execute()


class ProcessingPipeline(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.preproc = AbstractPipeline('preproc')
        self.kinetic_modeling = AbstractPipeline('kinetic_modeling')
    
    def run_preproc(self):
        self.preproc.run_steps()
    
    def run_kinetic_modeling(self):
        self.kinetic_modeling.run_steps()
    
    def add_preproc_step(self, step: AbstractStep) -> None:
        self.preproc.add_step(step)
    
    def add_kinetic_modeling_step(self, step: AbstractStep) -> None:
        self.kinetic_modeling.add_step(step)
    
    def list_preproc_steps(self) -> None:
        self.preproc.list_steps()
    
    def list_kinetic_modeling_steps(self) -> None:
        self.kinetic_modeling.list_steps()
