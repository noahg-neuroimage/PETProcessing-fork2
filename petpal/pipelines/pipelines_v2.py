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


class BaseFunctionBasedStep():
    def __init__(self, name: str, function: Callable, *args, **kwargs) -> None:
        self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.func_sig = inspect.signature(self.function)
        self.validate_kwargs_for_non_default_have_been_set()
        self._func_name = function.__name__
        
    def get_function_args_not_set_in_kwargs(self) -> ArgsDict:
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        arg_names = list(func_params)
        for arg_name in arg_names[len(self.args):]:
            if arg_name not in self.kwargs:
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
        
    