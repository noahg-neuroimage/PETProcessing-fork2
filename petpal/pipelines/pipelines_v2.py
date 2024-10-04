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
        
    def validate_kwargs_for_non_default_have_been_set(self) -> None:
        pass
    
    def get_empty_default_kwargs(self):
        pass
    
    def execute(self):
        print(f"(Info): Executing {self.name}")
        self.function(*self.args, **self.kwargs)
        print(f"(Info): Finished {self.name}")
        
    