from typing import Callable

class Step:
    def __init__(self, name: str, function: Callable,
                 input_image_path: str, output_image_path: str,
                 *args, **kwargs) -> None:
        self.name = name
        self.function = function
        self.input_image = input_image_path
        self.output_image = output_image_path
        self.args = args
        self.kwargs = kwargs
        
    def execute(self) -> None:
        print(f"(Info): Executing {self.name}")
        self.function(self.input_image, self.output_image, *self.args, **self.kwargs)
        print(f"(Info): Finished {self.name}")
        
        
        
class Pipeline:
    def __init__(self, name: str) -> None:
        self.name = name
        self.steps = []
        
    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        
    def list_steps(self) -> None:
        for step in self.steps:
            print(step.name)
        
    def run_steps(self) -> None:
        for step in self.steps:
            step.execute()