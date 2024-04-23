import os
import docker
from docker.errors import ImageNotFound, APIError
from typing import Union, Tuple


class PetPvc:

    def __init__(self):
        self.client = docker.from_env()
        self.image_name = "benthomas1984/petpvc"
        self._pull_image_if_not_exists()

    def run_petpvc(self,
                   pet_4d_filepath: str,
                   output_filepath: str,
                   pvc_method: str,
                   psf_dimensions: Union[Tuple[float, float, float], float],
                   mask_4d_filepath: str = None,
                   verbose: bool = False,
                   debug: bool = False) -> None:
        common_path = os.path.commonpath([pet_4d_filepath, output_filepath])
        docker_pet_input = "/data/" + pet_4d_filepath.replace(common_path, "").lstrip('/')
        docker_output = "/data/" + output_filepath.replace(common_path, "").lstrip('/')
        docker_volumes = {common_path: {'bind': '/data', 'mode': 'rw'}}
        command = f"petpvc --input {docker_pet_input} --output {docker_output} --pvc {pvc_method}"
        if mask_4d_filepath is not None:
            docker_mask_input = "/data/" + mask_4d_filepath.replace(common_path, "").lstrip('/')
            command = command + f" --mask {docker_mask_input}"
        if isinstance(psf_dimensions, tuple):
            command = command + f" -x {psf_dimensions[0]} -y {psf_dimensions[1]} -z {psf_dimensions[2]}"
        else:
            command = command + f" -x {psf_dimensions} -y {psf_dimensions} -z {psf_dimensions}"
        if debug:
            command = command + f" --debug"
        container = self.client.containers.run(self.image_name, command, volumes=docker_volumes, detach=False, stream=True, auto_remove=True)
        if verbose:
            for line in container:
                print(line.decode('utf-8').strip())

    def _pull_image_if_not_exists(self) -> None:
        client = docker.from_env()
        try:
            client.images.get(self.image_name)
            print(f"Image {self.image_name} is already available locally.")
        except docker.errors.ImageNotFound:
            print(f"Image {self.image_name} not found locally. Pulling from Docker Hub...")
            client.images.pull(self.image_name)
            print(f"Successfully pulled {self.image_name}.")
        except docker.errors.APIError as error:
            print(f"Failed to pull image due to API error: {error}")
