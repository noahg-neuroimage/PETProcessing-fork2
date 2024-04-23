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
                   mask_4d_filepath: str,
                   output_filepath: str,
                   pvc_method: str,
                   psf_dimensions: Union[Tuple[float, float, float], float],
                   docker_volumes: dict = None,
                   verbose: bool = False,
                   debug: bool = False) -> None:
        command = f"petpvc -i {pet_4d_filepath} -m {mask_4d_filepath} -o {output_filepath} --pvc {pvc_method}"
        if isinstance(psf_dimensions, tuple):
            command = command + f" -x {psf_dimensions[0]} -y {psf_dimensions[1]} -z {psf_dimensions[2]}"
        else:
            command = command + f" -x {psf_dimensions} -y {psf_dimensions} -z {psf_dimensions}"
        if debug:
            command = command + f" --debug"
        container = self.client.containers.run(self.image_name, command, volumes=docker_volumes)
        if verbose:
            for line in container.logs(stream=True):
                print(line.strip().decode('utf-8'))

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
