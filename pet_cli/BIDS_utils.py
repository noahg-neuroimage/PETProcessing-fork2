"""
BIDS utilities
"""
import os
import json
import numpy
from nibabel.nifti1 import Nifti1Image
from nibabel.filebasedimages import FileBasedImage
from registration_tools import ImageIO


class BidsProject:
    """
    Class to handle BIDS project filepaths and to sort input data into output BIDS.
    """

    def __init__(self,
                 project_path: str) -> None:

        self.project_path = project_path
        self.prefixes = {"subject": "sub-",
                         "session": "ses-",
                         "acquisition": "acq-",
                         "contrast_enhancing": "ce-",
                         "reconstruction": "rec-",
                         "space": "space-",
                         "description": "desc-"}

    def create_bids_scaffold(self):
        dirs_to_create = [
            "code",
            "derivatives",
            "sourcedata",
        ]
        files_to_create = {
            "CHANGES": "Initial commit.",
            #        "dataset_description.json": json.dumps({
            #            "Name": "Example Dataset",
            #            "BIDSVersion": "1.6.0",
            #            "DatasetType": "raw"
            #        }, indent=4),
            "participants.json": "{}",
            "participants.tsv": "participant_id\tsession_id",
            "README": "This BIDS dataset was created for the file outputs of the PetProcessing Pipeline."
        }

        # Create directories
        for dir_name in dirs_to_create:
            full_path = os.path.join(self.project_path, dir_name)
            os.makedirs(full_path, exist_ok=True)

        # Create files
        for file_name, content in files_to_create.items():
            full_path = os.path.join(self.project_path, file_name)
            with open(full_path, 'w') as f:
                f.write(content)

    def add_prefixes(self,
                     elements: dict) -> dict:
        for key in list(elements.keys()):
            value = elements[key]
            if isinstance(value, str):
                if key in self.prefixes:
                    if not value.startswith(self.prefixes[key]):
                        elements[key] = self.prefixes[key] + elements[key]
            else:
                elements.pop(key)
        return elements

    def create_bids_path(self,
                         subject: str,
                         session: str,
                         modality: str,
                         derivative_directory: str = None,
                         make_directories: bool = True) -> str:

        elements = locals().copy()
        elements = self.add_prefixes(elements)

        if derivative_directory is not None:
            modality_path = os.path.join(self.project_path,
                                         "derivatives",
                                         derivative_directory,
                                         elements["subject"],
                                         elements["session"],
                                         modality)
        else:
            modality_path = os.path.join(self.project_path,
                                         elements["subject"],
                                         elements["session"],
                                         modality)

        if make_directories:
            os.makedirs(modality_path, exist_ok=True)

        return modality_path

    def create_bids_filename(self,
                             subject: str,
                             session: str,
                             image_type: str,
                             acquisition: str = None,
                             contrast_enhancing: str = None,
                             reconstruction: str = None,
                             space: str = None,
                             description: str = None) -> str:

        elements = locals().copy()
        elements = self.add_prefixes(elements)

        bids_filename = '_'.join([value for value in elements.values() if value is not None])

        return bids_filename

    @staticmethod
    def write_symbolic_link(input_file_path: str,
                            bids_file_path: str) -> None:

        input_filename = os.path.basename(input_file_path)
        extension = '.' + '.'.join(input_filename.split('.')[1:])
        link_file_path = bids_file_path + extension

        if os.path.exists(link_file_path) or os.path.islink(link_file_path):
            os.remove(link_file_path)
        os.symlink(input_file_path, link_file_path)

    @staticmethod
    def write_file(file_input,
                   bids_file_path: str) -> None:

        if isinstance(file_input, FileBasedImage) or isinstance(file_input, Nifti1Image):
            print("Nifti")
            ImageIO.save_nii(image=file_input, out_file=bids_file_path)
        elif type(file_input) is dict:
            print("JSON")
            save_json(json_dict=file_input, filepath=bids_file_path)
        elif type(file_input) is numpy.array:
            print("TSV")
            save_array_as_tsv(array=file_input, filepath=bids_file_path)

    def extract_filepath(self,
                         subject: str,
                         session: str,
                         modality: str,
                         image_type: str,
                         extension: str,
                         acquisition: str = None,
                         contrast_enhancing: str = None,
                         reconstruction: str = None,
                         space: str = None,
                         description: str = None,
                         derivative_directory: str = None) -> str:

        bids_path = self.create_bids_path(subject=subject,
                                          session=session,
                                          modality=modality,
                                          derivative_directory=derivative_directory,
                                          make_directories=False)

        bids_filename = self.create_bids_filename(subject=subject,
                                                  session=session,
                                                  image_type=image_type,
                                                  acquisition=acquisition,
                                                  contrast_enhancing=contrast_enhancing,
                                                  reconstruction=reconstruction,
                                                  space=space,
                                                  description=description)

        filepath = self.create_bids_filepath(bids_path=bids_path,
                                             bids_filename=bids_filename,
                                             extension=extension)

        if not os.path.exists(filepath) and not os.path.islink(filepath):
            raise FileNotFoundError(f"The file '{filepath}' does not exist.")

        return filepath

    @staticmethod
    def create_bids_filepath(bids_path: str,
                             bids_filename: str,
                             extension: str = "") -> str:

        filepath = os.path.join(bids_path, bids_filename) + extension

        return filepath


def create_json() -> dict:
    return {}


def append_json(json_dict: dict = None,
                **kwargs) -> dict:
    if json_dict is None:
        json_dict = create_json()
    json_dict.update(**kwargs)
    return json_dict


def save_json(json_dict: dict,
              filepath: str) -> None:
    if not filepath.endswith(".json"):
        filepath += ".json"
    with open(filepath, 'w') as file:
        json.dump(json_dict, file, indent=4)
        file.write('\n')


def save_array_as_tsv(array: numpy.array,
                      filepath: str) -> None:
    numpy.savetxt(filepath, array, delimiter='\t', fmt='%s')
