"""
BIDS utilities
"""
import os
import json
import numpy
import warnings
from nibabel.nifti1 import Nifti1Image
from nibabel.filebasedimages import FileBasedImage
from registration_tools import ImageIO


class BidsProject:
    """
    Class to handle BIDS project filepaths and to sort input data into output BIDS.
    """

    def __init__(self,
                 project_path: str) -> None:

        self.path = project_path

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
            full_path = os.path.join(self.path, dir_name)
            os.makedirs(full_path, exist_ok=True)

        # Create files
        for file_name, content in files_to_create.items():
            full_path = os.path.join(self.path, file_name)
            with open(full_path, 'w') as f:
                f.write(content)


class BidsInstance:

    def __init__(self, bids_project: BidsProject):
        self.project = bids_project
        self.directory_path = ""
        self.file_basename = ""
        self.full_filepath = ""

        self.prefixes = {"subject": "sub-",
                         "session": "ses-",
                         "acquisition": "acq-",
                         "contrast_enhancing": "ce-",
                         "reconstruction": "rec-",
                         "space": "space-",
                         "description": "desc-"}

    def add_prefixes(self, elements: dict) -> dict:
        for key in list(elements.keys()):
            value = elements[key]
            if isinstance(value, str):
                if key in self.prefixes:
                    if not value.startswith(self.prefixes[key]):
                        elements[key] = self.prefixes[key] + elements[key]
            else:
                elements.pop(key)
        return elements

    def create_bids_filepath(self,
                             warn: bool = True,
                             bids_directory_path: str = None,
                             bids_file_basename: str = None,
                             file_extension: str = "",
                             subject: str = None,
                             session: str = None,
                             modality: str = None,
                             image_type: str = None,
                             acquisition: str = None,
                             contrast_enhancing: str = None,
                             reconstruction: str = None,
                             space: str = None,
                             description: str = None,
                             derivative_directory: str = None) -> None:

        parameters_dictionary = locals().copy()

        if bids_directory_path is not None and bids_file_basename is not None:
            self.manual_bids_filepath(bids_path=bids_directory_path,
                                      bids_filename=bids_file_basename,
                                      extension=file_extension)
            return

        # if bids_directory_path is None:
        if all(key is not None for key in (subject,
                                           session,
                                           modality)):
            self.create_bids_directory_path(subject=subject,
                                            session=session,
                                            modality=modality,
                                            derivative_directory=derivative_directory)
        elif any(key is not None for key in (subject,
                                             session,
                                             modality,
                                             derivative_directory)):
            input_dictionary = self.edit_filepath_constituents(constituent_string=self.directory_path,
                                                               input_dictionary=parameters_dictionary,
                                                               constituent_delimiter="/")

            previous_directory_path = self.directory_path

            self.create_bids_directory_path(subject=input_dictionary.get("subject"),
                                            session=input_dictionary.get("session"),
                                            modality=input_dictionary.get("modality"),
                                            derivative_directory=input_dictionary.get("derivative_directory"))
            if warn:
                warnings.warn(f"The directory_path variable in this BidsPath object has been edited:\n\
                FROM: \"{previous_directory_path}\"\n\
                TO: \"{self.directory_path}\"")

        if all(key is not None for key in (subject,
                                           session,
                                           image_type)):
            self.create_bids_file_basename(subject=subject,
                                           session=session,
                                           image_type=image_type,
                                           acquisition=acquisition,
                                           contrast_enhancing=contrast_enhancing,
                                           reconstruction=reconstruction,
                                           space=space,
                                           description=description)
        elif any(key is not None for key in (subject,
                                             session,
                                             image_type)):
            input_dictionary = self.edit_filepath_constituents(constituent_string=self.file_basename,
                                                               input_dictionary=parameters_dictionary,
                                                               constituent_delimiter="_")

            previous_file_basename = self.file_basename

            self.create_bids_file_basename(subject=input_dictionary.get("subject"),
                                           session=input_dictionary.get("session"),
                                           image_type=input_dictionary.get("image_type"),
                                           acquisition=input_dictionary.get("acquisition"),
                                           contrast_enhancing=input_dictionary.get("contrast_enhancing"),
                                           reconstruction=input_dictionary.get("reconstruction"),
                                           space=input_dictionary.get("space"),
                                           description=input_dictionary.get("description"))
            if warn:
                warnings.warn(f"The file_basename variable in this BidsPath object has been edited:\n\
                FROM: \"{previous_file_basename}\"\n\
                TO: \"{self.file_basename}\"")

        self.compile_bids_filepath(extension=file_extension)

    def edit_filepath_constituents(self,
                                   constituent_string: str,
                                   input_dictionary: dict,
                                   constituent_delimiter: str) -> dict:

        constituent_string_parts = constituent_string.split(constituent_delimiter)

        for key, value in input_dictionary.items():
            if value is None:
                prefix = self.prefixes.get(key)
                if prefix is not None:
                    for part in constituent_string_parts:
                        if part.startswith(prefix):
                            input_dictionary[key] = part[len(prefix):]
                            break
                elif ((constituent_delimiter == "/" and key == "modality") or
                      (constituent_delimiter == "_" and key == "image_type")):  # last part modality or image_type
                    input_dictionary[key] = constituent_string_parts[-1]
                elif constituent_delimiter == "/" and key == "derivative_directory":
                    if "derivatives" in constituent_string_parts:
                        input_dictionary[key] = constituent_string_parts[
                            constituent_string_parts.index("derivatives") + 1]

        return input_dictionary

    def create_bids_directory_path(self,
                                   subject: str,
                                   session: str,
                                   modality: str,
                                   derivative_directory: str = "main",
                                   make_directories: bool = True) -> None:

        elements = locals().copy()
        elements = self.add_prefixes(elements)

        if derivative_directory is not None and derivative_directory != "main":
            modality_path = os.path.join(self.project.path,
                                         "derivatives",
                                         derivative_directory,
                                         elements["subject"],
                                         elements["session"],
                                         modality)
        else:
            modality_path = os.path.join(self.project.path,
                                         elements["subject"],
                                         elements["session"],
                                         modality)

        if make_directories:
            os.makedirs(modality_path, exist_ok=True)

        self.directory_path = modality_path

    def create_bids_file_basename(self,
                                  subject: str,
                                  session: str,
                                  image_type: str,
                                  acquisition: str = None,
                                  contrast_enhancing: str = None,
                                  reconstruction: str = None,
                                  space: str = None,
                                  description: str = None) -> None:

        elements = locals().copy()
        elements = self.add_prefixes(elements)

        bids_filename = '_'.join(
            [value for key, value in elements.items() if value is not None and key != "image_type"])
        bids_filename = bids_filename + "_" + elements["image_type"]

        self.file_basename = bids_filename

    """
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
    """

    def compile_bids_filepath(self,
                              extension: str = "") -> None:

        self.full_filepath = os.path.join(self.directory_path, self.file_basename) + extension

    def manual_bids_filepath(self,
                             bids_path: str,
                             bids_filename: str,
                             extension: str = "") -> None:

        self.full_filepath = os.path.join(self.project.path, bids_path, bids_filename) + extension

    def write_symbolic_link(self,
                            input_file_path: str) -> None:

        input_filename = os.path.basename(input_file_path)
        extension = '.' + '.'.join(input_filename.split('.')[1:])
        link_file_path = self.full_filepath + extension

        if os.path.exists(link_file_path) or os.path.islink(link_file_path):
            os.remove(link_file_path)
        os.symlink(input_file_path, link_file_path)

    def write_file(self,
                   file_input) -> None:

        if isinstance(file_input, FileBasedImage) or isinstance(file_input, Nifti1Image):
            print("Nifti")
            ImageIO.save_nii(image=file_input, out_file=self.full_filepath)
        elif type(file_input) is dict:
            print("JSON")
            save_json(json_dict=file_input, filepath=self.full_filepath)
        elif type(file_input) is numpy.array:
            print("TSV")
            # save_tsv(object, bids_file_path)


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
