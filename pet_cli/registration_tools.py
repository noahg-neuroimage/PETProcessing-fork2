import nibabel.nifti1
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np
from typing import Union


def load_nii(file_path: str, verbose: bool) -> FileBasedImage:
    """

    Args:
        file_path:
        verbose:

    Returns:

    """
    image = nibabel.load(file_path)

    if verbose:
        print(f"(fileIO): {file} loaded")

    return image


def extract_image_from_nii_as_numpy(file: Union[str, FileBasedImage], verbose: bool) -> np.ndarray:
    """Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array.

    Args:
        file: The full path to the file or the nifti file itself.
        verbose:

    Returns:
        The data contained in the .nii or .nii.gz file as a numpy array

    """
    if isinstance(file, str):
        image_data: np.ndarray = nibabel.load(filename=file).get_fdata()
    elif isinstance(file, FileBasedImage):
        image_data: np.ndarray = file.get_fdata()
    else:
        raise TypeError("file must be a file path (str) or a Nifti1Image")

    if verbose:
        print(f"(fileIO): {file} has shape {image_data.shape}")

    return image_data


def extract_header_from_nii(file: Union[str, FileBasedImage], verbose: bool) -> FileBasedHeader:
    """Convenient wrapper to extract header information from a .nii or .nii.gz file as a nibabel file-based header

    Args:
        file: The full path to the file or the nifti file itself
        verbose:

    Returns:
        The nifti header

    """
    if isinstance(file, str):
        image_header: FileBasedHeader = nibabel.load(filename=file).header
    elif isinstance(file, FileBasedImage):
        image_header: FileBasedHeader = file.header
    else:
        raise TypeError("file must be a file path (str) or a Nifti1Image")

    if verbose:
        print(f"(fileIO): {file} header is: {image_header}")

    return image_header


def reorient_to_ras(file: str, verbose: bool) -> FileBasedImage:
    """

    Args:
        file_path:
        verbose:

    Returns:

    """
