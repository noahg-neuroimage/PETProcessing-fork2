"""
Image IO
"""
import json
import re
import os
import ants
import nibabel
from nibabel.filebasedimages import FileBasedHeader
from typing import Union
import numpy as np
import pandas as pd
from . import useful_functions


def write_dict_to_json(meta_data_dict: dict, out_path: str):
    """
    Save a metadata file in python to a directory.

    Args:
        meta_data_dict (dict): A dictionary with imaging metadata, to be saved to file.
        out_path (str): Directory to which `meta_file` is to be saved.
    """
    with open(out_path, 'w', encoding='utf-8') as copy_file:
        json.dump(meta_data_dict, copy_file, indent=4)


def convert_ctab_to_dseg(ctab_path: str,
                         dseg_path: str,
                         column_names: list[str]=None):
    """
    Convert a FreeSurfer compatible color table into a BIDS compatible label
    map ``dseg.tsv``.

    Args:
        ctab_path (str): Path to FreeSurfer compatible color table.
        dseg_path (str): Path to ``dseg.tsv`` label mapfile to save.
        column_names (list[str]): List of columns present in color table. Must
            include 'mapping' and 'name'.
    """
    if column_names==None:
        column_names = ['mapping','name','r','g','b','a','ttype']
    fs_ctab = pd.read_csv(ctab_path,
                          delim_whitespace=True,
                          header=None,
                          comment='#',
                          names=column_names)
    label_names = {'name': fs_ctab['name'],
                   'mapping': fs_ctab['mapping'],
                   'abbreviation': useful_functions.build_label_map(fs_ctab['name'])}
    label_map = pd.DataFrame(data=label_names,
                             columns=['name','abbreviation','mapping']).rename_axis('index')
    label_map = label_map.sort_values(by=['mapping'])
    label_map.to_csv(dseg_path,sep='\t')
    return label_map

def _gen_meta_data_filepath_for_nifti(nifty_path:str):
    """
    Generates the corresponding metadata file path for a given nifti file path.

    This function takes a nifti file path (with `.nii` or `.nii.gz` extension)
    and replaces the extension with `.json` to derive the expected metadata file path.

    Args:
        nifty_path (str): Path to the nifti file (with `.nii` or `.nii.gz` extension).

    Returns:
        str: The generated metadata file path with a `.json` extension.
    """
    meta_data_path = re.sub(r'\.nii\.gz$|\.nii$', '.json', nifty_path)
    return meta_data_path

def load_metadata_for_nifti_with_same_filename(image_path) -> dict:
    """
    Static method to load metadata. Assume same path as input image path.

    Args:
        image_path (str): Path to image for which a .json file of the
            same name as the file but with different extension exists.

    Returns:
        image_meta (dict): Dictionary where keys are fields in the image
            metadata file and values correspond to values in those fields.

    Raises:
        FileNotFoundError: If the provided image path cannot be found in the directory.
        Additionally, occurs if the metadata .json file cannot be found.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")

    meta_path = _gen_meta_data_filepath_for_nifti(image_path)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file {meta_path} not found. Does it have a different path?")

    with open(meta_path, 'r', encoding='utf-8') as meta_file:
        image_meta = json.load(meta_file)
    return image_meta


def safe_load_tac(filename: str, **kwargs) -> np.ndarray:
    """
    Loads time-activity curves (TAC) from a file.
    Tries to read a TAC from specified file and raises an exception if unable to do so. We assume that the file has two
    columns, the first corresponding to time and second corresponding to activity.
    Args:
        filename (str): The name of the file to be loaded.
    Returns:
        np.ndarray: A numpy array containing the loaded TAC. The first index corresponds to the times, and the second
        corresponds to the activity.
    Raises:
        Exception: An error occurred loading the TAC.
    """
    try:
        tac_data = np.asarray(np.loadtxt(filename, **kwargs).T, dtype=float, order='C')
    except ValueError:
        tac_data = np.asarray(np.loadtxt(filename, skiprows=1, **kwargs).T, dtype=float, order='C')
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e

    if np.max(tac_data[0]) >= 300:
        tac_data[0] /= 60.0

    return tac_data


def safe_copy_meta(input_image_path: str,
                   out_image_path: str):
    """
    Copy the metadata file from input image, to one with the same name as the
    output file. Intended to be used in functions operating on images in order
    to ensure a metadata file is associated with each new image.
    
    Args:
        input_image_path (str): Path to the input file for the function
            generating a new image.
        out_image_path (str): Path to the output file written by the function.
    """
    copy_meta_path = _gen_meta_data_filepath_for_nifti(out_image_path)
    meta_data_dict = load_metadata_for_nifti_with_same_filename(input_image_path)
    write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)


def get_half_life_from_meta(meta_data_file_path: str):
    """
    Extracts the radionuclide half-life (usually in seconds) from a nifti metadata file.

    Args:
        meta_data_file_path (str): Path to the nifti metadata file.

    Returns:
        float: The radionuclide half-life extracted from the metadata file.

    Raises:
        FileNotFoundError: If the metadata file does not exist at the provided path.
        KeyError: If the 'RadionuclideHalfLife' key is not found in the metadata file.
    """
    if not os.path.exists(meta_data_file_path):
        raise FileNotFoundError(f"Metadata file {meta_data_file_path} not found")
    with open(meta_data_file_path, 'r') as m_file:
        meta_data = json.load(m_file)
    try:
        half_life = meta_data['RadionuclideHalfLife']
        return half_life
    except KeyError:
        raise KeyError("RadionuclideHalfLife not found in meta-data file.")
    
def get_half_life_from_nifti(image_path:str):
    """
    Retrieves the radionuclide half-life from a nifti image file.

    This function first checks if the provided nifti image file exists. It then derives
    the corresponding metadata file path using :func:`_gen_meta_data_filepath_for_nifti`
    and finally retrieves the half-life from the metadata using :func:`get_half_life_from_meta`.

    Args:
        image_path (str): Path to the nifti image file.

    Returns:
        float: The radionuclide half-life extracted from the metadata file.

    Raises:
        FileNotFoundError: If the nifti image file does not exist at the provided path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    meta_path = _gen_meta_data_filepath_for_nifti(image_path)
    return get_half_life_from_meta(meta_path)


def get_frame_timing_info_for_nifti(image_path: str) -> dict[str, np.ndarray]:
    r"""
    Extracts frame timing information and decay factors from a NIfTI image metadata.
    Expects that the JSON metadata file has ``FrameDuration`` and ``DecayFactor`` or 
    ``DecayCorrectionFactor`` keys.

    .. important::
        This function tries to infer `FrameTimesEnd` and `FrameTimesStart` from the frame durations
        if those keys are not present in the metadata file. If the scan is broken, this might generate
        incorrect results.


    Args:
        image_path (str): Path to the NIfTI image file.

    Returns:
        dict: Frame timing information with the following keys:
            - `duration` (np.ndarray): Frame durations in seconds.
            - `start` (np.ndarray): Frame start times in seconds.
            - `end` (np.ndarray): Frame end times in seconds.
            - `decay` (np.ndarray): Decay factors for each frame.
    """
    _meta_data = load_metadata_for_nifti_with_same_filename(image_path=image_path)
    frm_dur = np.asarray(_meta_data['FrameDuration'], int)
    try:
        frm_ends = np.asarray(_meta_data['FrameTimesEnd'], int)
    except KeyError:
        frm_ends = np.cumsum(frm_dur)
    try:
        frm_starts = np.asarray(_meta_data['FrameTimesStart'], int)
    except KeyError:
        frm_starts = np.diff(frm_ends)
    try:
        decay = _meta_data['DecayCorrectionFactor']
    except KeyError:
        decay = _meta_data['DecayFactor']
    else:
        raise KeyError("Neither 'DecayCorrectionFactor' nor 'DecayFactor' keys found in metadata.")
    frm_info = {'duration': frm_dur,
                'start': frm_starts,
                'end': frm_ends,
                'decay': decay
                }

    return frm_info

class ImageIO:
    """
    :class:`ImageIO` to handle reading and writing imaging data and metadata.

    Provides several tools designed for reading and writing data within the Python environment.

    Key methods include:
        - :meth:`load_nii`: Loads a NIfTI file from a file path.
        - :meth:`save_nii`: Saves a loaded NIfTI file to a file path.
        - :meth:`extract_image_from_nii_as_numpy`: Extracts imaging data from a NIfTI file as a numpy array.
        - :meth:`extract_header_from_nii`: Extracts header information from a NIfTI file as a dictionary.
        - :meth:`extract_np_to_nibabel`: Wraps imaging information in numpy into an Nibabel image.

    Attributes:
        verbose (bool): Set to `True` to output processing information.
    """

    def __init__(self, verbose: bool = True, ):
        """
        Initializes :class:`ImageIO` and sets verbose.

        Args:
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        """
        self.verbose = verbose

    def load_nii(self, image_path: str) -> Union[nibabel.nifti1.Nifti1Image, nibabel.nifti2.Nifti2Image]:
        """
        Wrapper to load nifti from image_path.

        Args:
            image_path (str): Path to a .nii or .nii.gz file.

        Returns:
            The nifti FileBasedImage.

        Raises:
            FileNotFoundError: If the provided image path cannot be found in the directory.
            OSError: If the provided image path does not have a NIfTI extension.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")

        if not re.search(r'\.nii\.gz$|\.nii$', image_path):
            raise OSError(f"{image_path} does not have the extension .nii or .nii.gz")

        image = nibabel.load(image_path)

        if not isinstance(image, (nibabel.nifti1.Nifti1Image, nibabel.nifti2.Nifti2Image)):
            raise OSError(f'File at {image_path} is not in nifti1 or nifti2 format')

        if self.verbose:
            print(f"(ImageIO): {image_path} loaded")

        return image

    def save_nii(self, image: nibabel.nifti1.Nifti1Image, out_file: str):
        """
        Wrapper to save nifti to file.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.
            out_file (str): File path to which image will be written.
        """
        nibabel.save(image, out_file)
        if self.verbose:
            print(f"(ImageIO): Image saved to {out_file}")

    def extract_image_from_nii_as_numpy(self, image: nibabel.nifti1.Nifti1Image) -> np.ndarray:
        """
        Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.

        Returns:
            The data contained in the .nii or .nii.gz file as a numpy array.
        """
        image_data = image.get_fdata()

        if self.verbose:
            print(f"(ImageIO): Image has shape {image_data.shape}")

        return image_data

    def extract_header_from_nii(self, image: nibabel.nifti1.Nifti1Image) -> FileBasedHeader:
        """
        Convenient wrapper to extract header information from a .nii or .nii.gz
        file as a nibabel file-based header.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.

        Returns:
            image_header (FileBasedHeader): The nifti header.
        """
        image_header = image.header

        if self.verbose:
            print(f"(ImageIO): Image header is: {image_header}")

        return image_header

    def extract_np_to_nibabel(self,
                              image_array: np.ndarray,
                              header: FileBasedHeader,
                              affine: np.ndarray) -> nibabel.nifti1.Nifti1Image:
        """
        Wrapper to convert an image array into nibabel object.

        Args:
            image_array (np.ndarray): Array containing image data.
            header (FileBasedHeader): Header information to include.
            affine (np.ndarray): Affine information we need to keep when rewriting image.

        Returns:
            image_nibabel (nibabel.nifti1.Nifti1Image): Image stored in nifti-like nibabel format.
        """
        image_nibabel = nibabel.nifti1.Nifti1Image(image_array, affine, header)
        return image_nibabel

    @staticmethod
    def affine_parse(image_affine: np.ndarray) -> tuple:
        """
        Parse the components of an image affine to return origin, spacing, direction.

        Args:
            image_affine (np.ndarray): A 4x4 affine matrix defining spacing, origin,
                and direction of an image.
        """
        spacing = nibabel.affines.voxel_sizes(image_affine)
        origin = image_affine[:, 3]

        quat = nibabel.quaternions.mat2quat(image_affine[:3, :3])
        dir_3x3 = nibabel.quaternions.quat2mat(quat)
        direction = np.zeros((4, 4))
        direction[-1, -1] = 1
        direction[:3, :3] = dir_3x3

        return spacing, origin, direction

    def extract_np_to_ants(self, image_array: np.ndarray, affine: np.ndarray) -> ants.ANTsImage:
        """
        Wrapper to convert an image array into ants object.
        Note header info is lost as ANTs does not carry this metadata.

        Args:
            image_array (np.ndarray): Array containing image data.
            affine (np.ndarray): Affine information we need to keep when rewriting image.

        Returns:
            image_ants (ants.ANTsImage): Image stored in nifti-like nibabel format.
        """
        origin, spacing, direction = self.affine_parse(affine)
        image_ants = ants.from_numpy(data=image_array, spacing=spacing, origin=origin, direction=direction)
        return image_ants

    @staticmethod
    def read_label_map_tsv(label_map_file: str) -> dict:
        """
        Static method to read a label map, translating region indices to region names,
        as a dictionary. Assumes tsv format.

        Args:
            label_map_file (str): Path to a json-formatted label map file.

        Returns:
            label_map (pd.DataFrame): Dataframe matching region indices, names,
                abbreviations, and mappings.

        Raises:
            FileNotFoundError: If the provided ctab file cannot be found in the directory.
        """
        if not os.path.exists(label_map_file):
            raise FileNotFoundError(f"Image file {label_map_file} not found")

        label_map = pd.read_csv(label_map_file,sep='\t')

        return label_map


def safe_load_4dpet_nifti(filename: str) -> nibabel.nifti1.Nifti1Image:
    """
    Safely load a 4D PET NIfTI file.

    This function checks if the given file has a '.nii' or '.nii.gz' extension, then tries to load
    it as a NIfTI file using the nibabel library. If the file cannot be loaded, it raises an
    exception.

    Args:
        filename (str): The path of the NIfTI file to be loaded.

    Returns:
        Nifti1Image: The loaded NIfTI 4D PET image.

    Raises:
        ValueError: If the file does not have a '.nii' or '.nii.gz' extension.
        Exception:  If an error occurred while loading the NIfTI file.
    """
    if not filename.endswith(('.nii', '.nii.gz')):
        raise ValueError(
            "Invalid file extension. Only '.nii' and '.nii.gz' are supported.")

    try:
        return nibabel.load(filename=filename)
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e


def validate_two_images_same_dimensions(image_1: nibabel.nifti1.Nifti1Image,
                                        image_2: nibabel.nifti1.Nifti1Image,
                                        check_4d: bool=False):
    """
    Check the dimensions of two Nifti1Image objects and verify they have the same shape.

    Args:
        image_1 (nibabel.nifti1.Nifti1Image): The first image of the two to check image size.
        image_2 (nibabel.nifti1.Nifti1Image): The second image of the two to check image size.
        check_4d (bool): If true, checks all dimensions including validating the number of frames.
            If false, only checks first three dimensions. Default False.
    
    Raises:
        ValueError: If images do not have the same dimensions.
    """
    shape_1 = image_1.shape
    shape_2 = image_2.shape

    same_shape = False
    if check_4d:
        same_shape = shape_1 == shape_2
    else:
        same_shape = shape_1[:3] == shape_2[:3]

    if not same_shape:
        raise ValueError(f'Got incompatible image sizes: {shape_1}, {shape_2}.')

def get_window_index_pairs_from_durations(frame_durations: np.ndarray, w_size: float):
    r"""
    Computes start and end index pairs for windows of a given size based on frame durations.

    Args:
        frame_durations (np.ndarray): Array of frame durations in seconds.
        w_size (float): Window size in seconds.

    Returns:
        np.ndarray: Array of shape (2, N), where the first row contains start indices,
            and the second row contains end indices for each window.

    Raises:
        ValueError: If `w_size` is less than or equal to 0.
        ValueError: If `w_size` is greater than the total duration of all frames.
    """
    if w_size <= 0:
        raise ValueError("Window size has to be > 0")
    if w_size > np.sum(frame_durations):
        raise ValueError("Window size is larger than the whole scan.")
    _tmp_w_ids = [0]
    _w_dur_sum = 0
    for frm_id, frm_dur in enumerate(frame_durations):
        _w_dur_sum += frm_dur
        if _w_dur_sum >= w_size:
            _tmp_w_ids.append(frm_id + 1)
            _w_dur_sum = 0
    w_start_ids = np.asarray(_tmp_w_ids[:-1])
    w_end_ids = np.asarray(_tmp_w_ids[1:])
    id_pairs = np.vstack((w_start_ids, w_end_ids))
    return id_pairs


def get_window_index_pairs_for_image(image_path: str, w_size: float):
    """
    Computes start and end index pairs for windows of a given size
    based on the frame durations of a NIfTI image.

    Args:
        image_path (str): Path to the NIfTI image file.
        w_size (float): Window size in seconds.

    Returns:
        np.ndarray: Array of shape (2, N), where the first row contains start indices,
            and the second row contains end indices for each window.

    Raises:
        ValueError: If `w_size` is less than or equal to 0.
        ValueError: If `w_size` is greater than the total duration of all frames.

    See Also:
        :func:`get_window_index_pairs_from_durations`
    """
    image_frame_info = get_frame_timing_info_for_nifti(image_path=image_path)
    return get_window_index_pairs_from_durations(frame_durations=image_frame_info['duration'], w_size=w_size)

