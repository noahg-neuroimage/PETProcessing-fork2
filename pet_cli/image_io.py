"""
Image IO
"""
import json
import re
import ants
import nibabel
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np

class ImageIO():
    """
    Class handling 3D and 4D image file utilities.
    """
    def __init__(self,
            image_path: str,
            verbose: bool=True,
            ):
        """
        Args:
            image_path (str): Path to existing Nifti image file.
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        
        """
        self.image_path = image_path
        self.verbose = verbose


    def load_nii(self) -> FileBasedImage:
        """
        Wrapper to load nifti from image_path.

        Returns:
            The nifti FileBasedImage.

        """
        image = nibabel.load(self.image_path)

        if self.verbose:
            print(f"(ImageIO): {self.image_path} loaded")

        return image


    def save_nii(self,image: nibabel.nifti1.Nifti1Image,out_file: str) -> int:
        """
        Wrapper to save nifti to file.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.
            out_file (str): File path to which image will be written.
        """
        nibabel.save(image,out_file)

        return 0





    def extract_image_from_nii_as_numpy(self, image: nibabel.nifti1.Nifti1Image) -> np.ndarray:
        """
        Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array.

        Args:
            image: Nibabel-type image to write to file.
            verbose:

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
            image: Nibabel-type image to write to file.

        Returns:
            The nifti header.
        """
        image_header: FileBasedHeader = image.header

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
        image_nibabel = nibabel.nifti1.Nifti1Image(image_array,affine,header)
        return image_nibabel


    def affine_parse(self,image_affine: np.ndarray) -> tuple:
        """
        Parse the components of an image affine to return origin, spacing, direction

        Note: this function is a placeholder as we decide what the specific input and output
        of various functions should be. Should this not be useful, then it will be removed.
        """
        spacing = image_affine[:3,:3].diagonal() # the diagonal is spacing
        origin = image_affine[:,3] # the last column in the affine is origin

        # quaternions seem to be the answer to the bizzare "direction" property
        # that ants uses- but likely more foolproof to just use from_nibabel than from_numpy
        quat = nibabel.quaternions.mat2quat(image_affine[:3,:3])
        dir_3x3 = nibabel.quaternions.quat2mat(quat)
        direction = np.zeros((4,4))
        direction[-1,-1] = 1
        direction[:3,:3] = dir_3x3

        return spacing, origin, direction


    def extract_np_to_ants(self,
                           image_array: np.ndarray,
                           affine: np.ndarray) -> ants.ANTsImage:
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
        image_ants = ants.from_numpy(data=image_array,
                                     spacing=spacing,
                                     origin=origin,
                                     direction=direction)
        return image_ants


    def read_ctab(self,
                  ctab_file: str) -> dict:
        """
        Function to read a color table, translating region indices to region names, as a dictionary.
        Assumes json format.
        """
        ctab_json = json.load(ctab_file)
        return ctab_json


def load_meta(image_path) -> dict:
    """
    Wrapper to load metadata. Assume same path as input image path.
    """
    meta_path = re.sub('.nii.gz|.nii','.json',image_path)
    with open(meta_path,'r',encoding='utf-8') as meta_file:
        image_meta = json.load(meta_file)
    return image_meta