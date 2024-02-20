"""
Image utilities
"""
import ants
import nibabel
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np


class ImageIO():
    """
    Class handling 3D and 4D image file utilities.
    """
    def __init__(self, file_path: str, verbose: bool=True):
        """
        Args:
            file_path (str): Path to existing nifti image to be read.
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        
        """
        self.file_path = file_path
        self.verbose = verbose


    def load_nii(self) -> FileBasedImage:
        """
        Wrapper to load nifti from file path.

        Returns:
            The nifti FileBasedImage.

        """
        image = nibabel.load(self.file_path)

        if self.verbose:
            print(f"(ImageIO): {self.file_path} loaded")

        return image


    def save_nii(self,image: nibabel.nifti1.Nifti1Image,output_path: str) -> int:
        """
        Wrap to save nifti to file path.

        Args:
            image (nibabel.nifti1.Nifti1Image): Nibabel-type image to write to file.
            output_path (str): Path to which image will be written.
        """
        nibabel.save(image,output_path)

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



class ImageReg(ImageIO):
    """
    A class, extends ``ImageIO``, supplies tools to compute and run image registrations.
    Attributes:
        xfm_path (str): Path to image transformation file.
    """
    def __init__(self, file_path: str, verbose: bool, xfm_path: str = None):
        """
        Args:
            file_path (str): Path to existing nifti image to be read.
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        """
        self.xfm_path = xfm_path
        super().__init__(file_path, verbose)


    def reorient_to_ras(self, image: nibabel.nifti1.Nifti1Image) -> nibabel.nifti1.Nifti1Image:
        """
        Wrapper for the RAS reorientation used to ensure images are oriented the same.

        Args:
            image: Nibabel-type image to write to file.

        Returns:
            The reoriented nifti file.
        """
        reoriented_image = nibabel.as_closest_canonical(image)

        if self.verbose:
            print("(ImageReg): Image has been reoriented to RAS")

        return reoriented_image


    def rigid_registration_calc(self,
                           moving_image: np.ndarray,
                           fixed_image: np.ndarray,
                           moving_affine: np.ndarray,
                           fixed_affine: np.ndarray
                           ) -> np.ndarray:
        """
        Register two images and return the transformation matrix

        Args:
            moving_image (np.ndarray): Image to be registered
            fixed_image (np.ndarray): Reference image to be registered to

        Returns:
            rigid_transform (np.ndarray): Rigid transform array
        """
        moving_image_ants = ants.from_numpy(moving_image) # TODO: add origin, spacing, direction from affine
        #moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_numpy(fixed_image)

        _mov_fix_ants,_fix_mov_ants,mov_fix_xfm,_fix_mov_xfm = ants.registration(
            fixed_image_ants,
            moving_image_ants,
            type_of_transform='rigid'
        )

        return mov_fix_xfm


    def apply_registration(self,
                      moving_image: nibabel.nifti1,
                      fixed_image: nibabel.nifti1,
                      xfm_matrix: np.ndarray) -> np.ndarray:
        """
        Register a moving image to a fixed image using a supplied transform.

        Args:
            moving_image (np.ndarray): Image to be resampled onto fixed reference image.
            fixed_image (np.ndarray): Reference image onto which the moving_image is registered.
            xfm_matrix (np.ndarray): Ants-style transformation matrix used to apply transformation.

        Returns:
            moving_on_fixed_image (np.ndarray): Moving image registered onto the fixed image.
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)

        mov_fix_ants = ants.apply_transforms(
            fixed_image_ants,
            moving_image_ants,
            transformlist=xfm_matrix)

        moving_on_fixed_image = ants.to_nibabel(mov_fix_ants)

        return moving_on_fixed_image

# Can we use numba?
class ImageOps4D(ImageIO):
    """
    A class, extends ``ImageIO``, supplies tools to modify values of 4D images.
    
    Attributes:

    See Also:
        :class:`ImageIO`
    """
    def __init__(self, file_path: str, verbose: bool):
        super().__init__(file_path, verbose)

    def weighted_series_sum(self,
                            image_series: np.ndarray,
                            image_meta: dict,
                            half_life: float) -> np.ndarray:
        """
        Sum a 4D image series weighted based on time and re-corrected for decay correction.

        Args:
            image_series (np.ndarray): Input image to be summed.
            image_meta (dict): Metadata json file following BIDS standard, from which
                                     we collect frame timing and decay correction info.
            half_life (float): Half life of the PET radioisotope in seconds.

        Returns:
            summed_image (np.ndarray): Summed image 

        Credit to Avi Snyder who wrote the original version of this code in C.
        """
        image_frame_start = image_meta['FrameTimesStart']
        image_frame_duration = image_meta['FrameDuration']
        image_decay_correction = image_meta['DecayCorrectionFactor']
        tracer_isotope = image_meta['TracerRadionuclide']
        if self.verbose:
            print(f"(ImageOps4D): Radio isotope is {tracer_isotope} with half life {half_life} s")
        decay_constant = np.log(2) / half_life

        image_total_duration = np.sum(image_frame_duration)
        total_decay    = decay_constant * image_total_duration / \
            (1-np.exp(-1*decay_constant*image_total_duration)) / \
                np.exp(-1*decay_constant*image_frame_start[0])

        image_series_scaled = image_series[:,:,:] * image_frame_duration / image_decay_correction
        image_series_sum_scaled = np.sum(image_series_scaled,axis=3)
        image_weighted_sum = image_series_sum_scaled * total_decay / image_total_duration

        return image_weighted_sum
