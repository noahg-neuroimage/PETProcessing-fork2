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



    # File I/O functions: these will be in subclass ImageIO
    def load_nii(self) -> FileBasedImage:
        """
        Wrapper to load nifti from file path.

        Args:
            file_path: The full file path to the nifti file.
            verbose:

        Returns:
            The nifti FileBasedImage.

        """
        image = nibabel.load(self.file_path)

        if self.verbose:
            print(f"(fileIO): {self.file_path} loaded")

        return image


    def save_nii(self,image,output_path) -> int:
        """
        Wrap to save nifti to file path.

        Args:


        Returns:
        """
        nibabel.save(image,output_path)

        return 0

    def extract_image_from_nii_as_numpy(self, image: nibabel.nifti1) -> np.ndarray:
        """
        Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array.

        Args:
            image: The nifti file itself.
            verbose:

        Returns:
            The data contained in the .nii or .nii.gz file as a numpy array.

        """
        image_data = image.get_fdata()

        if self.verbose:
            print(f"(fileIO): Image has shape {image_data.shape}")

        return image_data


    def extract_header_from_nii(self, image: np.ndarray) -> FileBasedHeader:
        """
        Convenient wrapper to extract header information from a .nii or .nii.gz 
        file as a nibabel file-based header.

        Args:
            image: The nifti file itself.

        Returns:
            The nifti header.
        """
        image_header: FileBasedHeader = image.header

        if self.verbose:
            print(f"(fileIO): Image header is: {image_header}")

        return image_header
    

    def extract_np_to_nibabel(self, image: np.ndarray, header: FileBasedHeader, affine: np.ndarray) -> nibabel.nifti1:
        """
        Wrapper to convert an image array into nibabel object.
        
        """

        return 0


    def reorient_to_ras(self, image: np.ndarray) -> np.ndarray:
        """
        Wrapper for the RAS reorientation used to ensure images are oriented the same.

        Args:
            image: The nifti file itself.
            verbose:

        Returns:
            The reoriented nifti file.
        """
        reoriented_image: np.ndarray = nibabel.as_closest_canonical(image)

        if self.verbose:
            print("(fileIO): Image has been reoriented to RAS")

        return reoriented_image


    def rigid_registration_calc(self,
                           moving_image: np.ndarray,
                           fixed_image: np.ndarray,
                           ) -> np.ndarray:
        """
        Register two images and return the transformation matrix

        Args:
            moving_image (np.ndarray): Image to be registered
            fixed_image (np.ndarray): Reference image to be registered to

        Returns:
            rigid_transform (np.ndarray): Rigid transform array
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)

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
    A class, extends ``ImageIO``, has tools to modify values of 4D images.
    
    Attricbutes:
        verbose (bool): 

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

        Credit to Avi Snyder who wrote the original version of this code.
        """
        image_frame_start = image_meta['FrameTimesStart']
        image_frame_duration = image_meta['FrameDuration']
        image_decay_correction = image_meta['DecayCorrectionFactor']
        # TODO: Create a function to read half life from isotope
        tracer_isotope = image_meta['TracerRadionuclide']
        if self.verbose:
            print(f"(ImageOps4D): Radio isotope is {tracer_isotope} with half life {half_life} s")
        decay_constant = np.log(2) / half_life

        image_total_duration = np.sum(image_frame_duration)
        total_decay    = decay_constant * image_total_duration / \
            (1-np.exp(-1*decay_constant*image_total_duration)) / \
                np.exp(-1*decay_constant*image_frame_start[0])

        image_series_scaled = image_series[:,:,:] * image_frame_duration / image_decay_correction
        # TODO: Create a malleable solution to sum axis
        image_series_sum_scaled = np.sum(image_series_scaled,axis=3)
        image_weighted_sum = image_series_sum_scaled * total_decay / image_total_duration

        return image_weighted_sum
