"""
Image utilities
"""
import ants
import nibabel.nifti1
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np


class ImageUtil():
    """
    Class handling 3D and 4D image file utilities.
    """
    def __init__(self, file_path, verbose):
        self.file_path = file_path
        self.verbose = verbose


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
        image_data: np.ndarray = image.get_fdata()

        if self.verbose:
            print(f"(fileIO): Image has shape {image_data.shape}")

        return image_data


    def extract_header_from_nii(self, image: nibabel.nifti1) -> FileBasedHeader:
        """
        Convenient wrapper to extract header information from a .nii or .nii.gz 
        file as a nibabel file-based header.

        Args:
            image: The nifti file itself.
            verbose:

        Returns:
            The nifti header.
        """
        image_header: FileBasedHeader = image.header

        if self.verbose:
            print(f"(fileIO): Image header is: {image_header}")

        return image_header


    def reorient_to_ras(self, image: nibabel.nifti1) -> nibabel.nifti1:
        """
        Wrapper for the RAS reorientation used to ensure images are oriented the same.

        Args:
            image: The nifti file itself.
            verbose:

        Returns:
            The reoriented nifti file.
        """
        reoriented_image: nibabel.nifti1 = nibabel.as_closest_canonical(image)

        if self.verbose:
            print("(fileIO): Image has been reoriented to RAS")

        return reoriented_image


    def convert_nib_to_ants(self, nib_image: nibabel.nifti1) -> ants.ANTsImage:
        """
        Converts nibabel image format (nibabel.nifti1) to ants image format (ANTsImage).

        Args:
            nib_image: Image in nibabel.nifti1 format.

        Returns:
            Image in ANTsImage format.
        """
        nib_image_data: np.ndarray = nib_image.get_fdata()
        nib_image_affine = nib_image.affine
        ants_image = ants.from_numpy(nib_image_data)
        ants_image.set_spacing(tuple(np.diag(nib_image_affine)[:3]))

        return ants_image


    def rigid_registration_calc(self,
                           moving_image: nibabel.nifti1,
                           fixed_image: nibabel.nifti1,
                           ) -> np.ndarray:
        """
        Register two images and return the transformation matrix

        Args:
            moving_image (nibabel.nifti1): Image to be registered
            fixed_image (nibabel.nifti1): Reference image to be registered to

        Returns:
            rigid_transform (np.ndarray): Rigid transform array
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)
        print(moving_image_ants)
        print(fixed_image_ants)

        return 1


    def weighted_series_sum(self,
                            image_series,
                            image_meta,
                            half_life: float) -> nibabel.nifti1:
        """
        Sum a 4D image series weighted based on time and re-corrected for decay correction.

        Args:
            image_series (nibabel.nifti1): Input image to be summed.
            image_meta (json file?): Metadata json file following BIDS standard, from which
                                     we collect frame timing and decay correction info.

        Returns:
            summed_image (nibabel.nifti1): Summed image 
        """
        image_frame_start = image_meta['FrameTimesStart']
        image_frame_duration = image_meta['FrameDuration']
        image_decay_correction = image_meta['DecayCorrectionFactor']
        # TODO: Create a function to read half life from isotope
        #tracer_isotope = image_meta['TracerRadionuclide']
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
