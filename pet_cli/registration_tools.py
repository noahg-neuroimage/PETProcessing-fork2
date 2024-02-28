"""
Image utilities
"""
import json
import ants
import nibabel
from nibabel import processing
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np
import h5py

class ImageIO():
    """
    Class handling 3D and 4D image file utilities.
    """
    def __init__(self, file_path: str, verbose: bool=True):
        """
        Args:
            file_path (str): Path to existing Nifti image file.
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        
        """
        self.file_path = file_path
        self.verbose = verbose


    def load_nii(self) -> FileBasedImage:
        """
        Wrapper to load nifti from file_path.

        Returns:
            The nifti FileBasedImage.

        """
        image = nibabel.load(self.file_path)

        if self.verbose:
            print(f"(ImageIO): {self.file_path} loaded")

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


class ImageReg(ImageIO):
    """
    A class, extends ``ImageIO``, supplies tools to compute and run image registrations.
    Attributes:
    """
    def __init__(self, file_path: str, verbose: bool):
        """
        Args:
            file_path (str): Path to existing Nifti image file.
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        """
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


    def h5_parse(self, h5_file: str) -> np.ndarray:
        """
        Parse an h5 transformation file into an affine transform.

        Args:
            h5_path (str): Path to an h5 transformation file.

        Returns:
            xfm_mat (np.ndarray): Affine transform.
        """
        xfm_hdf5 = h5py.File(h5_file)
        xfm_mat  = xfm_hdf5['TransformGroup']['1']['TransformParameters'][:] \
                   .reshape((4,3))

        return xfm_mat


    def rigid_registration(self,
                           moving_image: nibabel.nifti1.Nifti1Image,
                           fixed_image: nibabel.nifti1.Nifti1Image
                           ) -> tuple[nibabel.nifti1.Nifti1Image,str,np.ndarray]:
        """
        Register two images under rigid transform assumptions and return the transformed 
        image and parameters.

        Args:
            moving_image (nibabel.nifti1.Nifti1Image): Image to be registered
            fixed_image (nibabel.nifti1.Nifti1Image): Reference image to be registered to

        Returns:
            mov_on_fix (nibabel.nifti1.Nifti1Image): Moving image on reference fixed image
            xfm_file (str): Path to the composite h5 transform written to file. Reference
                            for using h5 files can be found at:
                            https://open.win.ox.ac.uk/pages/fsl/fslpy/fsl.transform.x5.html
            out_mat (np.ndarray): affine transform matrix of parameters from moving to fixed.
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)

        xfm_output = ants.registration(
            moving=moving_image_ants,
            fixed=fixed_image_ants,
            type_of_transform='DenseRigid',
            write_composite_transform=True
        ) # NB: this is a dictionary!


        mov_on_fix_ants = xfm_output['warpedmovout']
        mov_on_fix = ants.to_nibabel(mov_on_fix_ants)
        xfm_file = xfm_output['fwdtransforms']
        out_mat = self.h5_parse(xfm_file)

        return mov_on_fix, xfm_file, out_mat


    def apply_xfm(self,
                  moving_image: nibabel.nifti1.Nifti1Image,
                  fixed_image: nibabel.nifti1.Nifti1Image,
                  xfm_path: str) -> nibabel.nifti1.Nifti1Image:
        """
        Register a moving image to a fixed image using a supplied transform.

        Args:
            moving_image (nibabel.nifti1.Nifti1Image): Image to be resampled onto fixed
                reference image.
            fixed_image (nibabel.nifti1.Nifti1Image): Reference image onto which the 
                moving_image is registered.
            xfm_matrix (nibabel.nifti1.Nifti1Image): Ants-style transformation matrix used
                to apply transformation.

        Returns:
            mov_on_fix (nibabel.nifti1.Nifti1Image): Moving image registered onto the fixed image.
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)
        img_type = len(fixed_image.shape) - 1  # specific to ants: 4D image has type 3

        mov_on_fix_ants = ants.apply_transforms(
            moving=moving_image_ants,
            fixed=fixed_image_ants,
            transformlist=xfm_path,
            imagetype=img_type)

        mov_on_fix = ants.to_nibabel(mov_on_fix_ants)

        return mov_on_fix

# Can we use numba?
class ImageOps4D(ImageIO):
    """
    A class, extends ``ImageIO``, supplies tools to modify values of 4D images.
    
    Attributes:
        image_meta (dict): Image metadata pulled from BIDS-compliant json file.
        half_life (float): Half life of radioisotope to be used for computations.
                           Default value 0.
    
    See Also:
        :class:`ImageIO`
    """
    def __init__(self,
                 image: ImageIO,
                 image_meta: str,
                 out_dir: str,
                 seg_image: nibabel.nifti1.Nifti1Image,
                 half_life: float=0):
        """
        Constructor for ImageOps4D

        Args:
        
        """
        file_path = image.file_path
        verbose = image.verbose
        super().__init__(file_path, verbose)
        self.image_meta = image_meta
        self.half_life = half_life
        self.pet_image = self.load_nii()
        self.image_series = self.pet_image.get_fdata()
        self.pet_upsampled = None
        self.out_dir = out_dir
        self.seg_image = seg_image

    def weighted_series_sum(self) -> np.ndarray:
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
        image_frame_start = self.image_meta['FrameTimesStart']
        image_frame_duration = self.image_meta['FrameDuration']
        image_decay_correction = self.image_meta['DecayCorrectionFactor']
        tracer_isotope = self.image_meta['TracerRadionuclide']
        if self.verbose:
            print(f"(ImageOps4D): Radio isotope is {tracer_isotope} \
                   with half life {self.half_life} s")
        decay_constant = np.log(2) / self.half_life

        image_total_duration = np.sum(image_frame_duration)
        total_decay    = decay_constant * image_total_duration / \
            (1-np.exp(-1*decay_constant*image_total_duration)) / \
                np.exp(-1*decay_constant*image_frame_start[0])

        image_series_scaled = self.image_series[:,:,:] \
            * image_frame_duration \
            / image_decay_correction
        image_series_sum_scaled = np.sum(image_series_scaled,axis=3)
        image_weighted_sum = image_series_sum_scaled * total_decay / image_total_duration

        return image_weighted_sum


    def mask_img_to_vals(self,
                         values: list[int]) -> np.ndarray:
        """
        Masks an input image based on a value or list of values, and returns an array
        with original image values in the regions based on values specified for the mask.

        Args:
            values (list[int]): List of values corresponding to regions to be masked.

        Returns:
            masked_image (np.ndarray): Masked image
        """


        image_fdata = self.image_series
        image_first_frame = image_fdata[:,:,:,0]
        num_frames = image_fdata.shape[3]
        seg_resampled = processing.resample_from_to(from_img=self.seg_image,
                                               to_vox_map=(image_first_frame.shape,
                                                           self.pet_image.affine),
                                               order=0)
        seg_fdata = seg_resampled.get_fdata()

        #masked_image = np.zeros(image_fdata.shape)
        for region in values:
            masked_voxels = seg_fdata==region
            masked_image = image_fdata[masked_voxels].reshape((-1,num_frames))
            tac_out = np.mean(masked_image,axis=0)
        return tac_out


    def write_tacs(self,ctab):
        """
        Function to write Tissue Activity Curves for each region, given a segmentation,
        4D PET image, and color table. Computes the average of the PET image within each
        region. Writes a JSON for each region with region name, frame start time, and mean 
        value within region.

        Args:

        Returns:
        """
        regions_list = ctab['data']
        for region_pair in regions_list:
            region_index, region_name = region_pair
            region_json = {'region_name': region_name}
            series_means = self.mask_img_to_vals([region_index]).tolist()
            region_json['frame_start_time'] = self.image_meta['FrameTimesStart']
            region_json['activity'] = series_means
            with open(f'{self.out_dir}/tacs/{region_name}-tac.json',
                      'w',encoding='ascii') as out_file:
                json.dump(obj=region_json,fp=out_file,indent=4)
        return 0
