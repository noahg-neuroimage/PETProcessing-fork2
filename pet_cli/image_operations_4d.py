""""
4D pet tools.
"""
import json
import ants
import nibabel
from nibabel import processing
import numpy as np
from . import image_io
from . import image_reg


ImageIO = image_io.ImageIO

class ImageOps4D():
    """
    A class, supplies tools to modify values of 4D images.
    
    Attributes:
        images (list[ImageIO]): Enforced order: PET, MRI, segmentation.
        image_meta (dict): Image metadata pulled from BIDS-compliant json file.
        half_life (float): Half life of radioisotope to be used for computations.
                           Default value 0.
    
    See Also:
        :class: `ImageIO`
    """
    def __init__(self,
        image_paths: list[str],
        out_path: str,
        half_life: float=0,
        color_table_path: str=None,
        verbose: bool=True
    ):
        """
        Constructor for ImageOps4D

        Args:
        
        """
        self.image_paths = image_paths
        # NB protected keywords: {'pet': pet_path,'mri': mri_path,'seg': seg_path}
        self.half_life = half_life
        self.out_path = out_path
        self.color_table_path = color_table_path
        self.verbose = verbose


    def weighted_series_sum(self) -> np.ndarray:
        """
        Sum a 4D image series weighted based on time and re-corrected for decay correction.

        Args:
            pet_series (np.ndarray): Input pet image to be summed.
            image_meta (dict): Metadata json file following BIDS standard, from which
                               we collect frame timing and decay correction info.
            half_life (float): Half life of the PET radioisotope in seconds.

        Returns:
            summed_image (np.ndarray): Summed image 

        Credit to Avi Snyder who wrote the original version of this code in C.
        """
        pet_meta = image_io.load_meta(self.image_paths['pet'])
        pet_image = nibabel.load(self.image_paths['pet'])
        pet_series = pet_image.get_fdata()
        image_frame_start = pet_meta['FrameTimesStart']
        image_frame_duration = pet_meta['FrameDuration']
        image_decay_correction = pet_meta['DecayCorrectionFactor']
        tracer_isotope = pet_meta['TracerRadionuclide']
        if self.verbose:
            print(f"(ImageOps4D): Radio isotope is {tracer_isotope}",
                   "with half life {self.half_life} s")
        decay_constant = np.log(2) / self.half_life

        image_total_duration = np.sum(image_frame_duration)
        total_decay    = decay_constant * image_total_duration / \
            (1-np.exp(-1*decay_constant*image_total_duration)) / \
                np.exp(-1*decay_constant*image_frame_start[0])

        pet_series_scaled = pet_series[:,:,:] \
            * image_frame_duration \
            / image_decay_correction
        pet_series_sum_scaled = np.sum(pet_series_scaled,axis=3)
        image_weighted_sum = pet_series_sum_scaled * total_decay / image_total_duration

        pet_sumimg = nibabel.nifti1.Nifti1Image(
            dataobj=image_weighted_sum,
            affine=pet_image.affine,
            header=pet_image.header
        )
        self.image_paths['pet_sumimg'] = f'{self.out_path}/sum_img/TEMP.nii'
        nibabel.save(pet_sumimg,self.image_paths['pet_sumimg'])

        return image_weighted_sum


    def motion_correction(self) -> np.ndarray:
        """
        Motion correct PET series
        """
        pet_nibabel = nibabel.load(self.image_paths['pet'])
        pet_sumimg = nibabel.load(self.image_paths['pet_sumimg'])
        pet_sumimg_ants = ants.from_nibabel(pet_sumimg)
        pet_ants = ants.from_nibabel(pet_nibabel)
        pet_moco_ants_dict = ants.motion_correction(pet_ants,
            pet_sumimg_ants,
            type_of_transform='Rigid')
        pet_moco_ants = pet_moco_ants_dict['motion_corrected']
        pet_moco_pars = pet_moco_ants_dict['motion_parameters']
        pet_moco_np = pet_moco_ants.numpy()
        pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)
        self.image_paths['pet_moco'] = f'{self.out_path}/moco/TEMP.nii' # TODO: fix path saving
        nibabel.save(pet_moco_nibabel,self.image_paths['pet_moco'])
        return pet_moco_np, pet_moco_pars


    def register_pet(self) -> nibabel.nifti1.Nifti1Image:
        """
        Perform registration PET -> MRI
        """
        pet_sumimg = nibabel.load(self.image_paths['pet_sumimg'])
        mri_img = nibabel.load(self.image_paths['mri'])
        pet_moco = nibabel.load(self.image_paths['pet_moco'])
        dummy = image_reg.ImageReg()
        _reg_out, xfm_path, _xfm_mat = dummy.rigid_registration(
            pet_sumimg,mri_img)
        pet_reg = dummy.apply_xfm(pet_moco,mri_img,xfm_path)
        self.image_paths['pet_moco_reg'] = f'{self.out_path}/registration/TEMP.nii'
        nibabel.save(pet_reg,self.image_paths['pet_moco_reg'])

    def mask_img_to_vals(self,
                         values: list[int],
                         resample_seg: bool=False) -> np.ndarray:
        """
        Masks an input image based on a value or list of values, and returns an array
        with original image values in the regions based on values specified for the mask.

        Args:
            values (list[int]): List of values corresponding to regions to be masked.
            resample_seg (bool): Determines whether or not to resample the segmentation.
                                 Set to True when the PET input (registered to MPR) and
                                 segmentation are different resolutions.

        Returns:
            masked_image (np.ndarray): Masked image
        """
        pet_image = nibabel.load(self.image_paths['pet_moco_reg'])
        seg_image = nibabel.load(self.image_paths['seg'])
        pet_series = pet_image.get_fdata()
        num_frames = pet_series.shape[3]
        if not self.image_paths['seg_resampled']:
            if resample_seg:
                image_first_frame = pet_series[:,:,:,0]
                seg_resampled = processing.resample_from_to(from_img=seg_image,
                                    to_vox_map=(image_first_frame.shape,
                                    pet_image.affine),
                                    order=0)
                self.image_paths['seg_resampled'] = f'{self.out_path}/segmentation/TEMP.nii'
                nibabel.save(seg_resampled,self.image_paths['seg_resampled'])
            else:
                self.image_paths['seg_resampled'] = self.image_paths['seg']

        seg_for_masking = nibabel.load(self.image_paths['seg_resampled']).get_fdata()

        for region in values:
            masked_voxels = seg_for_masking==region
            masked_image = pet_series[masked_voxels].reshape((-1,num_frames))
            tac_out = np.mean(masked_image,axis=0)
        return tac_out


    def write_tacs(self):
        """
        Function to write Tissue Activity Curves for each region, given a segmentation,
        4D PET image, and color table. Computes the average of the PET image within each
        region. Writes a JSON for each region with region name, frame start time, and mean 
        value within region.

        Args:

        Returns:
        """
        pet_meta = image_io.load_meta(self.image_paths['pet'])
        with open(self.color_table_path,'r',encoding='utf-8') as color_table_file:
            color_table = json.load(color_table_file)
        regions_list = color_table['data']
        for region_pair in regions_list:
            region_index, region_name = region_pair
            region_json = {'region_name': region_name}
            series_means = self.mask_img_to_vals([region_index]).tolist()
            region_json['frame_start_time'] = pet_meta['FrameTimesStart']
            region_json['activity'] = series_means
            with open(f'{self.out_path}/tacs/{region_name}-tac.json',
                      'w',encoding='ascii') as out_file:
                json.dump(obj=region_json,fp=out_file,indent=4)
        return 0
