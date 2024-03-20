""""
4D pet tools.
"""
import os
import json
import ants
import nibabel
from nibabel import processing
import numpy as np
from . import image_io
from . import math_lib


def weighted_series_sum(
    input_image_4d_path: str,
    out_image_path: str,
    half_life: float,
    verbose: bool
) -> np.ndarray:
    """
    Sum a 4D image series weighted based on time and re-corrected for decay correction.

    Args:
        pet_series (np.ndarray): Input pet image to be summed.
        image_meta (dict): Metadata json file following BIDS standard, from which
                            we collect frame timing and decay correction info.
        half_life (float): Half life of the PET radioisotope in seconds.

    Returns:
        summed_image (np.ndarray): Summed image 
    """
    if half_life is None:
        raise ValueError('(ImageOps4d): Radioisotope half life not set, cannot \
            run weighted_series_sum.')
    pet_meta = image_io.ImageIO.load_meta(input_image_4d_path)
    pet_image = nibabel.load(input_image_4d_path)
    pet_series = pet_image.get_fdata()
    image_frame_start = pet_meta['FrameTimesStart']
    image_frame_duration = pet_meta['FrameDuration']
    if 'DecayCorrectionFactor' in pet_meta.keys():
        image_decay_correction = pet_meta['DecayCorrectionFactor']
    elif 'DecayFactor' in pet_meta.keys():
        image_decay_correction = pet_meta['DecayFactor']
    if 'TracerRadionuclide' in pet_meta.keys():
        tracer_isotope = pet_meta['TracerRadionuclide']
        if verbose:
            print(f"(ImageOps4d): Radio isotope is {tracer_isotope}",
                "with half life {self.half_life} s")
    image_weighted_sum = math_lib.weighted_sum_computation(
        image_frame_duration,
        half_life,
        pet_series,
        image_frame_start,
        image_decay_correction
    )
    pet_sum_image = nibabel.nifti1.Nifti1Image(
        dataobj=image_weighted_sum,
        affine=pet_image.affine,
        header=pet_image.header
    )
    nibabel.save(pet_sum_image,out_image_path)
    if verbose:
        print(f"(ImageOps4d): weighted sum image saved to {out_image_path}")
    return pet_sum_image


def motion_correction(
    input_image_4d_path: str,
    reference_image_path: str,
    out_image_path: str,
    verbose: bool
) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Motion correct PET image series.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy array.
        pet_moco_pars (list[str]): List of ANTS registration files applied to each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure corresponding 
            to each frame transform.
    """
    pet_nibabel = nibabel.load(input_image_4d_path)
    pet_sum_image = nibabel.load(reference_image_path)
    pet_ants = ants.from_nibabel(pet_nibabel)
    pet_sum_image_ants = ants.from_nibabel(pet_sum_image)
    pet_moco_ants_dict = ants.motion_correction(pet_ants,
        pet_sum_image_ants,
        type_of_transform='Rigid')
    if verbose:
        print('(ImageOps4D): motion correction finished.')
    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_pars = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)
    nibabel.save(pet_moco_nibabel,out_image_path)
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_pars, pet_moco_fd


def register_pet(
    input_calc_image_path: str,
    input_reg_image_path: str,
    reference_image_path: str,
    out_image_path: str,
    verbose: bool
):
    """
    Register PET image series to anatomical data. Computes transform based on weighted average,
    which is then applied to the 4D PET image series.
    """
    pet_sum_image = ants.image_read(input_calc_image_path)
    mri_image = ants.image_read(reference_image_path)
    pet_moco = ants.image_read(input_reg_image_path)
    xfm_output = ants.registration(
        moving=pet_sum_image,
        fixed=mri_image,
        type_of_transform='DenseRigid',
        write_composite_transform=True)
    if verbose:
        print(f'Registration computed transforming image {input_calc_image_path} to',
              f'{reference_image_path} space')
    xfm_apply = ants.apply_transforms(
        moving=pet_moco,
        fixed=mri_image,
        transformlist=xfm_output['fwdtransforms'],
        imagetype=3)
    if verbose:
        print(f'Registration applied to {input_reg_image_path}')
    ants.image_write(xfm_apply,out_image_path)
    if verbose:
        print(f'Transformed image saved to {out_image_path}')


def resample_segmentation(
    input_image_4d_path: str,
    segmentation_image_path: str,
    out_seg_path: str,
    verbose: bool
):
    """
    Resamples a segmentation to 4D PET series affine

    Args:

    Returns:
    """
    pet_image = nibabel.load(input_image_4d_path)
    seg_image = nibabel.load(segmentation_image_path)
    pet_series = pet_image.get_fdata()
    image_first_frame = pet_series[:,:,:,0]
    seg_resampled = processing.resample_from_to(from_img=seg_image,
                        to_vox_map=(image_first_frame.shape,
                        pet_image.affine),
                        order=0)
    nibabel.save(seg_resampled,out_seg_path)
    if verbose:
        print(f'Resampled segmentation saved to {out_seg_path}')


def mask_image_to_vals(
    input_image_4d_path: str,
    segmentation_image_path: str,
    values: list[int],
    verbose: bool,
) -> np.ndarray:
    """
    Masks an input image based on a value or list of values, and returns an array
    with original image values in the regions based on values specified for the mask.

    Args:
        values (list[int]): List of values corresponding to regions to be masked.
        resample_seg (bool): Determines whether or not to resample the segmentation.
            Set to True when the PET input (registered to MPR) and segmentation are 
            different resolutions.

    Returns:
        tac_out (np.ndarray): Mean of values within mask for each frame in 4D PET series.
    """
    pet_image_4d = nibabel.load(input_image_4d_path).get_fdata()
    seg_image = nibabel.load(segmentation_image_path).get_fdata()
    num_frames = pet_image_4d.shape[3]
    for region in values:
        if verbose:
            print(f'Running TAC for region index {region}')
        masked_voxels = seg_image==region
        masked_image = pet_image_4d[masked_voxels].reshape((-1,num_frames))
        tac_out = np.mean(masked_image,axis=0)
    return tac_out


def write_tacs(
    input_image_4d_path: str,
    color_table_path: str,
    segmentation_image_path: str,
    out_tac_path: str,
    verbose: bool
):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and color table. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """
    pet_meta = image_io.ImageIO.load_meta(input_image_4d_path)
    with open(color_table_path,'r',encoding='utf-8') as color_table_file:
        color_table = json.load(color_table_file)
    regions_list = color_table['data']
    for region_pair in regions_list:
        region_index, region_name = region_pair
        region_json = {'region_name': region_name}
        series_means = mask_image_to_vals(
            input_image_4d_path=input_image_4d_path,
            segmentation_image_path=segmentation_image_path,
            values=[region_index],
            verbose=verbose
        ).tolist()
        region_json['frame_start_time'] = pet_meta['FrameTimesStart']
        region_json['activity'] = series_means
        tac_path = os.path.join(f'{out_tac_path}','tacs')
        with open(os.path.join(tac_path,f'-{region_name}-tac.json'),
                    'w',encoding='ascii') as out_file:
            json.dump(obj=region_json,fp=out_file,indent=4)


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
        sub_id: str,
        out_path: str,
        image_paths: dict=None,
        half_life: float=None,
        color_table_path: str=None,
        verbose: bool=True
    ):
        """
        Constructor for ImageOps4d

        Args:
            sub_id (str):
            image_paths (dict): Dictionary containing paths to files used in relevant
                preprocessing steps. This variable has protected 
            out_path (str): Path to output directory to which preprocessing files are written.
            half_life (float): Half life of tracer radioisotope used for the study in seconds.
                Default value 0.
            color_table_path (str): Path to location of color table, matching region names to
                region values in a segmentation file. See:
                https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles.
            verbose (bool): Control output of debugging information. Default value True.
        """
        self.sub_id = sub_id
        if image_paths is None:
            image_paths = {}
        self.image_paths = image_paths
        # NB protected keywords: {'pet': pet_path,'mri': mri_path,'seg': seg_path}
        self.half_life = half_life
        self.out_path = out_path
        self.color_table_path = color_table_path
        self.verbose = verbose


    def run_weighted_series_sum(self) -> np.ndarray:
        """
        Sum a 4D image series weighted based on time and re-corrected for decay correction.

        Args:
            pet_series (np.ndarray): Input pet image to be summed.
            image_meta (dict): Metadata json file following BIDS standard, from which
                               we collect frame timing and decay correction info.
            half_life (float): Half life of the PET radioisotope in seconds.

        Returns:
            summed_image (np.ndarray): Summed image 
        """
        sum_image_path = os.path.join(self.out_path,'sum_image')
        os.makedirs(sum_image_path,exist_ok=True)
        self.image_paths['pet_sum_image'] = os.path.join(
            sum_image_path,
            f'{self.sub_id}-sum.nii.gz')
        weighted_series_sum(
            input_image_4d_path=self.image_paths['pet'],
            out_image_path=self.image_paths['pet_sum_image'],
            half_life=self.half_life,
            verbose=self.verbose
        )


    def run_motion_correction(self) -> tuple[np.ndarray, list[str], list[float]]:
        """
        Motion correct PET image series.

        Returns:
            pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy array.
            pet_moco_pars (list[str]): List of ANTS registration files applied to each frame.
            pet_moco_fd (list[float]): List of framewise displacement measure corresponding 
                to each frame transform.
        """
        moco_path = os.path.join(self.out_path,'motion-correction')
        os.makedirs(moco_path,exist_ok=True)
        self.image_paths['pet_moco'] = os.path.join(moco_path,f'{self.sub_id}-moco.nii.gz')
        motion_correction(
            input_image_4d_path=self.image_paths['pet'],
            reference_image_path=self.image_paths['pet_sum_image'],
            out_image_path=self.image_paths['pet_moco'],
            verbose=self.verbose
        )


    def run_register_pet(self):
        """
        Register PET image series to anatomical data. Computes transform based on weighted average,
        which is then applied to the 4D PET image series.
        """
        reg_path = os.path.join(self.out_path,'registration')
        os.makedirs(reg_path,exist_ok=True)
        self.image_paths['pet_moco_reg'] = os.path.join(reg_path,f'{self.sub_id}-reg.nii.gz')
        register_pet(
            input_calc_image_path=self.image_paths['pet_sum_image'],
            input_reg_image_path=self.image_paths['pet_moco'],
            reference_image_path=self.image_paths['mri'],
            out_image_path=self.image_paths['pet_moco_reg'],
            verbose=self.verbose
        )


    def run_mask_image_to_vals(self,
                         values: list[int],
                         resample_seg: bool=False) -> np.ndarray:
        """
        Masks an input image based on a value or list of values, and returns an array
        with original image values in the regions based on values specified for the mask.

        Args:
            values (list[int]): List of values corresponding to regions to be masked.
            resample_seg (bool): Determines whether or not to resample the segmentation.
                Set to True when the PET input (registered to MPR) and segmentation are 
                different resolutions.

        Returns:
            tac_out (np.ndarray): Mean of values within mask for each frame in 4D PET series.
        """
        if resample_seg:
            seg_res_path = os.path.join(self.out_path,'segmentation')
            os.makedirs(seg_res_path,exist_ok=True)
            self.image_paths['seg_resampled'] = os.path.join(
                seg_res_path,
                f'{self.sub_id}-segmentation-resampled.nii.gz')
            resample_segmentation(
                input_image_4d_path=self.image_paths['pet_moco_reg'],
                segmentation_image_path=self.image_paths['seg'],
                out_seg_path=self.image_paths['seg_resampled'],
                verbose=self.verbose
            )
        tac_out = mask_image_to_vals(
            input_image_4d_path=self.image_paths['pet_moco_reg'],
            segmentation_image_path=self.image_paths['seg_resampled'],
            values=values,
            verbose=self.verbose
        )
        return tac_out


    def run_write_tacs(self):
        """
        Function to write Tissue Activity Curves for each region, given a segmentation,
        4D PET image, and color table. Computes the average of the PET image within each
        region. Writes a JSON for each region with region name, frame start time, and mean 
        value within region.
        """
        pet_meta = image_io.ImageIO.load_meta(self.image_paths['pet'])
        with open(self.color_table_path,'r',encoding='utf-8') as color_table_file:
            color_table = json.load(color_table_file)
        regions_list = color_table['data']
        res = True
        for region_pair in regions_list:
            region_index, region_name = region_pair
            region_json = {'region_name': region_name}
            series_means = self.run_mask_image_to_vals([region_index],resample_seg=res).tolist()
            res=False
            region_json['frame_start_time'] = pet_meta['FrameTimesStart']
            region_json['activity'] = series_means
            tac_path = os.path.join(f'{self.out_path}','tacs')
            os.makedirs(tac_path,exist_ok=True)
            with open(os.path.join(tac_path,f'{self.sub_id}-{region_name}-tac.json'),
                      'w',encoding='ascii') as out_file:
                json.dump(obj=region_json,fp=out_file,indent=4)
