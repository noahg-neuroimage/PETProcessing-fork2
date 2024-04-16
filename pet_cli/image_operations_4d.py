"""
The 'image_operations_4d' module provides several functions used to do preprocessing
on 4D PET imaging series. These functions typically take one or more paths to imaging
data in NIfTI format, and save modified data to a NIfTI file, and may return the
modified imaging array as output.

Class :class:`ImageOps4D` is also included in this module, and provides specific
implementations of the functions presented herein.

TODOs:
    * (weighted_series_sum) Refactor the DecayFactor key extraction into its own function
    * (weighted_series_sum) Refactor verbose reporting into the class as it is unrelated to
      computation
    * (write_tacs) Shift to accepting color-key dictionaries rather than a file path.
    * (extract_tac_from_4dnifty_using_mask) Write the number of voxels in the mask, or the
      volume of the mask. This is necessary for certain analyses with the resulting tacs,
      such as finding the average uptake encompassing two regions.

"""
import os
import re
import ants
import nibabel
from nibabel import processing
import numpy as np
from . import image_io
from . import math_lib


def weighted_series_sum(input_image_4d_path: str,
                        out_image_path: str,
                        half_life: float,
                        verbose: bool) -> np.ndarray:
    r"""
    Sum a 4D image series weighted based on time and re-corrected for decay correction.

    First, a scaled image is produced by multiplying each frame by its length in seconds,
    and dividing by the decay correction applied:

    .. math::
    
        f_i'=f_i\times \frac{t_i}{d_i}

    Where :math:`f_i,t_i,d_i` are the i-th frame, frame duration, and decay correction factor of
    the PET series. This scaled image is summed over the time axis. Then, to get the output, we
    multiply by a factor called `total decay` and divide by the full length of the image:

    .. math::

        d_{S} = \frac{\lambda*t_{S}}{(1-\exp(-\lambda*t_{S}))(\exp(\lambda*t_{0}))}

    .. math::
    
        S(f) = \sum(f_i') * d_{S} / t_{S}

    where :math:`\lambda=\log(2)/T_{1/2}` is the decay constant of the radio isotope,
    :math:`t_0` is the start time of the first frame in the PET series, the subscript :math:`S`
    indicates the total quantity computed over all frames, and :math:`S(f)` is the final weighted
    sum image.

    
    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image on which the weighted sum is calculated. Assume a metadata
            file exists with the same path and file name, but with extension .json,
            and follows BIDS standard.
        out_image_path (str): Path to a .nii or .nii.gz file to which the weighted
            sum is written.
        half_life (float): Half life of the PET radioisotope in seconds.
        verbose (bool): Set to `True` to output processing information.

    Returns:
        summed_image (np.ndarray): 3D image array, in the same space as the input,
            with the weighted sum calculation applied.

    Raises:
        ValueError: If `half_life` is zero or negative.
    """
    if half_life <= 0:
        raise ValueError('(ImageOps4d): Radioisotope half life is zero or negative.')
    pet_meta = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    pet_image = nibabel.load(input_image_4d_path)
    pet_series = pet_image.get_fdata()
    image_frame_start = pet_meta['FrameTimesStart']
    image_frame_duration = pet_meta['FrameDuration']

    if 'DecayCorrectionFactor' in pet_meta.keys():
        image_decay_correction = pet_meta['DecayCorrectionFactor']
    elif 'DecayFactor' in pet_meta.keys():
        image_decay_correction = pet_meta['DecayFactor']
    else:
        raise ValueError("Neither 'DecayCoorectionFactor' nor 'DecayFactor' exist in meta-data file")

    if 'TracerRadionuclide' in pet_meta.keys():
        tracer_isotope = pet_meta['TracerRadionuclide']
        if verbose:
            print(f"(ImageOps4d): Radio isotope is {tracer_isotope} "
                f"with half life {half_life} s")

    image_weighted_sum = math_lib.weighted_sum_computation(image_frame_duration=image_frame_duration,
                                                           half_life=half_life,
                                                           pet_series=pet_series,
                                                           image_frame_start=image_frame_start,
                                                           image_decay_correction=image_decay_correction)

    pet_sum_image = nibabel.nifti1.Nifti1Image(dataobj=image_weighted_sum,
                                               affine=pet_image.affine,
                                               header=pet_image.header)
    nibabel.save(pet_sum_image, out_image_path)
    if verbose:
        print(f"(ImageOps4d): weighted sum image saved to {out_image_path}")
    return pet_sum_image


def motion_correction(input_image_4d_path: str,
                      reference_image_path: str,
                      out_image_path: str,
                      verbose: bool,
                      type_of_transform: str='DenseRigid',
                      **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion correction module
    from Advanced Normalisation Tools (ANTs) with default inputs. 

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D reference
            image in the same space as the input PET image. Can be a weighted series sum,
            first or last frame, an average over a subset of frames, or another option depending
            on the needs of the data.
        out_image_path (str): Path to a .nii or .nii.gz file to which the motion corrected PET
            series is written.
        type_of_transform (str): Type of transform to perform on the PET image, must be one of antspy's
            transformation types, i.e. 'DenseRigid' or 'Translation'. Any transformation type that uses
            >6 degrees of freedom is not recommended, use with caution. See :py:func:`ants.registration`.
        verbose (bool): Set to `True` to output processing information.
        kwargs (keyword arguments): Additional arguments passed to `ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy array.
        pet_moco_params (list[str]): List of ANTS registration files applied to each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure corresponding 
        to each frame transform.
    """
    pet_nibabel = nibabel.load(input_image_4d_path)
    pet_ref_image = nibabel.load(reference_image_path)
    pet_ants = ants.from_nibabel(pet_nibabel)
    pet_sum_image_ants = ants.from_nibabel(pet_ref_image)

    pet_moco_ants_dict = ants.motion_correction(image=pet_ants,
                                                fixed=pet_sum_image_ants,
                                                type_of_transform=type_of_transform,
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)

    copy_meta_path = re.sub('.nii.gz|.nii', '.json', out_image_path)
    meta_data_dict = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(
        input_image_4d_path)
    image_io.write_dict_to_json(meta_data_dict=meta_data_dict, out_path=copy_meta_path)

    nibabel.save(pet_moco_nibabel, out_image_path)
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd


def register_pet(input_calc_image_path: str,
                 input_reg_image_path: str,
                 reference_image_path: str,
                 out_image_path: str,
                 verbose: bool,
                 type_of_transform: str='DenseRigid',
                 **kwargs):
    """
    Computes and runs rigid registration of 4D PET image series to 3D anatomical image, typically
    a T1 MRI. Runs rigid registration module from Advanced Normalisation Tools (ANTs) with  default
    inputs. Will upsample PET image to the resolution of anatomical imaging.

    Args:
        input_calc_image_path (str): Path to a .nii or .nii.gz file containing a 3D reference
            image in the same space as the input PET image, to be used to compute the rigid 
            registration to anatomical space. Can be a weighted series sum, first or last frame,
            an average over a subset of frames, or another option depending on the needs of the 
            data.
        input_reg_image_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be registered to anatomical space.
        reference_image_path (str): Path to a .nii or .nii.gz file containing a 3D
            anatomical image to which PET image is registered.
        type_of_transform (str): Type of transform to perform on the PET image, must be one of antspy's
            transformation types, i.e. 'DenseRigid' or 'Translation'. Any transformation type that uses
            >6 degrees of freedom is not recommended, use with caution. See :py:func:`ants.registration`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the registered PET series
            is written.
        verbose (bool): Set to `True` to output processing information.
        kwargs (keyword arguments): Additional arguments passed to :py:func:`ants.registration`.
    """
    pet_sum_image = ants.image_read(input_calc_image_path)
    mri_image = ants.image_read(reference_image_path)
    pet_moco = ants.image_read(input_reg_image_path)
    xfm_output = ants.registration(moving=pet_sum_image,
                                   fixed=mri_image,
                                   type_of_transform=type_of_transform,
                                   write_composite_transform=True,
                                   **kwargs)
    if verbose:
        print(f'Registration computed transforming image {input_calc_image_path} to '
              f'{reference_image_path} space')

    xfm_apply = ants.apply_transforms(moving=pet_moco,
                                      fixed=mri_image,
                                      transformlist=xfm_output['fwdtransforms'],
                                      imagetype=3)
    if verbose:
        print(f'Registration applied to {input_reg_image_path}')

    ants.image_write(xfm_apply, out_image_path)
    if verbose:
        print(f'Transformed image saved to {out_image_path}')


def resample_segmentation(input_image_4d_path: str,
                          segmentation_image_path: str,
                          out_seg_path: str,
                          verbose: bool):
    """
    Resamples a segmentation image to the resolution of a 4D PET series image. Takes the affine 
    information stored in the PET image, and the shape of the image frame data, as well as the 
    segmentation image, and applies NiBabel's `resample_from_to` to resample the segmentation to
    the resolution of the PET image. This is used for extracting TACs from PET imaging where the 
    PET and ROI data are registered to the same space, but have different resolutions.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space, to which the segmentation file is resampled.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions.
        out_seg_path (str): Path to a .nii or .nii.gz file to which the resampled segmentation
            image is written.
        verbose (bool): Set to `True` to output processing information.
    """
    pet_image = nibabel.load(input_image_4d_path)
    seg_image = nibabel.load(segmentation_image_path)
    pet_series = pet_image.get_fdata()
    image_first_frame = pet_series[:, :, :, 0]
    seg_resampled = processing.resample_from_to(from_img=seg_image,
                                                to_vox_map=(image_first_frame.shape, pet_image.affine),
                                                order=0)
    nibabel.save(seg_resampled, out_seg_path)
    if verbose:
        print(f'Resampled segmentation saved to {out_seg_path}')


def extract_tac_from_4dnifty_using_mask(input_image_4d_path: str,
                                        segmentation_image_path: str,
                                        region: int,
                                        verbose: bool) -> np.ndarray:
    """
    Creates a time-activity curve (TAC) by computing the average value within a region, for each 
    frame in a 4D PET image series. Takes as input a PET image, which has been registered to
    anatomical space, a segmentation image, with the same sampling as the PET, and a list of values
    corresponding to regions in the segmentation image that are used to compute the average
    regional values. Currently, only the mean over a single region value is implemented.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image, registered to anatomical space.
        segmentation_image_path (str): Path to a .nii or .nii.gz file containing a 3D segmentation
            image, where integer indices label specific regions. Must have same sampling as PET
            input.
        region (int): Value in the segmentation image corresponding to a region
            over which the TAC is computed.
        verbose (bool): Set to `True` to output processing information.

    Returns:
        tac_out (np.ndarray): Mean of PET image within regions for each frame in 4D PET series.

    Raises:
        ValueError: If the segmentation image and PET image have different
            sampling.
    """

    pet_image_4d = nibabel.load(input_image_4d_path).get_fdata()
    num_frames = pet_image_4d.shape[3]
    seg_image = nibabel.load(segmentation_image_path).get_fdata()

    if seg_image.shape!=pet_image_4d.shape[:3]:
        raise ValueError('Mis-match in image shape of segmentation image '
                         f'({seg_image.shape}) and PET image '
                         f'({pet_image_4d.shape[:3]}). Consider resampling '
                         'segmentation to PET or vice versa.')

    tac_out = np.zeros(num_frames, float)
    if verbose:
        print(f'Running TAC for region index {region}')
    masked_voxels = seg_image == region
    masked_image = pet_image_4d[masked_voxels].reshape((-1, num_frames))
    tac_out = np.mean(masked_image, axis=0)
    return tac_out


def write_tacs(input_image_4d_path: str,
               label_map_path: str,
               segmentation_image_path: str,
               out_tac_dir: str,
               verbose: bool,
               time_frame_keyword: str = 'FrameReferenceTime'):
    """
    Function to write Tissue Activity Curves for each region, given a segmentation,
    4D PET image, and color table. Computes the average of the PET image within each
    region. Writes a JSON for each region with region name, frame start time, and mean 
    value within region.
    """

    if time_frame_keyword not in ['FrameReferenceTime', 'FrameTimesStart']:
        raise ValueError("'time_frame_keyword' must be one of "
                         "'FrameReferenceTime' or 'FrameTimesStart'")

    pet_meta = image_io.ImageIO.load_metadata_for_nifty_with_same_filename(input_image_4d_path)
    color_table = image_io.ImageIO.read_color_table_json(label_map_file=label_map_path)
    regions_list = color_table['data']

    tac_extraction_func = extract_tac_from_4dnifty_using_mask

    for region_index, region_name in regions_list:
        extracted_tac = tac_extraction_func(input_image_4d_path=input_image_4d_path,
                                            segmentation_image_path=segmentation_image_path,
                                            region=int(region_index),
                                            verbose=verbose)
        region_tac_file = np.array([pet_meta[time_frame_keyword],extracted_tac]).T
        header_text = f'{time_frame_keyword}\t{region_name}_mean_activity'
        out_tac_path = os.path.join(out_tac_dir, f'tac-{region_name}.tsv')
        np.savetxt(out_tac_path,region_tac_file,delimiter='\t',header=header_text,comments='')


class ImageOps4d():
    """
    :class:`ImageOps4D` to provide basic implementations of the preprocessing functions in module
    `image_operations_4d`.

    Preprocessing can be run on individual subjects by specifying information such as the subject
    id, output path, paths to PET, anatomical, and segmentation images, etc. Then individual
    methods can be run in succession.

    Key methods include:
        - :meth:`run_weighted_series_sum`: Runs :meth:`weighted_series_sum` on input data.
        - :meth:`run_motion_correction`: Runs :meth:`motion_correction` on input data, with the
          output of :func:`weighted_series_sum` as reference.
        - :meth:`run_register_pet`: Runs :meth:`register_pet` on motion corrected PET with the
          output of :func:`weighted_series_sum` used to compute registration.
        - :meth:`run_mask_image_to_vals`: Runs :meth:`extract_tac_from_4dnifty_using_mask`, to be
          used with :meth:`run_write_tacs`.
        - :meth:`run_write_tacs`: Runs :meth:`write_tacs` on preprocessed PET data to produce
          regional TACs.
    
    Attributes:
        sub_id (str): The subject ID, used for naming output files.
        out_path (str): Path to an output directory, to which processed files are saved.
        image_paths (dict): A dictionary with designated keys for different types of images.
            Designated keys include 'pet' for input PET data, 'mri' for anatomical data, 'seg' for
            segmentation in anatomical space, 'pet_sum_image' for the output to
            :meth:`weighted_series_sum`, 'pet_moco' for motion corrected PET image, pet_moco_reg
            for motion corrected and registered PET image, and 'seg_resampled' for a segmentation
            resampled onto PET resolution.
        half_life (float): Half-life of the radioisotope used in PET study in seconds.
        label_map_path (str): Path to a color table .json file used to match region names to
            region indices.
        verbose (bool): Set to `True` to output processing information.
    
    See Also:
        :class:`ImageIO`
    
    """
    def __init__(self,
                 output_directory: str,
                 output_filename_prefix: str) -> None:
        """
        Init
        """
        self.output_directory = os.path.abspath(output_directory)
        self.output_filename_prefix = output_filename_prefix
        self.preproc_props = self._init_preproc_props()

    def _init_preproc_props(self) -> dict:
        """
        Initializes preproc properties dictionary.
        """
        props = {'FilePathPET': None,
                 'FilePathMocoInp': None,
                 'FilePathRegInp': None,
                 'FilePathAnat': None,
                 'FilePathPETRef': None,
                 'FilePathTACInput': None,
                 'FileWeightedPET': None,
                 'FilePathSeg': None,
                 'FilePathLabelMap': None,
                 'MethodName': None,
                 'MocoPars': None,
                 'RegPars': None,
                 'HalfLife': None,
                 'RegionExtract': None,
                 'TimeFrameKeyword': None,
                 'Verbose': False}
        return props
    

    def update_props(self,new_props: dict) -> dict:
        """
        Update the processing properties with items from a new dictionary.

        Returns the updated `props` dictionary.
        """
        props = self.props
        updated_props = props.copy()
        keys_to_update = [*new_props]
        for key in keys_to_update:
            updated_props[key] = new_props[key]
        self.props = updated_props
        return updated_props


    def _check_method_props_exist(self,
                                 method_name: str) -> None:
        """
        Check if all necessary properties exist in the `props` dictionary to
        run the given method.
        """
        props = self.props
        existing_keys = [*props]

        if method_name=='weighted_series_sum':
            required_keys = ['FilePathPet','HalfLife','Verbose']
        elif method_name=='motion_correction':
            required_keys = ['FilePathMocoInp','FilePathPETRef','Verbose']
        elif method_name=='register_pet':
            required_keys = ['FilePathPETRef','FilePathRegInp','FilePathAnat','Verbose']
        elif method_name=='resample_segmentation':
            required_keys = ['FilePathTACInput','FilePathSeg','Verbose']
        elif method_name=='extract_tac_from_4dnifty_using_mask':
            required_keys = ['FilePathTACInput','FilePathSeg','RegionExtract','Verbose']
        elif method_name=='write_tacs':
            required_keys = ['FilePathTACInput','FilePathLabelMap','FilePathSeg','Verbose','TimeFrameKeyword']
        else:
            raise ValueError("Invalid method_name! Must be either"
                             "'weighted_series_sum', 'motion_correction', "
                             "'register_pet', 'resample_segmentation', "
                             "'extract_tac_from_4dnifty_using_mask', or "
                             f"'write_tacs'. Got {method_name}")
        for key in required_keys:
            if key not in existing_keys:
                raise ValueError(f"Preprocessing method requires property"
                                 f" {key}, however {key} was not found in "
                                 "processing properties. Existing properties "
                                 f"are: {existing_keys}, while needed keys to "
                                 f"run {method_name} are: {required_keys}.")
    

    def run_preproc(self,
                    method_name: str):
        """
        Run a specific preprocessing step
        """
        props = self.props
        self._check_method_props_exist(method_name=method_name)
        if method_name=='weighted_series_sum':
            output_file_name = f'{self.output_filename_prefix}_wss.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            weighted_series_sum(input_image_4d_path=props.FilePathPET,
                                out_image_path=outfile,
                                half_life=props.HalfLife,
                                verbose=props.Verbose)
        elif method_name=='motion_correction':
            output_file_name = f'{self.output_filename_prefix}_moco.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            motion_correction(input_image_4d_path=props.FilePathMocoInp,
                              reference_image_path=props.FilePathPETRef,
                              out_image_path=outfile,
                              verbose=props.Verbose,
                              kwargs=props.MocoPars)
        elif method_name=='register_pet':
            output_file_name = f'{self.output_filename_prefix}_reg.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            register_pet(input_calc_image_path=props.FilePathPETRef,
                         input_reg_image_path=props.FilePathRegInp,
                         reference_image_path=props.FilePathAnat,
                         out_image_path=outfile,
                         verbose=props.Verbose,
                         kwargs=props.RegPars)
        elif method_name=='resample_segmentation':
            output_file_name = f'{self.output_filename_prefix}_seg-res.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            resample_segmentation(input_image_4d_path=props.FilePathTACInput,
                                  segmentation_image_path=props.FilePathSeg,
                                  out_seg_path=outfile,
                                  verbose=props.Verbose)
            self.update_props({'FilePathSeg': outfile})
        elif method_name=='extract_tac_from_4dnifty_using_mask':
            return extract_tac_from_4dnifty_using_mask(input_image_4d_path=props.FilePathTACInput,
                                                segmentation_image_path=props.FilePathSeg,
                                                region=props.RegionExtract,
                                                verbose=props.Verbose)
        elif method_name=='write_tacs':
            outdir = os.path.join(self.output_directory,'tacs')
            write_tacs(input_image_4d_path=props.FilePathTACInput,
                       label_map_path=props.FilePathLabelMap,
                       segmentation_image_path=props.FilePathSeg,
                       out_tac_dir=outdir,
                       verbose=props.Verbose,
                       time_frame_keyword=props.TimeFrameKeyword)
        else:
            raise ValueError("Invalid method_name! Must be either"
                             "'weighted_series_sum', 'motion_correction', "
                             "'register_pet', 'resample_segmentation', "
                             "'extract_tac_from_4dnifty_using_mask', or "
                             f"'write_tacs'. Got {method_name}")
        return None
