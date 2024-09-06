"""
Provides methods to motion correct 4D PET data. Includes method
:meth:`determine_motion_target`, which produces a flexible target based on the
4D input data to optimize contrast when computing motion correction or
registration.
"""
import os
import tempfile
from typing import Union
import ants
import nibabel
import numpy as np
from ..utils import image_io
from ..preproc.image_operations_4d import weighted_series_sum


def determine_motion_target(motion_target_option: Union[str,tuple,list],
                            input_image_4d_path: str=None,
                            half_life: float=None) -> str:
    """
    Produce a motion target given the ``motion_target_option`` from a method
    running registrations on PET, i.e. :meth:`motion_correction` or
    :meth:`register_pet`.

    The motion target option can be a string or a tuple. If it is a string,
    then if this string is a file, use the file as the motion target.

    If it is the option ``weighted_series_sum``, then run
    :meth:`weighted_series_sum` and return the output path.

    If it is a tuple, run a weighted sum on the PET series on a range of 
    frames. The elements of the tuple are treated as times in seconds, counted
    from the time of the first frame, i.e. (0,300) would average all frames 
    from the first to the frame 300 seconds later. If the two elements are the
    same, returns the one frame closest to the entered time.

    Args:
        motion_target_option (str | tuple | list): Determines how the method behaves,
            according to the above description. Can be a file, a method
            ('weighted_series_sum'), or a tuple range e.g. (0,600).
        input_image_4d_path (str): Path to the PET image. This is intended to
            be supplied by the parent method employing this function. Default
            value None.
        half_life (float): Half life of the radiotracer used in the image
            located at ``input_image_4d_path``. Only used if a calculation is
            performed.
    
    Returns:
        out_image_file (str): File to use as a target to compute
            transformations on.

    Raises:
        ValueError: If ``motion_target_option`` does not match an acceptable option.
        TypeError: If start and end time are incompatible with ``float`` type.
    """
    if isinstance(motion_target_option,str):
        if os.path.exists(motion_target_option):
            return motion_target_option

        elif motion_target_option=='weighted_series_sum':        
            out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
            weighted_series_sum(input_image_4d_path=input_image_4d_path,
                                out_image_path=out_image_file,
                                half_life=half_life,
                                verbose=False)
            return out_image_file

        else:
            raise ValueError("motion_target_option did not match a file or 'weighted_series_sum'")

    elif isinstance(motion_target_option,tuple) or isinstance(motion_target_option,list):

        start_time = motion_target_option[0]
        end_time = motion_target_option[1]

        try:
            float(start_time)
            float(end_time)
        except:
            raise TypeError('Start time and end time of calculation must be '
                            'able to be cast into float! Provided values are '
                            f"{start_time} and {end_time}.")

        out_image_file = tempfile.mkstemp(suffix='_wss.nii.gz')[1]
        weighted_series_sum(input_image_4d_path=input_image_4d_path,
                            out_image_path=out_image_file,
                            half_life=half_life,
                            verbose=False,
                            start_time=float(start_time),
                            end_time=float(end_time))

        return out_image_file

    else:
        raise ValueError('motion_target_option did not match str or tuple type.')


def motion_corr(input_image_4d_path: str,
                motion_target_option: Union[str,tuple],
                out_image_path: str,
                verbose: bool,
                type_of_transform: str='DenseRigid',
                half_life: float=None,
                **kwargs) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Correct PET image series for inter-frame motion. Runs rigid motion
    correction module from Advanced Normalisation Tools (ANTs) with default
    inputs.

    Args:
        input_image_4d_path (str): Path to a .nii or .nii.gz file containing a 4D
            PET image to be motion corrected.
        motion_target_option (str | tuple): Target image for computing
            transformation. See :meth:`determine_motion_target`.
        out_image_path (str): Path to a .nii or .nii.gz file to which the
            motion corrected PET series is written.
        verbose (bool): Set to ``True`` to output processing information.
        type_of_transform (str): Type of transform to perform on the PET image,
            must be one of antspy's transformation types, i.e. 'DenseRigid' or
            'Translation'. Any transformation type that uses >6 degrees of
            freedom is not recommended, use with caution. See 
            :py:func:`ants.registration`.
        half_life (float): Half life of the PET radioisotope in seconds. Used
            for certain settings of ``motion_target_option``.
        kwargs (keyword arguments): Additional arguments passed to
            :py:func:`ants.motion_correction`.

    Returns:
        pet_moco_np (np.ndarray): Motion corrected PET image series as a numpy
            array.
        pet_moco_params (list[str]): List of ANTS registration files applied to
            each frame.
        pet_moco_fd (list[float]): List of framewise displacement measure
            corresponding to each frame transform.
    """
    pet_ants = ants.image_read(input_image_4d_path)
    motion_target_image_path = determine_motion_target(motion_target_option=motion_target_option,
                                                       input_image_4d_path=input_image_4d_path,
                                                       half_life=half_life)

    motion_target_image = ants.image_read(motion_target_image_path)
    pet_moco_ants_dict = ants.motion_correction(image=pet_ants,
                                                fixed=motion_target_image,
                                                type_of_transform=type_of_transform,
                                                **kwargs)
    if verbose:
        print('(ImageOps4D): motion correction finished.')

    pet_moco_ants = pet_moco_ants_dict['motion_corrected']
    pet_moco_params = pet_moco_ants_dict['motion_parameters']
    pet_moco_fd = pet_moco_ants_dict['FD']
    pet_moco_np = pet_moco_ants.numpy()
    pet_moco_nibabel = ants.to_nibabel(pet_moco_ants)

    image_io.safe_copy_meta(input_image_path=input_image_4d_path,out_image_path=out_image_path)

    nibabel.save(pet_moco_nibabel, out_image_path)
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")
    return pet_moco_np, pet_moco_params, pet_moco_fd


def motion_corr_to_t1(input_image_4d_path: str,
                      t1_image_path: str,
                      motion_target_option: Union[str, tuple],
                      out_image_path: str,
                      verbose: bool,
                      frames_list: list = None,
                      type_of_transform: str = 'AffineFast',
                      transform_metric: str = "mattes",
                      half_life: float = None):
    r"""
    Perform motion correction of a 4D PET image to a T1 anatomical image.

    This function corrects motion in a 4D PET image by registering it to a T1 anatomical
    image. The method uses a two-step process: first registering an intermediate motion
    target to the T1 image (either the time-averaged image or a weighted-series-sum), and
    then using the calculated transform to correct motion in individual frames of the PET series.
    The motion-target-option is registered to the T1 anatomical image. For each frame in the 4D-PET
    we want to motion-correct, if the frame mean voxel value is less than the entire time-series
    mean voxel value, the frame is simply transformed to the motion-target in T1-space, else the frame
    is registered to the motion-target in T1-space.

    Args:
        input_image_4d_path (str): Path to the 4D PET image to be corrected.
        t1_image_path (str): Path to the 3D T1 anatomical image.
        motion_target_option (str | tuple): Option for selecting the motion target image.
            Can be a path to a file or a tuple range. If None, the average of the PET timeseries
            is used.
        out_image_path (str): Path to save the motion-corrected 4D image.
        verbose (bool): Set to True to print verbose output during processing.
        frames_list (list, optional): List of frame indices to correct. If None, all frames
            are corrected.
        type_of_transform (str): Type of transformation used in registration. Default is 'AffineFast'.
        transform_metric (str): Metric for transformation optimization. Default is 'mattes'.
        half_life (float, optional): Half-life of the PET radioisotope. Used if a calculation
            is required for the motion target.

    Returns:
        None

    Raises:
        AssertionError: If maximum frame index in `frames_list` exceeds the number of frames in the PET image.

    Example:
        
        .. code-block:: python
        
        
        motion_corr_to_t1(input_image_4d_path='pet_timeseries.nii.gz',
                          t1_image_path='t1_image.nii.gz',
                          motion_target_option='average',
                          out_image_path='pet_corrected.nii.gz',
                          verbose=True)
                        
    """
    
    input_image = ants.image_read(input_image_4d_path)
    t1_image = ants.image_read(t1_image_path)
    
    if motion_target_option is None:
        motion_target = input_image.get_average_of_timeseries()
    else:
        motion_target_path = determine_motion_target(motion_target_option=motion_target_option,
                                                     input_image_4d_path=input_image_4d_path,
                                                     half_life=half_life)
        motion_target = ants.image_read(motion_target_path)
    
    motion_target_to_mpr_reg = ants.registration(fixed=t1_image,
                                                 moving=motion_target,
                                                 type_of_transform=type_of_transform,
                                                 aff_metric=transform_metric, )
    
    motion_target_in_t1 = motion_target_to_mpr_reg['warpedmovout']
    motion_transform_matrix = motion_target_to_mpr_reg['fwdtransforms']
    
    if frames_list is None:
        frames_to_correct = np.arange(input_image.shape[-1], dtype=int)
    else:
        assert max(frames_list) < input_image.shape[-1]
        frames_to_correct = frames_list
    
    total_mean_voxel_value = t1_image.mean()
    
    out_image = []
    input_image_list = input_image.ndimage_to_list()
    
    if verbose:
        print("(Info): On frame:", end=' ')
    
    for frame_id in frames_to_correct:
        print(f"{frame_id:>02}", end=' ')
        this_frame = input_image_list[frame_id]
        frame_mean_val = this_frame.mean()
        if frame_mean_val < total_mean_voxel_value:
            tmp_transform = ants.apply_transforms(fixed=motion_target_in_t1,
                                                  moving=this_frame,
                                                  transforms=motion_transform_matrix, )
            out_image.append(tmp_transform)
        else:
            tmp_reg = ants.registration(fixed=motion_target_in_t1,
                                        moving=this_frame,
                                        type_of_transform=type_of_transform,
                                        aff_metric=transform_metric, )
            out_image.append(tmp_reg['warpedmovout'])
    if verbose:
        print("... done!\n")
    tmp_image = _gen_nd_image_based_on_image_list(out_image)
    out_image = ants.list_to_ndimage(tmp_image, out_image)
    out_image = ants.to_nibabel(out_image)
    
    nibabel.save(out_image, out_image_path)
    
    if verbose:
        print(f"(ImageOps4d): motion corrected image saved to {out_image_path}")


def _gen_nd_image_based_on_image_list(image_list: list[ants.core.ants_image.ANTsImage]):
    r"""
    Generate a 4D ANTsImage based on a list of 3D ANTsImages.

    This function takes a list of 3D ANTsImages and constructs a new 4D ANTsImage,
    where the additional dimension represents the number of frames (3D images) in the list.
    The 4D image retains the spacing, origin, direction, and shape properties of the 3D images,
    with appropriate modifications for the additional dimension.

    Args:
        image_list (list[ants.core.ants_image.ANTsImage]):
            List of 3D ANTsImage objects to be combined into a 4D image.
            The list must contain at least one image, and all images must have the same
            dimensions and properties.

    Returns:
        ants.core.ants_image.ANTsImage:
            A 4D ANTsImage constructed from the input list of 3D images. The additional
            dimension corresponds to the number of frames (length of the image list).

    Raises:
        AssertionError: If the `image_list` is empty or if the images in the list are not 3D.

    See Also
        * :func:`~petpal.preproc.motion_corr.motion_corr_to_t1`

    Example:
        
        .. code-block:: python
        
        
            import ants
            image1 = ants.image_read('frame1.nii.gz')
            image2 = ants.image_read('frame2.nii.gz')
            image_list = [image1, image2]
            result = _gen_nd_image_based_on_image_list(image_list)
            print(result.dimension)  # 4
            image4d = ants.list_to_ndimage(result, image_list)
        
    """
    assert len(image_list) > 0
    assert image_list[0].dimension == 3
    
    num_frames = len(image_list)
    spacing_3d = image_list[0].spacing
    origin_3d = image_list[0].origin
    shape_3d = image_list[0].shape
    direction_3d = image_list[0].direction
    
    direction_4d = np.eye(4)
    direction_4d[:3, :3] = direction_3d
    spacing_4d = (*spacing_3d, 1.0)
    origin_4d = (*origin_3d, 0.0)
    shape_4d = (*shape_3d, num_frames)
    
    tmp_image = ants.make_image(imagesize=shape_4d, spacing=spacing_4d, origin=origin_4d, direction=direction_4d)
    return tmp_image
