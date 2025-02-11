"""Decay Correction Module.

Provides functions for undo-ing decay correction and recalculating it."""

import numpy as np
import pathlib


import petpal.utils.image_io
from petpal.utils import image_io
from petpal.utils.image_io import load_metadata_for_nifti_with_same_filename


def undo_decay_correction(input_image_path: str,
                          output_image_path: str,
                          verbose: bool = False) -> np.ndarray:
    """Uses decay factors from the .json sidecar file for an image to remove decay correction for each frame."""

    image_loader = image_io.ImageIO(verbose=verbose)

    nifti_image = image_io.safe_load_4dpet_nifti(filename=input_image_path)
    json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)
    frame_info = image_io.get_frame_timing_info_for_nifti(image_path=input_image_path)
    decay_factors = frame_info['decay_factors']

    frame_num = 0
    for decay_factor in  decay_factors:
        image_data[..., frame_num] = image_data[..., frame_num] / decay_factor
        frame_num += 1

    output_image = image_loader.extract_np_to_nibabel(image_array=image_data,
                                                      header=nifti_image.header,
                                                      affine=nifti_image.affine)

    image_loader.save_nii(image=output_image,
                          out_file=output_image_path)

    json_data['DecayFactor'] = np.ones(len(decay_factors))
    output_json_path = image_io._gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
    write_dict_to_json(meta_data_dict=json_data,
                       out_path=output_json_path)

    return image_data

def decay_correct(input_image_path: str,
                  output_image_path: str,
                  verbose: bool = False) -> np.ndarray:
    """Recalculate decay_correction for nifti image based on frame reference times"""

    json_data = image_io.load_metadata_for_nifti_with_same_filename(image_path=input_image_path)


    image_loader = image_io.ImageIO(verbose=verbose)

    nifti_image = image_io.safe_load_4dpet_nifti(filename=input_image_path)
    frame_info = image_io.get_frame_timing_info_for_nifti(image_path=input_image_path)
    frame_times_start = frame_info['start']
    frame_durations = frame_info['durations']
    frame_reference_times = [start+(duration/2) for start, duration in zip(frame_times_start, frame_durations)]

    frame_num = 0
    for frame_reference_time in frame_reference_times:
        image_data[..., frame_num] = image_data[..., frame_num]
        frame_num += 1

    output_header = nifti_image.header.copy()
    output_header['descrip'] = b'decay-uncorrected'
    output_image = image_loader.extract_np_to_nibabel(image_array=image_data,
                                                      header=output_header,
                                                      affine=nifti_image.affine)

    image_loader.save_nii(image=output_image,
                          out_file=output_image_path)

    json_data['DecayFactor'] = np.ones(len(decay_factors))
    output_json_path = image_io._gen_meta_data_filepath_for_nifti(nifty_path=output_image_path)
    write_dict_to_json(meta_data_dict=json_data,
                       out_path=output_json_path)

    return image_data

