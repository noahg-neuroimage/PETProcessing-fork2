import numpy as np
import nibabel
import pathlib


def extract_from_nii_as_numpy(file_path: str, verbose: bool) -> np.ndarray:
    """
    Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array
    :param file_path: The full path to the file.
    :return: The data contained in the .nii or .nii.gz file as a numpy array
    """
    image_data = nibabel.load(filename=file_path).get_fdata()
    
    if verbose:
        print(f"(fileIO): {file_path} has shape {image_data.shape}")
    
    return image_data
    

def compute_average_over_mask(mask: np.ndarray, image: np.ndarray) -> float:
    """
    We compute the average value of `image` over the provided `mask`. We multiply the mask
    and the image element-wise, which gives us the image pixels corresponding to the mask. We then take the standard mean
    over those values by summing and dividing by the number of values. Note that the implementation does not care about
    the dimensions of mask and image; just that they be the same.
    :param mask: 3D numpy array that contains 1s and 0s.
    :param image: 3D numpy array corresponding to an image.
    :return: Average value of the image over the mask.
    """
    #TODO: Move this assert to a higher level function if speed is greatly affected by this when iterating over many frames
    assert mask.shape == image.shape, "The mask and the image should have the same dimensions."
    
    return np.sum(mask * image) / np.sum(mask)

#TODO: Add documentation
def compute_average_over_mask_of_multiple_frames(mask: np.ndarray,
                                                 image_series: np.ndarray,
                                                 start: int,
                                                 step: int,
                                                 stop: int):
    """
    Given a mask, and multiple images, we compute the average value of the image over the mask. It is assumed that the last
    index for the images corresponds to different images or time-points. A simple and explicit iterator to stride over the
    images is included.
    :param mask:
    :param image_series:
    :param start:
    :param step:
    :param stop:
    :return:
    """
    #TODO: A smarter way to stride over images so that negative indecies can be used.
    assert start >= 0, "`start` has to be >= 0."
    assert stop <= image_series.shape[-1], "`end` has to be smaller than the number of frames in the image array."
    assert step >= 0, "`stride` has to be >= 0."
    
    frame_it = range(start, stop, step)
    num_frames = len(frame_it)
    avg_values = np.zeros(num_frames, float)
    
    for t, frmID in enumerate(frame_it):
        avg_values[t] = compute_average_over_mask(mask=mask, image=image_series[:, :, :, frmID])
    
    return avg_values

#TODO: Add documentation
def calculate_image_derived_input_function(mask_file: str,
                                           pet_file: str,
                                           start: int,
                                           step: int,
                                           stop: int):
    """
    
    :param mask_file:
    :param pet_file:
    :param start:
    :param step:
    :param stop:
    :param verbose:
    :return:
    """
    
    assert pathlib.Path(mask_file).is_file(), f"Mask file path (${mask_file}) is incorrect or does not exist."
    assert pathlib.Path(pet_file).is_file(), f"Images file path (${pet_file}) is incorrect or does not exist."
    
    mask = extract_from_nii_as_numpy(file_path=mask_file, verbose=False)
    images = extract_from_nii_as_numpy(file_path=pet_file, verbose=False)
    
    avg_vals = compute_average_over_mask_of_multiple_frames(mask=mask, image_series=images, start=start, stop=stop,
                                                            step=step)
    return avg_vals


def calculate_and_save_image_derived_input_function(mask_file: str,
                                                    pet_file: str,
                                                    out_file: str,
                                                    start: int,
                                                    step: int,
                                                    stop: int,
                                                    verbose: bool,
                                                    print_to_screen: bool):
    """
    
    :param mask_file:
    :param pet_file:
    :param out_file:
    :param start:
    :param step:
    :param stop:
    :param verbose:
    :param print_to_screen:
    :return:
    """
    
    assert pathlib.Path(mask_file).is_file(), f"Mask file path (${mask_file}) is incorrect or does not exist."
    assert pathlib.Path(pet_file).is_file(), f"Images file path (${pet_file}) is incorrect or does not exist."
    
    mask = extract_from_nii_as_numpy(file_path=mask_file, verbose=verbose)
    images = extract_from_nii_as_numpy(file_path=pet_file, verbose=verbose)
    
    avg_vals = compute_average_over_mask_of_multiple_frames(mask=mask, image_series=images, start=start, stop=stop,
                                                            step=step)
    np.savetxt(fname=out_file, X=avg_vals, delimiter=', ')
    if print_to_screen:
        print(avg_vals.shape)
        print(avg_vals)
    return avg_vals



