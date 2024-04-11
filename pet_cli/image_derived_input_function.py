import numpy as np


def make_early_mean_image_from_4d_pet(pet_4d_data: np.ndarray,
                                      start_frame: int = 3,
                                      end_frame: int = 7) -> np.ndarray:
    """
    Calculate the mean image across a specified range of frames in a 4D numpy array.

    Parameters:
    - data: A 4D numpy array, with the first dimension being time/frames.
    - start_frame: The starting frame index (inclusive).
    - end_frame: The ending frame index (inclusive).

    Returns:
    - early_mean: A 3D numpy array representing the mean image across the specified frames.
    """
    # Ensure the frame indices are within the bounds of the data's first dimension
    if start_frame < 0 or end_frame >= pet_4d_data.shape[0]:
        raise ValueError("Frame indices are out of bounds.")

    # Calculate the mean across the specified frames
    early_mean = np.mean(pet_4d_data[start_frame:end_frame + 1], axis=0)

    return early_mean


def crop_by_input_function_mask(early_mean_data: np.ndarray,
                                carotid_mask_data: np.ndarray) -> np.ndarray:
    if early_mean_data.shape != carotid_mask_data.shape:
        raise ValueError("array1 and array2 must have the same shape.")

    masked_data = early_mean_data * carotid_mask_data
    return masked_data


def make_threshold_binary_mask(masked_data: np.ndarray,
                               threshold: float) -> np.ndarray:
    threshold_binary_data = np.where(masked_data < threshold, 0, 1)
    return threshold_binary_data


def apply_threshold_binary_mask_to_4d_pet(pet_4d_data: np.ndarray,
                                          threshold_binary_data: np.ndarray) -> np.ndarray:
    masked_4d_pet = pet_4d_data * threshold_binary_data
    return masked_4d_pet


def average_masked_4d_pet_into_tac(masked_4d_pet_data: np.ndarray) -> np.ndarray:
    frame_averages = np.mean(masked_4d_pet_data, axis=(1, 2, 3))
    return frame_averages
