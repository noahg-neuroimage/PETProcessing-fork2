import numpy as np


def make_early_mean_image_from_4d_pet(pet_4d_data: np.ndarray,
                                      start_frame: int = 3,
                                      end_frame: int = 7) -> np.ndarray:
    """
    Calculate the mean image across a specified range of frames in a 4D PET data array.

    Args:
        pet_4d_data (np.ndarray): A 4D numpy array representing PET data over time, with shape (time_frames, height, width, depth).
        start_frame (int): The starting frame index (inclusive). Defaults to 3.
        end_frame (int): The ending frame index (inclusive). Defaults to 7.

    Returns:
        np.ndarray: A 3D numpy array representing the mean image across the specified frames.

    Raises:
        ValueError: If the start or end frame indices are out of the array's bounds.
    """
    if start_frame < 0 or end_frame >= pet_4d_data.shape[0]:
        raise ValueError("Frame indices are out of bounds.")

    early_mean = np.mean(pet_4d_data[start_frame:end_frame + 1], axis=0)

    return early_mean


def crop_by_input_function_mask(early_mean_data: np.ndarray,
                                carotid_mask_data: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an early mean image, effectively cropping or masking the image.

    Args:
        early_mean_data (np.ndarray): The early mean image data as a 3D numpy array.
        carotid_mask_data (np.ndarray): A binary mask data as a 3D numpy array of the same shape as early_mean_data.

    Returns:
        np.ndarray: The masked image data, retaining only the parts of early_mean_data where carotid_mask_data is not zero.

    Raises:
        ValueError: If early_mean_data and carotid_mask_data do not have the same shape.
    """
    if early_mean_data.shape != carotid_mask_data.shape:
        raise ValueError("array1 and array2 must have the same shape.")

    masked_data = early_mean_data * carotid_mask_data
    return masked_data


def make_threshold_binary_mask(masked_data: np.ndarray,
                               threshold: float) -> np.ndarray:
    """
    Create a binary mask based on a threshold, setting values below the threshold to 0 and values at or above to 1.

    Args:
        masked_data (np.ndarray): The image data to threshold as a numpy array.
        threshold (float): The threshold value.

    Returns:
        np.ndarray: A binary mask of the same shape as masked_data.
    """
    threshold_binary_data = np.where(masked_data < threshold, 0, 1)
    return threshold_binary_data


def apply_threshold_binary_mask_to_4d_pet(pet_4d_data: np.ndarray,
                                          threshold_binary_data: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to each frame of a 4D PET data array.

    Args:
        pet_4d_data (np.ndarray): The 4D PET data array with shape (time_frames, height, width, depth).
        threshold_binary_data (np.ndarray): A binary mask with the same spatial dimensions as the frames in pet_4d_data.

    Returns:
        np.ndarray: The masked 4D PET data.
    """
    masked_4d_pet = pet_4d_data * threshold_binary_data
    return masked_4d_pet


def average_masked_4d_pet_into_tac(masked_4d_pet_data: np.ndarray) -> np.ndarray:
    """
    Average the values of each 3D frame in a 4D masked PET data array into a 1D numpy array of the averages.

    Args:
        masked_4d_pet_data (np.ndarray): The 4D masked PET data array with shape (time_frames, height, width, depth).

    Returns:
        np.ndarray: A 1D numpy array where each element represents the average of the corresponding 3D frame in the masked 4D PET data.
    """
    frame_averages = np.mean(masked_4d_pet_data, axis=(1, 2, 3))
    return frame_averages
