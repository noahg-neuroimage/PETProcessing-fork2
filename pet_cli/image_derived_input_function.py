import numpy as np
import pandas as pd


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

# Below is longer methods
#-----------------------------------------------------------------

def get_frame_time_midpoints(frame_start_times: np.ndarray,
                             frame_duration_times: np.ndarray) -> np.ndarray:
    """
    Calculate the midpoints of frame times given the start times and durations.

    Args:
        frame_start_times (np.ndarray): An array of frame start times.
        frame_duration_times (np.ndarray): An array of durations corresponding to each start time.

    Returns:
        np.ndarray: An array of midpoint times calculated as the start time plus half of the duration for each frame.

    """
    frame_midpoint_times = frame_start_times + (frame_duration_times / 2)
    return frame_midpoint_times


def load_fslmeants_to_numpy(fslmeants_filepath: str) -> np.ndarray:
    """
    Load the fslmeants output file from a CSV and convert it to a NumPy array, removing the first three rows
    which are assumed to be coordinate data.

    Args:
        fslmeants_filepath (str): The file path to the CSV containing the fslmeants output.

    Returns:
        np.ndarray: A 2D NumPy array where each row corresponds to a time point and each column to a different ROI.

    """
    data = pd.read_csv(fslmeants_filepath, header=None)
    if data.shape[1] > data.shape[0]:
        data = data.iloc[3:]

    return data.values


def get_idif_from_fslmeants_file_of_4d_pet_necktangle(fslmeants_vals: np.ndarray,
                 percentile: float,
                 frame_midpoint_times: np.ndarray) -> np.ndarray:
    """
    Process the fslmeants data to identify the bolus frame based on the highest mean value within the first ten frames,
    determine 'carotid' voxels based on a percentile cut-off around the bolus frame, and calculate a specified percentile
    for these voxels across all frames.

    Args:
        fslmeants_vals (np.ndarray): The NumPy array of fslmeants values, where each row is a time point and each column is an ROI.
        percentile (float): The percentile threshold used to identify 'carotid' voxels.
        frame_midpoint_times (np.ndarray): An array of frame times to be associated with each percentile value calculated.

    Returns:
        np.ndarray: A 2D array with each row containing a frame time and the corresponding percentile value for the 'carotid' voxels.

    Raises:
        ValueError: If the length of frame_midpoint_times is less than the number of percentile values calculated.

    """
    frame_averages = np.mean(fslmeants_vals, axis=1)
    bolus_frame = np.argmax(frame_averages[:10])
    bolus_window_vals = fslmeants_vals[bolus_frame - 1:bolus_frame + 2, :]
    carotid_cut = 90
    carotid_inds = np.where(np.mean(bolus_window_vals, axis=0) > np.percentile(np.mean(bolus_window_vals, axis=0), carotid_cut))[0]
    percentile_vals_z = np.percentile(fslmeants_vals[:, carotid_inds], percentile, axis=1)

    if len(frame_midpoint_times) < len(percentile_vals_z):
        raise ValueError("Frame times array is shorter than the number of percentile values calculated.")

    output_data = np.column_stack((frame_midpoint_times[:len(percentile_vals_z)], percentile_vals_z))

    return output_data
