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
# -----------------------------------------------------------------

def get_frame_time_midpoints(frame_start_times: np.ndarray,
                             frame_duration_times: np.ndarray) -> np.ndarray:
    """
        Calculates the midpoint times of each frame based on the start times and duration times.

        Args:
            frame_start_times (np.ndarray): An array of frame start times.
            frame_duration_times (np.ndarray): An array of frame duration times.

        Returns:
            frame_midpoint_times (np.ndarray): An array of frame midpoint times.

        Raises:
            None.
        """
    frame_midpoint_times = (frame_start_times + (frame_duration_times / 2)).astype(int)
    return frame_midpoint_times


def load_fslmeants_to_numpy_3d(fslmeants_filepath: str) -> np.ndarray:
    """
    Loads `fslmeants <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PPIHowToRun?highlight=%28fslmeants%29>`_ (--show-all) data from a file and converts it into a 3D numpy array.

    Args:
        fslmeants_filepath (str): The filepath of the fslmeants data file.

    Returns:
        numpy_3d_array (np.ndarray): A 3D numpy array representing the fslmeants data.

    Raises:
        None.
    """
    data = np.loadtxt(fslmeants_filepath)
    x_coord_min = min(data[0].astype(int))
    y_coord_min = min(data[1].astype(int))
    z_coord_min = min(data[2].astype(int))
    x_dim = (max(data[0].astype(int)) - x_coord_min) + 1
    y_dim = (max(data[1].astype(int)) - y_coord_min) + 1
    z_dim = (max(data[2].astype(int)) - z_coord_min) + 1
    t_dim = data.shape[0] - 3
    numpy_3d_array = np.zeros((x_dim, y_dim, z_dim, t_dim), dtype=float)
    for location in range(data.shape[1]):
        x_coord = data[0, location].astype(int) - x_coord_min
        y_coord = data[1, location].astype(int) - y_coord_min
        z_coord = data[2, location].astype(int) - z_coord_min
        numpy_3d_array[x_coord, y_coord, z_coord, :] = data[3:, location]

    return numpy_3d_array


def get_idif_from_4d_pet_necktangle(necktangle_matrix: np.ndarray,
                                    percentile: float,
                                    frame_midpoint_times: np.ndarray) -> np.ndarray:
    """
    Computes the IDIF from a 4D PET necktangle matrix given a percentile for thresholding.
    This function finds the highest mean frame from the first 10 frames of the 4D PET, creates a mean 3D image of that frame, the one before it, and the one after it.
    Then, this function applies an automatic percentile thresholding of 90% to that mean image to generate a carotid mask.
    Finally, that carotid mask is applied to the 4D PET image, and the resulting 4D image undergoes percentile thresholding of the "percentile" value frame by frame to get the TAC.

    Args:
        necktangle_matrix (np.ndarray): A 4D numpy array representing the PET necktangle matrix.
        percentile (float): The percentile value to calculate the manual threshold.
        frame_midpoint_times (np.ndarray): An array of frame midpoint times.

    Returns:
        tac (np.ndarray): A 2D numpy array representing the time-activity curve (TAC) with frame midpoint times and manual thresholds.

    Raises:
        None.
    """
    first_ten_frames = necktangle_matrix[:, :, :, :10]
    frame_averages = np.nanmean(first_ten_frames, axis=(0, 1, 2))
    bolus_index = np.argmax(frame_averages)
    bolus_window_4d = necktangle_matrix[:, :, :, bolus_index - 1:bolus_index + 2]
    bolus_window_average_3d = np.nanmean(bolus_window_4d, axis=3)
    automatic_threshold_value = np.nanpercentile(bolus_window_average_3d, 90)
    automatic_threshold_mask_3d = np.where(bolus_window_average_3d > automatic_threshold_value, 1, np.nan)
    tac = np.zeros((2, necktangle_matrix.shape[3]))
    tac[0, :] = frame_midpoint_times
    for frame in range(tac.shape[1]):
        current_frame = necktangle_matrix[:, :, :, frame]
        automatic_masked_frame = np.where(automatic_threshold_mask_3d == 1, current_frame, np.nan)
        manual_threshold_value = np.nanpercentile(automatic_masked_frame, percentile)
        tac[1, frame] = manual_threshold_value

    return tac
