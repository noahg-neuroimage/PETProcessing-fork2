from scipy.optimize import curve_fit as sp_fit
from scipy.interpolate import interp1d as sp_interp
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


# TODO: Maybe a class that tracks unitful quantities so we don't have to worry about units
class BloodInputFunction(object):
    """
    A general purpose class to deal with blood input function related data. The primarily functionality is to be able to
    compute the blood input function at any time, given the raw time and activity data. Using a manual threshold, we split
    the raw data into two parts:
    - Below the threshold, we have a simple linear interpolation.
    - Above the threshold, we fit a line to the data.
    When the object is instantiated, we automatically find the interpolation and the fit. Then, we can simply call the
    `calc_blood_input_function` function to give us blood input function values at any time.
    Lastly, note that this class can be used to deal with any type of (x,y) data where we wish to interpolate the first
    half, and fit a line to the second half.
    """
    
    def __init__(self, time: np.ndarray, activity: np.ndarray, thresh_in_mins: float):
        """
        Given the input time, activity, and threshold, we calculate the interpolating function for the first half (before
        the threshold) and the linear fit for the data in the second half (after the threshold). The threshold corresponds
        to the time, and not the activity.
        Currently, there must be at least 3 data points beyond the threshold to fit a line; else, we raise an `AssertionError`
        :param time: Time, in minutes.
        :param activity: Activity, assumed to be decay-corrected, corresponding to the times.
        :param thresh_in_mins: The threshold time, in minutes, such that before thresh we use an interpolant, and after thresh we use a linear fit.
        """
        assert time.shape == activity.shape, "`time` and `activity` must have the same dimensions."
        assert np.sum(time >= thresh_in_mins) >= 3, "Need at least 3 data-points above `thresh` to fit a line"
        self.thresh = thresh_in_mins
        below_thresh = time < self.thresh
        above_thresh = time >= self.thresh
        
        self._raw_times = time
        self._raw_activity = activity
        
        self.below_func = sp_interp(x=time[below_thresh], y=activity[below_thresh], assume_sorted=True, kind='linear',
                                    fill_value='extrapolate')
        
        self.above_func = BloodInputFunction.linear_fitting_func(x_data=time[above_thresh],
                                                                 y_data=activity[above_thresh])
    
    def calc_blood_input_function(self, x: np.ndarray):
        """
        Given new time data, assumed to be in minutes,
        :param x:
        :return:
        """
        y = np.zeros_like(x)
        below_thresh = x < self.thresh
        above_thresh = x >= self.thresh
        
        y[below_thresh] = self.below_func(x[below_thresh])
        y[above_thresh] = self.above_func(x[above_thresh])
        
        return y
    
    @staticmethod
    def _linear_function(x: np.ndarray, m: float, b: float):
        """
        Simple equation for a line. `y = m * x + b`
        :param x: Independent variable
        :param m: Slope of the line
        :param b: Intercept of the line
        :return: m * x + b
        """
        return m * x + b
    
    @staticmethod
    def linear_fitting_func(x_data: np.ndarray, y_data: np.ndarray):
        """
        Given x-data and y-data, we return a function corresponding to the linear fit.
        :param x_data: Independent variable
        :param y_data: Dependent variable corresponding to the x_data
        :return: A callable function that takes x-data as an input to compute the line values
        """
        
        popt, _ = sp_fit(f=BloodInputFunction._linear_function, xdata=x_data, ydata=y_data)
        
        def fitted_line_function(x):
            return BloodInputFunction._linear_function(x, *popt)
        
        return fitted_line_function
