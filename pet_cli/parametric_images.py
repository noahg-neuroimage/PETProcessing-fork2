import numpy as np
import numba
from typing import Tuple, Callable
from . import graphical_analysis


@numba.njit()
def apply_linearized_analysis_to_all_voxels(pTAC_times: np.ndarray,
                                            pTAC_vals: np.ndarray,
                                            tTAC_img: np.ndarray,
                                            t_thresh_in_mins: float,
                                            analysis_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates parametric images for 4D-PET data using the provided analysis method.

    This function iterates over each voxel in the given `tTAC_img` and applies the provided `analysis_func`
    to compute analysis values. The `analysis_func` should be a numba.jit function for optimization and
    should be following a signature compatible with either of the following: patlak_analysis, logan_analysis,
    or alt_logan_analysis.

    Args:
        pTAC_times (np.ndarray): A 1D array representing the input TAC times in minutes.
        
        pTAC_vals (np.ndarray): A 1D array representing the input TAC values. This array should
                                be of the same length as `pTAC_times`.

        tTAC_img (np.ndarray): A 4D array representing the 3D PET image over time.
                               The shape of this array should be (x, y, z, time).

        t_thresh_in_mins (float): A float representing the threshold time in minutes.
                                  It is applied when calling the `analysis_func`.

        analysis_func (Callable): A numba.jit function to apply to each voxel for given PET data.
                                  It should take the following arguments:

                                    - input_tac_values: 1D numpy array for input TAC values
                                    - region_tac_values: 1D numpy array for regional TAC values
                                    - tac_times_in_minutes: 1D numpy array for TAC times in minutes
                                    - t_thresh_in_minutes: a float for threshold time in minutes

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two 3D numpy arrays representing the calculated slope image
                                       and the intercept image, each of the same spatial dimensions as
                                       `tTAC_img`.
    """
    img_dims = tTAC_img.shape
    
    slope_img = np.zeros((img_dims[0], img_dims[1], img_dims[2]), float)
    intercept_img = np.zeros_like(slope_img)
    
    for i in range(0, img_dims[0], 1):
        for j in range(0, img_dims[1], 1):
            for k in range(0, img_dims[2], 1):
                analysis_vals = analysis_func(input_tac_values=pTAC_vals, region_tac_values=tTAC_img[i, j, k, :],
                                              tac_times_in_minutes=pTAC_times, t_thresh_in_minutes=t_thresh_in_mins)
                slope_img[i, j, k] = analysis_vals[0]
                intercept_img[i, j, k] = analysis_vals[1]
    
    return slope_img, intercept_img


def generate_parametric_images_with_graphical_method(pTAC_times: np.ndarray,
                                                     pTAC_vals: np.ndarray,
                                                     tTAC_img: np.ndarray,
                                                     t_thresh_in_mins: float,
                                                     method_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates parametric images for 4D-PET data using a specified graphical analysis method.

    This function maps one of the predefined method names to the corresponding analysis function,
    and then generates parametric images by applying it to the given 4D-PET data using the
    `apply_linearized_analysis_to_all_voxels` function.

    Args:
        pTAC_times (np.ndarray): A 1D array representing the input TAC times in minutes.

        pTAC_vals (np.ndarray): A 1D array representing the input TAC values. This array should
                                be of the same length as `pTAC_times`.

        tTAC_img (np.ndarray): A 4D array representing the 3D PET image over time.
                               The shape of this array should be (x, y, z, time).

        t_thresh_in_mins (float): A float representing the threshold time in minutes.

        method_name (str): The analysis method's name to apply. Must be one of: 'patlak', 'logan', or 'alt_logan'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two 3D numpy arrays representing the calculated slope image
                                       and the intercept image, each of the same spatial dimensions as
                                       `tTAC_img`.

    Raises:
       ValueError: If the `method_name` is not one of the following: 'patlak', 'logan', 'alt_logan'.
    """
    
    if method_name == "patlak":
        analysis_func = graphical_analysis.patlak_analysis
    elif method_name == "logan":
        analysis_func = graphical_analysis.logan_analysis
    elif method_name == "alt_logan":
        analysis_func = graphical_analysis.alternative_logan_analysis
    else:
        raise ValueError("Invalid method_name! Must be either 'patlak' or 'logan' or 'alt-logan'")
    
    slope_img, intercept_img = apply_linearized_analysis_to_all_voxels(pTAC_times=pTAC_times, pTAC_vals=pTAC_vals,
                                                                       tTAC_img=tTAC_img,
                                                                       t_thresh_in_mins=t_thresh_in_mins,
                                                                       analysis_func=analysis_func)
    
    return slope_img, intercept_img


class ParametricImageAnalysis:
    def __init__(self,
                 input_tac_path: str,
                 pet4D_tac_path: str,
                 output_directory: str,
                 output_filename_prefix: str) -> None:
        self.input_tac_path = input_tac_path
        self.pet4D_tac_path = pet4D_tac_path
        self.output_directory = output_directory
        self.output_filename_prefix = output_filename_prefix
        self.analysis_props = self.init_analysis_props()
        self.slope_image: np.ndarray = None
        self.intercept_image: np.ndarray = None
    
    def init_analysis_props(self):
        props = {
            'FilePathPTAC': None,
            'FilePathTTAC': None,
            'ImageDimensions': None,
            'StartFrameTime': None,
            'EndFrameTime': None,
            'ThresholdTime': None,
            'NumberOfPointsFit': None,
            'SlopeMaximum': None,
            'SlopeMinimum': None,
            'SlopeMean': None,
            'SlopeVariance': None,
            'InterceptMaximum': None,
            'InterceptMinimum': None,
            'InterceptMean': None,
            'InterceptVariance': None,
            }
        return props
    
    def run_analysis(self, method_name: str):
        pTAC_times, pTAC_vals = _safe_load_tac(self.input_tac_path)
        
        
        pass
    