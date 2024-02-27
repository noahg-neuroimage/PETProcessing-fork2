import nibabel
import numpy as np
import numba
from scipy.integrate import cumulative_trapezoid

from typing import Tuple

# TODO: Add documentation
@numba.njit()
def make_rhs_matrix_for_linear_fit(xdata: np.ndarray) -> np.ndarray:
    """
    
    :param xdata:
    :return:
    """
    mat_A = np.ones((len(xdata), 2))
    for i in range(len(xdata)):
        mat_A[i, 0] = xdata[i]
    return mat_A


# TODO: Add documentation
@numba.njit()
def line_fit(xdata: np.ndarray, ydata: np.ndarray) -> float:
    """
    
    :param xdata:
    :param ydata:
    :return:
    """
    matrix = make_rhs_matrix_for_linear_fit(xdata)
    fit_ans = np.linalg.lstsq(matrix, ydata)[0]
    return fit_ans



# TODO: Add documentation
@numba.njit()
def generate_parametric_image_with_patlak(intensity_image: np.ndarray,
                                          input_func: np.ndarray,
                                          input_cum_int: np.ndarray,
                                          t_thresh: int) -> np.ndarray:
    """
    
    :param intensity_image:
    :param input_func:
    :param input_cum_int:
    :param t_thresh:
    :return:
    """
    
    x_var = input_cum_int[t_thresh:] / input_func[t_thresh:]
    image_shape = intensity_image.shape
    x_dim = image_shape[0]
    y_dim = image_shape[1]
    z_dim = image_shape[2]
    par_image = np.zeros((x_dim, y_dim, z_dim), float)
    
    for i in range(0, x_dim, 1):
        for j in range(0, y_dim, 1):
            for k in range(0, z_dim, 1):
                yVar = intensity_image[i, j, k, t_thresh:] / input_func[t_thresh:]
                fit_vals = line_fit(x_var, yVar)
                par_image[i, j, k] = fit_vals[0]
    return par_image

#TODO: Should re-factor into a IO sub-module.
def read_tsv_tac(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    times, tac = np.loadtxt(fname=file_path, delimiter='\t', dtype=float).T
    return np.array(times, float), np.array(tac, float)

#TODO: Add documentation
def save_parametric_image_from_4DPET_using_patlak(pet_file: str,
                                                  input_func_file: str,
                                                  out_file: str,
                                                  thresh_in_mins: float,
                                                  verbose=True):
    """
    
    :param pet_file:
    :param input_func_file:
    :param out_file:
    :param thresh_in_mins:
    :param verbose:
    :return:
    """
    pet_file = nibabel.load(filename=pet_file)
    
    input_times, input_tac = read_tsv_tac(input_func_file)
    try:
        t_thresh = np.argwhere(input_times >= thresh_in_mins)[0, 0]
    except IndexError:
        print(f"There are no data for t>=t* ({thresh_in_mins} minutes). Consider lowering the threshold.")
        raise IndexError
    
    if verbose:
        print(f"{len(input_times)-t_thresh} points will be fit for each TAC.")
    
    input_cum_int = np.array(cumulative_trapezoid(y=input_tac, x=input_times, initial=0.0), float)
    
    parametric_values = generate_parametric_image_with_patlak(intensity_image=pet_file.get_fdata()/37000.,
                                                              input_func=input_tac,
                                                              input_cum_int=input_cum_int,
                                                              t_thresh=t_thresh)

    parametric_image = nibabel.Nifti1Image(parametric_values, pet_file.affine, pet_file.header)

    nibabel.save(img=parametric_image, filename=f"{out_file}.nii.gz")
    
    return None


@numba.njit()
def generate_graphical_analysis_parametric_images_with_method(pTAC_times: np.ndarray,
                                                              pTAC_vals: np.ndarray,
                                                              tTAC_img: np.ndarray,
                                                              t_thresh_in_mins: float,
                                                              analysis_func) -> Tuple[np.ndarray, np.ndarray]:
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
