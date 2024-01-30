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
