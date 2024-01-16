import numpy as np
import numba


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
# TODO: Should the zeroth index be 0?
@numba.njit()
def integrate_input_function(input_function: np.ndarray, input_times: np.ndarray) -> np.ndarray:
    """
    
    :param input_function:
    :param input_times:
    :return:
    """
    integral = np.zeros_like(input_function)
    for i in range(1, len(input_function)):
        integral[i] = np.trapz(y=input_function[1:i], x=input_times[1:i])
    return integral


# TODO: Add documentation
@numba.njit()
def generate_parametric_image_with_naive_patlak(intensity_image: np.ndarray,
                                                input_func: np.ndarray,
                                                input_times: np.ndarray,
                                                t_thresh: int) -> np.ndarray:
    """
    
    :param intensity_image:
    :param input_func:
    :param input_times:
    :param t_thresh:
    :return:
    """
    input_int = integrate_input_function(input_func, input_times)
    x_var = input_int[t_thresh:] / input_func[t_thresh:]
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
