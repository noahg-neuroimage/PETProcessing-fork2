"""Collection of functions to perform graphical analysis on tissue activity curves (TACs).


"""

__version__ = '0.1'

import numpy as np
import numba
import typing

@numba.njit()
def _line_fitting_make_rhs_matrix_from_xdata(xdata: np.ndarray) -> np.ndarray:
    """Generates the RHS matrix for linear least squares fitting

    Args:
        xdata (numba.float64[:]): array of independent variable values

    Returns:
        numba.float64[:,:]: 2D matrix where first column is `xdata` and the second column is 1s.
        
    """
    out_matrix = np.ones((len(xdata), 2), float)
    out_matrix[:, 0] = xdata
    return out_matrix

# TODO: Check if documentation is good.
@numba.njit()
def fit_line_to_data_using_lls(xdata: np.ndarray, ydata: np.ndarray) -> np.ndarray:
    """Find the linear least squares solution given the x and y variables.
    
    Performs a linear least squares fit to the provided data. Explicitly calls numpy's `linalg.lstsq` method by
    constructing the matrix equations. We assume that `xdata` and `ydata` have the same number of elements.
    
    Args:
        xdata: Array of independent variable values
        ydata: Array of dependent variable values

    Returns:
       Linear least squares solution. (m, b) values
       
    """
    make_2d_matrix = _line_fitting_make_rhs_matrix_from_xdata
    matrix = make_2d_matrix(xdata)
    fit_ans = np.linalg.lstsq(matrix, ydata)[0]
    return fit_ans