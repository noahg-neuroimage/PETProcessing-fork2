"""Collection of functions to perform graphical analysis on tissue activity curves (TACs).


"""

__version__ = '0.1'

import numpy as np
import numba
import typing

@numba.njit()
def _line_fitting_make_rhs_matrix_from_xdata(xdata: numba.float64[::1]) -> numba.float64[:, ::1]:
    """Generates the RHS matrix for linear least squares fitting

    Args:
        xdata (numba.float64[:]): array of independent variable values

    Returns:
        numba.float64[:,:]: 2D matrix where first column is `xdata` and the second column is 1s.
    """
    out_matrix = np.ones((len(xdata), 2), float)
    out_matrix[:, 0] = xdata
    return out_matrix


