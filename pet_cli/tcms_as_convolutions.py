"""

"""

import numpy as np
import numba


@numba.njit()
def response_function_1tcm(t: np.ndarray[float], k1: float, k2: float) -> np.ndarray:
    """The response function for the 1TCM :math:`f(t)=k_1 e^{-k_{2}t}`
    
    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood

    Returns:
        (np.ndarray[float]): Array containing response function values given the constants.
    """
    return k1 * np.exp(-k2 * t)

