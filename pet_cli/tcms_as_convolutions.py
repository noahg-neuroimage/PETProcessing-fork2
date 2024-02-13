"""A collection of functions to compute TACs as explicit convolutions for common Tissue Compartment Models (TCMs).


TODO:
    * Add the derivations of the solutions to the TCMs in the module docstring.
"""

import numba
import numpy as np


def calc_convolution_with_check(f: np.ndarray[float], g: np.ndarray[float], dt: float) -> np.ndarray:
    """Performs a discrete convolution of two arrays, assumed to represent time-series data.
    
    Let ``f``:math:`=f(t)` and ``g``:math:`=g(t)` where both functions are 0 for :math:`t\leq0`. Then,
    the output, :math:`h(t)`, is
    .. math::
        h(t) = \int_{0}^{t}f(s)g(s-t)\mathrm{d}s
    
    Args:
        f (np.ndarray[float]): Array containing the values for the input function.
        g (np.ndarray[float]): Array containing values for the response function.
        dt (np.ndarray[float]): The step-size, in the time-domain, between samples for ``f`` and ``g``.

    Returns:
        (np.ndarray): Convolution of the two arrays scaled by ``dt``.
        
    Notes:
        This function does not use `numba.njit()`.
    """
    assert len(f) == len(g), f"The provided arrays must have the same lengths! f:{len(f):<6} and g:{len(g):<6}."
    vals = np.convolve(f, g, mode='full')
    return vals[:len(f)] * dt


@numba.njit()
def response_function_1tcm_c1(t: np.ndarray[float], k1: float, k2: float) -> np.ndarray:
    r"""The response function for the 1TCM :math:`f(t)=k_1 e^{-k_{2}t}`
    
    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.

    Returns:
        (np.ndarray[float]): Array containing response function values given the constants.
    """
    return k1 * np.exp(-k2 * t)


@numba.njit()
def response_function_2tcm_with_k4zero_c1(t: np.ndarray[float], k1: float, k2: float, k3: float) -> np.ndarray:
    r"""The response function for first compartment in the serial 2TCM with :math:`k_{4}=0`; :math:`f(t)=k_{1}e^{-(k_{2} + k_{3})t}`.
    
    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        (np.ndarray[float]): Array containing response function values for first compartment given the constants.
        
    See Also:
        :func:`response_function_2tcm_with_k4zero_c2`
        
    """
    return k1 * np.exp(-(k2 + k3) * t)


@numba.njit()
def response_function_2tcm_with_k4zero_c2(t: np.ndarray[float], k1: float, k2: float, k3: float) -> np.ndarray:
    r"""The response function for second compartment in the serial 2TCM with :math:`k_{4}=0`; :math:`f(t)=\frac{k_{1}k_{3}}{k_{2}+k_{3}}(1-e^{-(k_{2} + k_{3})t})`.

    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        (np.ndarray[float]): Array containing response function values for first compartment given the constants.
    
    See Also:
        :func:`response_function_2tcm_with_k4zero_c1`
    """
    return ((k1 * k3) / (k2 + k3)) * (1.0 - np.exp(-(k2 + k3) * t))


@numba.njit()
def response_function_serial_2tcm_c1(t: np.ndarray[float], k1: float, k2: float, k3: float, k4: float) -> np.ndarray:
    r"""The response function for first compartment in the *serial* 2TCM.
    
    .. math::
        f(t) = \frac{k_{1}}{a} \left[ (k_{4}-\alpha_{1})e^{-\alpha_{1}t} + (\alpha_{2}-k_{4})e^{-\alpha_{2}t}\right]
    
    where
    
    .. math::
        \begin{align*}
        a&= k_{2}+k_{3}+k_{4}\\
        \alpha_{1}&=\frac{a-\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \alpha_{1}&=\frac{a+\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \end{align*}
    
    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from first tissue compartment to second tissue compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        (np.ndarray[float]): Array containing response function values for first compartment given the constants.
        
    See Also:
        * :func:`response_function_serial_2tcm_c2`
        * :func:`response_function_2tcm_with_k4zero_c1` for when :math:`k_{4}=0` (irreversible second compartment).
    """
    a = k2 + k3 + k4
    alpha_1 = (a - np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    alpha_2 = (a + np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    
    return (k1 / a) * ((k4 - alpha_1) * np.exp(-alpha_1 * t) + (alpha_2 - k4) * np.exp(-alpha_2 * t))


@numba.njit()
def response_function_serial_2tcm_c2(t: np.ndarray[float], k1: float, k2: float, k3: float, k4: float) -> np.ndarray:
    r"""The response function for second compartment in the *serial* 2TCM.

    .. math::
        f(t) = \frac{k_{1}k_{3}}{a} \left[ e^{-\alpha_{1}t} - e^{-\alpha_{2}t}\right]

    where

    .. math::
        \begin{align*}
        a&= k_{2}+k_{3}+k_{4}\\
        \alpha_{1}&=\frac{a-\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \alpha_{1}&=\frac{a+\sqrt{a^{2}-4k_{2}k_{4}}}{2}\\
        \end{align*}

    Args:
        t (np.ndarray[float]): Array containing time-points where :math:`t\geq0`.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from first tissue compartment to second tissue compartment.
        k4 (float): Rate constant for transport from second tissue compartment back to first tissue compartment.

    Returns:
        (np.ndarray[float]): Array containing response function values for second compartment given the constants.
        
    See Also:
        * :func:`response_function_serial_2tcm_c2`
        * :func:`response_function_2tcm_with_k4zero_c2` for when :math:`k_{4}=0` (irreversible second compartment).
    """
    a = k2 + k3 + k4
    alpha_1 = (a - np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    alpha_2 = (a + np.sqrt((a ** 2.) - 4.0 * k2 * k4)) / 2.0
    
    return (k1 * k3 / a) * (np.exp(-alpha_1 * t) - np.exp(-alpha_2 * t))


def generate_tac_1tcm_c1_from_tac(tac_times: np.ndarray[float], tac_vals: np.ndarray[float], k1: float, k2: float) -> \
np.ndarray[float, float]:
    r"""Calculate the TTAC, given the input TAC, for a 1TCM as an explicit convolution.
    
    Args:
        tac_times (np.ndarray[float]): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray[float]): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.

    Returns:
        ((np.ndarray[float], np.ndarray[float])): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        :func:`response_function_1tcm_c1` for more details about the 1TCM response function used for the convolution.
    """
    
    _resp_vals = response_function_1tcm_c1(t=tac_times, k1=k1, k2=k2)
    dt = tac_times[1] - tac_times[0]
    c1 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c1])


def generate_tac_2tcm_with_k4zero_c1_from_tac(tac_times: np.ndarray[float],
                                              tac_vals: np.ndarray[float],
                                              k1: float,
                                              k2: float,
                                              k3: float) -> np.ndarray[float, float]:
    """Calculate the TTAC of the first comparment, given the input TAC, for a 2TCM (with :math:`k_{4}=0`) as an explicit convolution.
    
    Args:
        tac_times (np.ndarray[float]): Array containing time-points where :math:`t\geq0` and equal time-steps.
        tac_vals (np.ndarray[float]): Array containing TAC activities.
        k1 (float): Rate constant for transport from plasma/blood to tissue compartment.
        k2 (float): Rate constant for transport from first tissue compartment back to plasma/blood.
        k3 (float): Rate constant for transport from tissue compartment to irreversible compartment.

    Returns:
        ((np.ndarray[float], np.ndarray[float])): Arrays containing the times and TTAC given the input TAC and parameters.
        
    See Also:
        * :func:`response_function_2tcm_with_k4zero_c1` for more details about the 2TCM response function of the first compartment used for the convolution.
        * :func:`response_function_2tcm_with_k4zero_c2` for more details about the 2TCM response function of the second compartment used for the convolution.
    """
    _resp_vals = response_function_2tcm_with_k4zero_c1(t=tac_times, k1=k1, k2=k2, k3=k3)
    dt = tac_times[1] - tac_times[0]
    c1 = calc_convolution_with_check(f=tac_vals, g=_resp_vals, dt=dt)
    return np.asarray([tac_times, c1])
