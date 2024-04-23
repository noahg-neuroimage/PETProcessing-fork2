import numpy as np
from scipy.optimize import curve_fit as sp_fit
from . import tcms_as_convolutions as tcms_conv


def calc_srtm_tac(tac_times: np.ndarray, r1: float, k2: float, bp: float, ref_tac_vals: np.ndarray) -> np.ndarray:
    r"""
    Calculate the Time Activity Curve (TAC) using the Simplified Reference Tissue Model (SRTM) with the given reference
    TAC and kinetic parameters.
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.
    
    
    The SRTM TAC can be calculated as:
    
    .. math::
        
        C(t)=R_{1}C_\mathrm{R}(t) + \left(k_{2} - \frac{R_{1}k_{2}}{1+\mathrm{BP}}\right)C_\mathrm{R}(t)\otimes
        \exp\left(- \frac{k_{2}t}{1+\mathrm{BP}}\right),
    
    
    where :math:`C_\mathrm{R}(t)` is the reference TAC, :math:`R_{1}=\frac{k_1^\prime}{k_1}`, :math:`k_{2}` is the
    rate-constant from the tissue compartment to plasma, and :math:`\mathrm{BP}` is the binding potential.
    
    
    Args:
        tac_times (np.ndarray): Times for the reference TAC.
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        k2 (float): The rate constant for the transfer of the tracer from tissue compartment to plasma.
        bp (float): The binding potential of the tracer in the tissue.
        ref_tac_vals (np.ndarray): The values of the reference TAC.

    Returns:
        np.ndarray: TAC values calculated using SRTM.

    """
    first_term = r1 * ref_tac_vals
    bp_coeff = k2 / (1.0 + bp)
    exp_term = np.exp(-bp_coeff * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = (k2 - r1 * bp_coeff) * tcms_conv.calc_convolution_with_check(f=exp_term, g=ref_tac_vals, dt=dt)
    
    return first_term + second_term


def _calc_simplified_frtm_tac(tac_times: np.ndarray,
                              r1: float,
                              a1: float,
                              a2: float,
                              alpha_1: float,
                              alpha_2: float,
                              ref_tac_vals: np.ndarray) -> np.ndarray:
    r"""
    Calculate the Time Activity Curve (TAC) for the Full Reference Tissue Model (FRTM) given the reference TAC and
    simplified coefficients. The coefficients can be generated from kinetic constants using
    :func:`_calc_frtm_params_from_kinetic_params`
    
    .. important::
        This function assumes that the reference TAC is uniformly sampled with respect to time since we perform
        convolutions.
    
    We use a more compact form for the FRTM:
    
    .. math::
    
        C(t) = R_{1}C_\mathrm{R}(t) + \left[ A_{1}e^{-\alpha_{1}t} + A_{2}e^{-\alpha_{2}t} \right] \otimes
        C_\mathrm{R}(t),
    
    where :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`, and :math:`A_{1},\,A_{2},\,\alpha_1,\,\alpha_2` can be calculated
    from the underlying kinetic constants. See :func:`_calc_frtm_params_from_kinetic_params` for more details about
    the parameter calculation.
    
    Args:
        tac_times (np.ndarray): Times for the reference TAC.
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        a1 (float): Coefficient of the first exponential term.
        a2 (float): Coefficient of the second exponential term.
        alpha_1 (float): Coefficient inside the first exponential.
        alpha_2 (float): Coefficient inside the second exponential.
        ref_tac_vals (np.ndarray): The values of the reference TAC.

    Returns:
        np.ndarray: TAC values calculated using FRTM.

    """
    first_term = r1 * ref_tac_vals
    exp_funcs = a1 * np.exp(-alpha_1 * tac_times) + a2 * np.exp(-alpha_2 * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = tcms_conv.calc_convolution_with_check(f=exp_funcs, g=ref_tac_vals, dt=dt)
    return first_term + second_term


def _calc_frtm_params_from_kinetic_params(r1: float,
                                          k2: float,
                                          k3: float,
                                          k4: float) -> tuple[float, float, float, float, float]:
    r"""
    Calculates the parameters (coefficients) for the simplified FRTM function (:func:`_calc_simplified_frtm_tac`) given
    the kinetic constants.
    
    The parameters are defined as:
    
    .. math::
    
        \alpha_{1} &= \frac{\beta - \chi}{2}\\
        \alpha_{2} &= \frac{\beta + \chi}{2}\\
        A_{1} &= \left(\frac{k_{3} + k_{4} -\alpha_{2}}{\chi} \right)\left( \frac{k_{2}}{R_{1}} - \alpha_{2} \right)\\
        A_{2} &= \left(\frac{\alpha_{1}-k_{3} - k_{4} }{\chi} \right)\left( \frac{k_{2}}{R_{1}} - \alpha_{1} \right),
    
    where additionally we have:
    
    .. math::
    
        \alpha_{1} &= \frac{k_{2} + k_{3} + k_{4} - \sqrt{\left( k_{2} + k_{3} + k_{4} \right)^2 - 4k_{2}k_{4}}}{2}\\
        \alpha_{2} &= \frac{k_{2} + k_{3} + k_{4} + \sqrt{\left( k_{2} + k_{3} + k_{4} \right)^2 - 4k_{2}k_{4}}}{2}\\
        A_{1} &= \left( \frac{k_{3} + k_{4} -\alpha_{2}}{\alpha_{1} - \alpha_{2}} \right)\left( \frac{k_{2}}{R_{1}}
        - \alpha_{2} \right)\\
        A_{2} &= \left(  \frac{\alpha_{1}-k_{3} - k_{4} }{\alpha_{1} - \alpha_{2}} \right)\left( \frac{k_{2}}{R_{1}}
        - \alpha_{1} \right)
    
    
    Args:
        r1 (float): The ratio of the clearance rate of tracer from plasma to the reference to the transfer rate of the
            tracer from plasma to the tissue; :math:`R_{1}\equiv\frac{k_1^\prime}{k_1}`.
        k2 (float): The rate of tracer transfer from the first tissue compartment to plasma.
        k3 (float): The rate of tracer transfer from the first tissue compartment to the second tissue compartment.
        k4 (float): The rate of tracer transfer from the second tissue compartment to the first tissue compartment.

    Returns:
        tuple: (``r1``, ``a1``, ``a2``, ``alpha_1``, ``alpha_2``) parameters for :func:`_calc_simplified_frtm_tac`.
    """
    beta = k2 + k3 + k4
    chi = np.sqrt(beta ** 2. - 4.0 * k2 * k4)
    alpha_1 = (beta - chi) / 2.0
    alpha_2 = (beta + chi) / 2.0
    a1 = (k3 + k4 - alpha_2) / chi * (k2 / r1 - alpha_2)
    a2 = (alpha_1 - k3 - k4) / chi * (k2 / r1 - alpha_1)
    return r1, a1, a2, alpha_1, alpha_2


def calc_frtm_tac(tac_times: np.ndarray,
                  r1: float,
                  k2: float,
                  k3: float,
                  k4: float,
                  ref_tac_vals: np.ndarray) -> np.ndarray:
    r1_n, a1, a2, alpha_1, alpha_2 = _calc_frtm_params_from_kinetic_params(r1=r1, k2=k2, k3=k3, k4=k4)
    return _calc_simplified_frtm_tac(tac_times=tac_times, r1=r1_n, a1=a1, a2=a2, alpha_1=alpha_1, alpha_2=alpha_2,
                                     ref_tac_vals=ref_tac_vals)


def fit_srtm_model_to_tac(tgt_tac_vals: np.ndarray,
                          ref_tac_times: np.ndarray,
                          ref_tac_vals: np.ndarray,
                          r1_start: float = 0.5,
                          k2_start: float = 0.5,
                          bp_start: float = 0.5) -> tuple:
    def _fitting_srtm(tac_times, r1, k2, bp):
        return calc_srtm_tac(tac_times=tac_times, r1=r1, k2=k2, bp=bp, ref_tac_vals=ref_tac_vals)
    
    starting_values = [r1_start, k2_start, bp_start]
    
    return sp_fit(f=_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)


def fit_srtm_model_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                      ref_tac_times: np.ndarray,
                                      ref_tac_vals: np.ndarray,
                                      r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      k2_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      bp_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    def _fitting_srtm(tac_times, r1, k2, bp):
        return calc_srtm_tac(tac_times=tac_times, r1=r1, k2=k2, bp=bp, ref_tac_vals=ref_tac_vals)
    
    st_values = (r1_bounds[0], k2_bounds[0], bp_bounds[0])
    lo_values = (r1_bounds[1], k2_bounds[1], bp_bounds[1])
    hi_values = (r1_bounds[2], k2_bounds[2], bp_bounds[2])
    
    return sp_fit(f=_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals,
                  p0=st_values, bounds=[lo_values, hi_values])


def fit_frtm_model_to_tac(tgt_tac_vals: np.ndarray,
                          ref_tac_times: np.ndarray,
                          ref_tac_vals: np.ndarray,
                          r1_start: float = 0.5,
                          k2_start: float = 0.5,
                          k3_start: float = 0.5,
                          k4_start: float = 0.5) -> tuple:
    def _fitting_frtm(tac_times, r1, k2, k3, k4):
        return calc_frtm_tac(tac_times=tac_times, r1=r1, k2=k2, k3=k3, k4=k4, ref_tac_vals=ref_tac_vals)

    starting_values = (r1_start, k2_start, k3_start, k4_start)
    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)


def fit_frtm_model_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                      ref_tac_times: np.ndarray,
                                      ref_tac_vals: np.ndarray,
                                      r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      k2_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      k3_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      k4_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    def _fitting_frtm(tac_times, r1, k2, k3, k4):
        return calc_frtm_tac(tac_times=tac_times, r1=r1, k2=k2, k3=k3, k4=k4, ref_tac_vals=ref_tac_vals)
    
    st_values = (r1_bounds[0], k2_bounds[0], k3_bounds[0], k4_bounds[0])
    lo_values = (r1_bounds[1], k2_bounds[1], k3_bounds[1], k4_bounds[1])
    hi_values = (r1_bounds[2], k2_bounds[2], k3_bounds[2], k4_bounds[2])
    
    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals,
                  p0=st_values, bounds=[lo_values, hi_values])
