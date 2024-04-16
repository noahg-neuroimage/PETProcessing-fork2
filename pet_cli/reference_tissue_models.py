import numpy as np
from scipy.optimize import curve_fit as sp_fit
from . import tcms_as_convolutions as tcms_conv


def calc_srtm_tac(tac_times: np.ndarray, r1: float, k2: float, bp: float, ref_tac_vals: np.ndarray) -> np.ndarray:
    first_term = r1 * ref_tac_vals
    bp_coeff = k2 / (1.0 + bp)
    exp_term = np.exp(-bp_coeff * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = (k2 - r1 * bp_coeff) * tcms_conv.calc_convolution_with_check(f=exp_term, g=ref_tac_vals, dt=dt)
    
    return first_term + second_term


def calc_frtm_tac(tac_times: np.ndarray,
                  r1: float,
                  a1: float,
                  a2: float,
                  alpha_1: float,
                  alpha_2: float,
                  ref_tac_vals: np.ndarray) -> np.ndarray:
    first_term = r1 * ref_tac_vals
    exp_funcs = a1 * np.exp(-alpha_1 * tac_times) + a2 * np.exp(-alpha_2 * tac_times)
    dt = tac_times[1] - tac_times[0]
    second_term = tcms_conv.calc_convolution_with_check(f=exp_funcs, g=ref_tac_vals, dt=dt)
    return first_term + second_term


def calc_frtm_params_from_kinetic_params(r1: float,
                                         k2: float,
                                         k3: float,
                                         k4: float) -> tuple[float, float, float, float, float]:
    beta = k2 + k3 + k4
    chi = np.sqrt(beta ** 2. - 4.0 * k2 * k4)
    alpha_1 = (beta - chi) / 2.0
    alpha_2 = (beta + chi) / 2.0
    a1 = (k3 + k4 - alpha_2) / chi * (k2 / r1 - alpha_2)
    a2 = (alpha_1 - k3 - k4) / chi * (k2 / r1 - alpha_1)
    return r1, a1, a2, alpha_1, alpha_2


def fit_srtm_model_to_tac(tgt_tac_vals: np.ndarray,
                          ref_tac_times: np.ndarray,
                          ref_tac_vals: np.ndarray,
                          r1_start: float = 0.5,
                          k2_start: float = 0.5,
                          bp_start: float = 0.5) -> tuple:
    def _gen_fitting_srtm(tac_times, r1, k2, bp):
        return calc_srtm_tac(tac_times=tac_times, r1=r1, k2=k2, bp=bp, ref_tac_vals=ref_tac_vals)
    
    starting_values = [r1_start, k2_start, bp_start]
    
    return sp_fit(f=_gen_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)


def fit_srtm_model_to_tac_with_bounds(tgt_tac_vals: np.ndarray,
                                      ref_tac_times: np.ndarray,
                                      ref_tac_vals: np.ndarray,
                                      r1_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      k2_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0]),
                                      bp_bounds: np.ndarray = np.asarray([0.5, 0.0, 10.0])) -> tuple:
    def _gen_fitting_srtm(tac_times, r1, k2, bp):
        return calc_srtm_tac(tac_times=tac_times, r1=r1, k2=k2, bp=bp, ref_tac_vals=ref_tac_vals)
    
    starting_values = (r1_bounds[0], k2_bounds[0], bp_bounds[0])
    lo_values = (r1_bounds[1], k2_bounds[1], bp_bounds[1])
    hi_values = (r1_bounds[2], k2_bounds[2], bp_bounds[2])
    
    return sp_fit(f=_gen_fitting_srtm, xdata=ref_tac_times, ydata=tgt_tac_vals,
                  p0=starting_values, bounds=[lo_values, hi_values])


def fit_frtm_model_to_tac(tgt_tac_vals: np.ndarray,
                          ref_tac_times: np.ndarray,
                          ref_tac_vals: np.ndarray,
                          r1_start: float = 0.5,
                          k2_start: float = 0.5,
                          k3_start: float = 0.5,
                          k4_start: float = 0.5) -> tuple:
    def _fitting_frtm(tac_times, r1_n, k2, k3, k4):
        r1, a1, a2, alpha_1, alpha_2 = calc_frtm_params_from_kinetic_params(r1=r1_n, k2=k2, k3=k3, k4=k4)
        return calc_frtm_tac(tac_times=tac_times,
                             r1=r1, a1=a1, a2=a2, alpha_1=alpha_1, alpha_2=alpha_2,
                             ref_tac_vals=ref_tac_vals)

    starting_values = (r1_start, k2_start, k3_start, k4_start)
    return sp_fit(f=_fitting_frtm, xdata=ref_tac_times, ydata=tgt_tac_vals, p0=starting_values)
