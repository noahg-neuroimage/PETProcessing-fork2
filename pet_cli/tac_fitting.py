import inspect
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit as sp_cv_fit
from . import tcms_as_convolutions as pet_tcms
from . import blood_input as pet_bld
import os


def get_fitting_params_for_tcm_func(f: Callable):
    return list(inspect.signature(f).parameters.keys())[2:]


def get_number_of_fit_params_for_tcm_func(f: Callable):
    return len(get_fitting_params_for_tcm_func(f))


class TACFitter(object):
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: np.ndarray = None,
                 tcm_func: Callable = pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                 fit_bounds: np.ndarray = None,
                 resample_num: int = 2048,
                 aif_fit_thresh_in_mins: float = 30.0):
        
        self.tcm_func = None
        self.number_of_fit_params = None
        self.fit_param_names = None
        self.weights = None
        self.bounds = None
        self.initial_guesses = None
        self.lo_bounds = None
        self.hi_bounds = None
        
        self.get_tcm_func_properties(tcm_func)
        self.set_bounds_and_initial_guesses(fit_bounds)
        
        self.raw_p_tac = pTAC.copy()
        self.raw_t_tac = tTAC.copy()
        self.t_tac_san = None
        self.resample_times = None
        self.delta_t = None
        self.p_tac_san = None
        self._p_tac_intp = None
        self.t_tac = None
        self.p_tac = None
        
        self.resample_tacs_evenly(aif_fit_thresh_in_mins, resample_num)
        
        self.set_weights(weights)
        
        self.p_tac_vals = self.p_tac[1]
        self.tgt_tac_vals = self.t_tac[1]
        self.fit_results = None
    
    def set_weights(self, weights) -> None:
        
        if isinstance(weights, float):
            tmp_ar = np.sqrt(np.exp(-weights * self.t_tac[0]) * self.t_tac[1])
            zero_idx = tmp_ar == 0.0
            tmp_ar[zero_idx] = np.inf
            self.weights = tmp_ar
        elif isinstance(weights, np.ndarray):
            self.weights = np.interp(x=self.t_tac[0], xp=self.raw_t_tac[0], fp=weights)
        else:
            self.weights = np.ones_like(self.t_tac[1])
    
    def set_bounds_and_initial_guesses(self, fit_bounds: np.ndarray) -> None:
        if fit_bounds is not None:
            assert fit_bounds.shape == (self.number_of_fit_params, 3), (
                "Fit bounds has the wrong shape. For each potential"
                " fitting parameter in `tcm_func`, we require the "
                "tuple: `(initial, lower, upper)`.")
            self.bounds = fit_bounds.copy()
        else:
            bounds = np.zeros((self.number_of_fit_params, 3), float)
            for pid, param in enumerate(bounds[:-1]):
                bounds[pid] = [0.1, 1.0e-8, 5.0]
            bounds[-1] = [0.1, 0.0, 1.0]
            self.bounds = bounds.copy()
        
        self.initial_guesses = self.bounds[:, 0]
        self.lo_bounds = self.bounds[:, 1]
        self.hi_bounds = self.bounds[:, 2]
        
    def resample_tacs_evenly(self, fit_thresh_in_mins: float, resample_num: int):
        self.t_tac_san = self.sanitize_tac(*self.raw_t_tac)
        self.resample_times = np.linspace(self.t_tac_san[0][0], self.t_tac_san[0][-1], resample_num)
        self.delta_t = self.resample_times[1] - self.resample_times[0]
        
        self.p_tac_san = self.sanitize_tac(*self.raw_p_tac)
        self._p_tac_intp = pet_bld.BloodInputFunction(time=self.p_tac_san[0], activity=self.p_tac_san[1],
                                                      thresh_in_mins=fit_thresh_in_mins)
        
        self.t_tac = self.resample_tac_on_new_times(*self.t_tac_san, self.resample_times)
        self.p_tac = np.asarray(
                [self.resample_times[:], self._p_tac_intp.calc_blood_input_function(t=self.resample_times)])
        
    def get_tcm_func_properties(self, tcm_func):
        assert tcm_func in [pet_tcms.generate_tac_1tcm_c1_from_tac,
                            pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                            pet_tcms.generate_tac_serial_2tcm_cpet_from_tac], \
            ("`tcm_func should be one of `pet_tcms.generate_tac_1tcm_c1_from_tac`, "
             "`pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac`, "
             "`pet_tcms.generate_tac_serial_2tcm_cpet_from_tac`")
        
        self.tcm_func = tcm_func
        self.fit_param_names = get_fitting_params_for_tcm_func(self.tcm_func)
        self.number_of_fit_params = len(self.fit_param_names)
    
    @staticmethod
    def sanitize_tac(tac_times, tac_vals):
        assert tac_times.shape == tac_vals.shape, "`tac_times` and `tac_vals` must have the same shape."
        if tac_times[0] != 0.0:
            return np.asarray([np.append(0, tac_times), np.append(0, tac_vals)])
        else:
            out_vals = tac_vals[:]
            out_vals[0] = 0.0
            return np.asarray([tac_times, out_vals])
    
    @staticmethod
    def resample_tac_on_new_times(tac_times, tac_vals, new_times):
        return new_times, np.interp(x=new_times, xp=tac_times, fp=tac_vals)
        
    def fitting_func(self, x, *params):
        return self.tcm_func(x, self.p_tac_vals, *params)[1]
    
    def run_fitting(self):
        self.fit_results = sp_cv_fit(f=self.fitting_func,
                                     xdata=self.resample_times,
                                     ydata=self.tgt_tac_vals,
                                     p0=self.initial_guesses,
                                     bounds=(self.lo_bounds, self.hi_bounds),
                                     sigma=self.weights)
