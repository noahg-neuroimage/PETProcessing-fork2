import inspect
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit as sp_cv_fit
from . import tcms_as_convolutions as pet_tcms
from . import blood_input as pet_bld
from abc import ABC, abstractmethod
import os


class TACFitter(object):
    def __init__(self,
                 pTAC: np.ndarray,
                 tTAC: np.ndarray,
                 weights: np.ndarray = None,
                 tcm_func: Callable = pet_tcms.generate_tac_2tcm_with_k4zero_cpet_from_tac,
                 fit_bounds: np.ndarray = None,
                 resample_num: int = 2048,
                 aif_fit_thresh_in_mins: float = 30.0):
        
        self.tcm_func = tcm_func
        self.number_of_fit_params = len(inspect.signature(tcm_func).parameters[2:])
        
        if fit_bounds is None:
            self.bounds = self.generate_fit_bounds()
        else:
            self.bounds = fit_bounds[:]
        
        assert fit_bounds.shape == (self.number_of_fit_params, 3), (
                                                                "Fit bounds has the wrong shape. For each potential"
                                                                " fitting parameter in `tcm_func`, we require the "
                                                                "tuple: `(initial, lower, upper)`.")
        
        self.initial_guesses = self.bounds[:, 0]
        self.lo_bounds = self.bounds[:, 1]
        self.hi_bounds = self.bounds[:, 2]
        
        self._p_tac = pTAC[:]
        self._t_tac = tTAC[:]
        if weights is None:
            self._weights = np.ones_like(self._t_tac[1])
        
        self.t_tac_san = self.sanitize_tac(*self._t_tac)
        self.resample_times = np.linspace(self.t_tac_san[0][0], self.t_tac_san[0][-1], resample_num)
        
        self.p_tac_san = self.sanitize_tac(*self._p_tac)
        self._p_tac_intp = pet_bld.BloodInputFunction(time=self.p_tac_san[0],
                                                      activity=self.p_tac_san[1],
                                                      thresh_in_mins=aif_fit_thresh_in_mins)
        
        self.t_tac = self.resampled_tac(*self.t_tac_san, self.resample_times)
        self.p_tac = np.asarray([self.resample_times[:],
                                 self._p_tac_intp.calc_blood_input_function(t=self.resample_times)])
        
        self.p_tac_vals = self._p_tac[1]
        self.tgt_tac_vals = self._t_tac[1]
        self.fit_results = None
    
    def generate_fit_bounds(self):
        bounds = np.zeros((self.number_of_fit_params, 3))
        for pid, param in enumerate(bounds[:-1]):
            bounds[pid][0] = [0.1, 1.0e-8, 5.0]
        bounds[-1] = [0.1, 0.0, 1.0]
        return bounds
    
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
    def resampled_tac(tac_times, tac_vals, new_times):
        return new_times, np.interp(x=new_times, xp=tac_times, fp=tac_vals)
        
    def fitting_func(self, x, *params):
        return self.tcm_func(x, self.p_tac_vals, *params)
    
    def run_fitting(self):
        self.fit_results = sp_cv_fit(f=self.fitting_func,
                                     xdata=self.resample_times,
                                     ydata=self.tgt_tac_vals,
                                     p0=self.initial_guesses,
                                     bounds=(self.lo_bounds, self.hi_bounds),
                                     sigma=self._weights)
