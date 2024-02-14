"""Module for interpolating Time Activity Curves (TACs). Mostly used for evenly interpolating PET TACs which tend to be
sampled unevenly with respect to time.

"""

import numpy as np
from scipy.interpolate import interp1d as sp_interpolation
from typing import Tuple


class EvenlyInterpolate:
    """A class for basic evenly interpolating TACs with respect to time
    
    When performing convolutions with respect to time, care needs to be taken to account for the time-step between
    samples. One way to circumvent this problem is to resample data evenly with respect to the independent variable,
    or time.

    Uses `scipy.interpolate.interp1d` to perform linear interpolation.

    Attributes:
        interp_func (scipy.interpolate.interp1d): Interpolation function given the provided TAC.
        resample_times (np.ndarray): Array containing evenly spaced TAC times.
        resample_vals (np.ndarray): Interpolated activities at the calculated resample times.
    """
    def __init__(self, tac_times: np.ndarray[float], tac_values: np.ndarray[float], delta_time: float) -> None:
        r"""Constructor for EvenlyInterpolate.
        
        Uses ``scipy.interpolate.interp1d`` to perform linear interpolation of the provided TAC.

        Args:
            tac_times (np.ndarray[float]): The time-points of the provided TAC.
            tac_values (np.ndarray[float]): The activity values of the provided TAC.
            delta_time (float): The :math:`\Delta t` for the resampled times.
        """
        self._tac_times = tac_times
        self._tac_values = tac_values
        self.interp_func = sp_interpolation(x=self._tac_times, y=self._tac_values, kind='linear', bounds_error=True)
        self.resample_times = np.arange(tac_times[0], tac_times[-1], delta_time)
        self.resample_vals = self.interp_func(self.resample_times)

    def get_resampled_tac(self) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        return self.resample_times, self.resample_vals
    
    
    
class EvenlyInterpolateWithMax(EvenlyInterpolate):
    r"""A class, extends ``EvenlyInterpolate``, and modifies the :math:`\Delta t` calculation such that the maximum value
    of the TAC is explicitly sampled.
    Attributes:
        interp_func (scipy.interpolate.interp1d): Interpolation function given the provided TAC.
        resample_times (np.ndarray): Array containing evenly spaced TAC times.
        resample_vals (np.ndarray): Interpolated activities at the calculated resample times.
        dt (float): The :math:`\Delta t` for the resampled times such that the maximum value is explicitly sampled.
        
    See Also:
        :class:`EvenlyInterpolate`
        
    """
    def __init__(self, tac_times: np.ndarray[float], tac_values: np.ndarray[float], samples_before_max: float = 10.0):
        """
        
        Args:
            tac_times (np.ndarray[float]): The time-points of the provided TAC.
            tac_values (np.ndarray[float]): The activity values of the provided TAC.
            samples_before_max (float): Number of samples before the max TAC value. Defaults to 10.0.
        """
        self.dt = self.calculate_dt_for_even_spacing_with_max_sampled(tac_times, tac_values, samples_before_max)
        super().__init__(tac_times, tac_values, self.dt)
    
    @staticmethod
    def calculate_dt_for_even_spacing_with_max_sampled(tac_times: np.ndarray[float],
                                                       tac_values: np.ndarray[float],
                                                       samples_before_max: float) -> float:
        r"""Calculate :math:`\Delta t` such that TAC is evenly sampled while still sampling the maximum TAC value.

        .. math::
            \Delta t = \frac{t_{\mathrm{max} - t_{0}}{N}
        

        Args:
            tac_times (np.ndarray): Array containing TAC times.
            tac_values (np.ndarray): Array containing TAC activities.
            samples_before_max (float):

        Returns:
            (float): dt such that the TAC is evenly sampled and the TAC max is still explicitly sampled.
        """
        t_start = tac_times[0]
        t_for_max_val = tac_times[np.argmax(tac_values)]
        dt = (t_for_max_val - t_start) / samples_before_max
        return dt
