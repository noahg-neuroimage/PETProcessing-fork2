"""A sub-module for interpolating TACs evenly with respect to time.

"""

import numpy as np
from scipy.interpolate import interp1d as sp_interpolation
from typing import Tuple

class EvenlyInterpolateTAC(object):
    """A class for interpolating TACs evenly with respect to time.
    
    When performing convolutions with respect to time, care needs to be taken to account for the time-step between
    samples. One way to circumvent this problem is to resample data evenly with respect to the independent variable,
    or time.
    
    Uses `scipy.interpolate.interp1d` to perform linear interpolation.
    
    Attributes:
        interp_func (scipy.interpolate.interp1d): Interpolation function given the provided TAC.
        resample_times (np.ndarray): Array containing evenly spaced TAC times.
        resample_vals (np.ndarray): Interpolated activities at the calculated resample times.
    """
    
    def __init__(self, tac_times: np.ndarray, tac_values: np.ndarray):
        """
        
        Args:
            tac_times:
            tac_values:

        Returns:
        
        """
        self._tac_times = tac_times
        self._tac_vals = tac_values
        self.interp_func = sp_interpolation(x=self._tac_times, y=self._tac_vals, kind='linear', bounds_error=True)
        self.resample_times = None
        self.resample_vals = None
        
    # TODO: Make sure the usage pattern is not clunky
    def calculate_resample_times(self, num_points_before_max: float = 10.0):
        """Calculate resample times such that we do not miss the max in the TAC and that we have at
        least ``num_points_before_max`` samples before the max.
        
        Args:
            num_points_before_max (int): Number of time-samples before maximum value in the provided TAC.

        Returns:

        """
        t_start = self._tac_times[0]
        t_end = self._tac_times[-1]
        time_for_max_val = self._tac_times[np.argmax(self._tac_vals)]
        new_dt = (time_for_max_val - t_start) / num_points_before_max
        new_times = np.arange(t_start, t_end, new_dt)
        self.resample_times = new_times
        
    # TODO: Add better documentation
    def calculate_resample_activities(self):
        """Given the calculated resampled times, we calculate the interpolated values at those resampled times.
        
        Returns:

        """
        assert self.resample_times is not None, "Resample times have not been calculated yet."
        self.resample_vals = self.interp_func(self.resample_times)
        
    def generate_resampled_tac(self, num_points_before_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        
        Args:
            num_points_before_max:

        Returns:

        """
        self.calculate_resample_times(num_points_before_max=num_points_before_max)
        self.calculate_resample_activities()
        return self.get_resample_tac()
    
        
    def get_resample_tac(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        
        Returns:
            (np.ndarray, np.ndarray):
        """
        assert (self.resample_vals is not None), "Resample values have not been calculated yet."
        return (self.resample_times, self.resample_vals)