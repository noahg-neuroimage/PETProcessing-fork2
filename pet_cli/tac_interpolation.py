"""A sub-module for interpolating TACs evenly with respect to time.

"""

import numpy as np
from scipy.interpolate import interp1d as sp_interpolation

class EvenlyInterpolateTAC(object):
    """A class for interpolating TACs evenly with respect to time.
    
    When performing convolutions with respect to time, care needs to be taken to account for the time-step between
    samples. One way to circumvent this problem is to resample data evenly with respect to the independent variable,
    or time.
    
    Uses `scipy.interpolate.interp1d` to perform linear interpolation.
    
    """
    
    def __int__(self, tac_times: np.ndarray, tac_values: np.ndarray):
        """
        
        Args:
            tac_times:
            tac_values:

        Returns:
        
        """
        self._tac_times = tac_times
        self._tac_vals = tac_values
        self._interp_func = sp_interpolation(x=self._tac_times, y=self._tac_vals, kind='linear', bounds_error=True)
        
        
        
    