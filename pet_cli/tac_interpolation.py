"""
This module provides classes for time-activity curve (TAC) interpolation for Positron Emission Tomography (PET)
data. It enables the resampling of data evenly with respect to time, which is particularly useful when performing
convolutions with respect to time. We use :py:class:`scipy.interpolate.interp1d` for the interpolation.

The module comprises two classes: :class:`EvenlyInterpolate` and :class:`EvenlyInterpolateWithMax`.

The :class:`EvenlyInterpolate` class takes in TAC times and values and a specified delta time, :math:`\\Delta t`, to
resample data by interpolating TACs evenly with respect to time.

The :class:`EvenlyInterpolateWithMax` class extends the functionality of `EvenlyInterpolate` by modifying the
calculation of delta time, :math:`\\Delta t`, to explicitly sample the maximum value of the TAC.

Example:

    
    .. plot::
        :include-source:
        
        import numpy as np
        from pet_cli.tac_interpolation import EvenlyInterpolate, EvenlyInterpolateWithMax
        import matplotlib.pyplot as plt
        
        # define some dummy TAC times and values
        tac_times = np.array([0., 1., 2.5, 4.1, 7.])
        tac_values = np.array([0., 0.8, 2., 1.5, 0.])
    
        # instantiate EvenlyInterpolate object and resample TAC (and add shift for better visualization)
        even_interp = EvenlyInterpolate(tac_times=tac_times, tac_values=tac_values+0.25, delta_time=1.0)
        resampled_tac = even_interp.get_resampled_tac()
    
        # instantiate EvenlyInterpolateWithMax object and resample TAC (and add shift for better visualization)
        even_interp_max = EvenlyInterpolateWithMax(tac_times=tac_times, tac_values=tac_values+0.5, samples_before_max=3)
        resampled_tac_max = even_interp_max.get_resampled_tac()
        
        fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(8,4))
        plt.plot(tac_times, tac_values, 'ko--', label='Raw TAC', zorder=2)
        plt.plot(*resampled_tac, 'ro-', label='Evenly Resampled TAC', zorder=1)
        plt.plot(*resampled_tac_max, 'bo-', label='Evenly Resampled TAC w/ Max', zorder=0)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('TAC Value', fontsize=16)
        plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
        plt.show()
        

Note:
    This module utilises :py:class:`scipy.interpolate.interp1d` for linear interpolation. Ensure Scipy is installed for
    this package to function.

"""

import numpy as np
from scipy.interpolate import interp1d as sp_interpolation
from typing import Tuple


class EvenlyInterpolate:
    """A class for basic evenly interpolating TACs with respect to time
    
    When performing convolutions with respect to time, care needs to be taken to account for the time-step between
    samples. One way to circumvent this problem is to resample data evenly with respect to the independent variable,
    or time.

    Uses :py:class:`scipy.interpolate.interp1d` to perform linear interpolation.

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

    def get_resampled_tac(self) -> np.ndarray:
        """
        Returns the resampled times and values of the Time-Activity Curve (TAC).

        The function combines the resampled times and values into a single numpy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array corresponds to the
            resampled times and the second array corresponds to the resampled activity values of the TAC.
            
        """
        return np.asarray([self.resample_times, self.resample_vals])
    
    
class EvenlyInterpolateWithMax(EvenlyInterpolate):
    r"""A class, extends :class:`EvenlyInterpolate`, and modifies the :math:`\Delta t` calculation such that the
    maximum value of the TAC is explicitly sampled.
    
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

        .. math:: \Delta t = \frac{t_{\mathrm{max} - t_{0}}{N}
        

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
