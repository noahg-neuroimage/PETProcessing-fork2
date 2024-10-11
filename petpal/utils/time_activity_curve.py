"""
Class to handle data related to time activity curves (TACs).

TODO:
    * Add more unit handling functionality
    * Cover exception handling
    * Refactor safe_load_tac to this module as a public method

"""
import numpy as np
from .image_io import safe_load_tac

class TimeActivityCurve:
    """
    Class to handle data related to time activity curves (TACs).
    """
    def __init__(self,
                 tac_path: str):
        """
        Initialize TimeActivityCurve class

        Args:
            tac_path (str): Path to the TAC that will be analyzed.
        """
        self.tac_path = tac_path
        self.tac_times_in_minutes, self.tac_vals = self.get_tac_data()

    def get_tac_data(self):
        """
        Retrieves data from the TAC file. Uses :meth:`petpal.utils.image_io.safe_load_tac`.

        See also:
            * :meth:`petpal.utils.image_io.safe_load_tac`

        """
        return safe_load_tac(self.tac_path)

    def safe_load_tac(self, file_path: str):
        """
        Reads TAC from file. Currently placeholder that does nothing.
        """

    def safe_write_tac(self, file_path: str):
        """
        Writes TAC to file. Currently placeholder that does nothing.
        """

    def get_frame_durations(self):
        """
        Get array containing the duration of each frame in minutes.

        For a set of N frames, the first N-1 frame durations are estimated as the difference
        between each frame time and the next frame time. Frame N is then inferred as being the same
        duration as frame N-1.
        """
        tac_times_in_minutes = self.tac_times_in_minutes
        tac_durations_in_minutes = np.zeros((len(tac_times_in_minutes)))

        tac_durations_in_minutes[:-1] = tac_times_in_minutes[1:]-tac_times_in_minutes[:-1]
        tac_durations_in_minutes[-1] = tac_durations_in_minutes[-2]

        return tac_durations_in_minutes
