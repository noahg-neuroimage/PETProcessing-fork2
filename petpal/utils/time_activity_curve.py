"""
Class to handle data related to time activity curves (TACs).

TODO:
    * Add more unit handling functionality
    * Cover exception handling
    * Refactor safe_load_tac to this module as a public method

"""
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
