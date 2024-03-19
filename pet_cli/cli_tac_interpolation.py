import argparse
import numpy as np
from . import tac_interpolation as tac_intp


# TODO: Use the safe loading of TACs function from an IO module when it is implemented
def _safe_load_tac(filename: str) -> np.ndarray:
    """
    Loads time-activity curves (TAC) from a file.

    Tries to read a TAC from specified file and raises an exception if unable to do so. We assume that the file has two
    columns, the first corresponding to time and second corresponding to activity.

    Args:
        filename (str): The name of the file to be loaded.

    Returns:
        np.ndarray: A numpy array containing the loaded TAC. The first index corresponds to the times, and the second
        corresponds to the activity.

    Raises:
        Exception: An error occurred loading the TAC.
    """
    try:
        return np.array(np.loadtxt(filename).T, dtype=float, order='C')
    except Exception as e:
        print(f"Couldn't read file {filename}. Error: {e}")
        raise e

