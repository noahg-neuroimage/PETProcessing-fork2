"""
CLI - TAC Interpolation
-----------------------

This module contains various functions related to the Time-Activity Curve (TAC)
interpolation process in the PET (Positron Emission Tomography) pipeline.

It includes the following main functions:

1. :func:`_safe_load_tac`: Safely loads TAC data from a file.

2. :func:`_safe_write_tac`: Safely writes the TAC data to a file.

3. :func:`_print_tac_to_screen`: Prints the TAC times and values to the console.

4. :func:`main`: The main function that invokes these processes.

TODO:
    * Refactor the reading and writing of TACs when IO module is mature.

"""

import argparse
import numpy as np
from . import tac_interpolation as tac_intp
import os


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


# TODO: Use the safe writing of TACs function from an IO module when it is implemented
def _safe_write_tac(tac_times: np.ndarray, tac_values: np.ndarray, filename: str) -> None:
    """
    Writes time-activity curves (TAC) to a file.

    Tries to write a TAC to the specified file and raises an exception if unable to do so. The TAC is expected to be a
    numpy array where the first index corresponds to the times, and the second corresponds to the activity.

    Args:
        tac_times (np.ndarray): A numpy array containing the time points of the TAC.
        tac_values (np.ndarray): A numpy array containing the activity values of the TAC.
        filename (str): The name of the file to write to.

    Raises:
        Exception: An error occurred writing the TAC.
    """
    out_arr = np.array([tac_times, tac_values]).T
    try:
        np.savetxt(fname=f"{filename}.tsv", X=out_arr, delimiter="\t", header="Time\tActivity", fmt="%.6e")
    except Exception as e:
        print(f"Couldn't write file {filename}. Error: {e}")
        raise e


def _print_tac_to_screen(tac_times: np.ndarray, tac_values: np.ndarray):
    """
    Prints the Time-Activity Curve (TAC) times and values to the console.

    This function takes as input two numpy arrays, one with the TAC times and the other with the TAC values, and prints
    them to the console in a formatted manner. The format is '%.6e\t%.6e'.

    Args:
        tac_times (np.ndarray): A numpy array containing the TAC times.
        tac_values (np.ndarray): A numpy array containing the TAC values.

    """
    print(f"#{'Time':<9}\tActivity")
    for time, value in zip(tac_times, tac_values):
        print(f"{time:<.6e}\t{value:<.6e}")


def main():
    parser = argparse.ArgumentParser(prog="TAC Interpolation", description="Evenly resample TACs.",
                                     epilog="TAC interpolation complete.")
    
    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument("-i", "--tac-path", help="Path to TAC file.", required=True)
    io_grp.add_argument("-o", "--out-tac-path", help="Path of output file.", required=True)
    
    interp_grp = parser.add_argument_group("Interpolation")
    mutually_exclusive_group = interp_grp.add_mutually_exclusive_group(required=True)
    mutually_exclusive_group.add_argument("--delta-time", type=float,
                                          help="The time difference for the resampled times.")
    mutually_exclusive_group.add_argument("--samples-before-max", type=float,
                                          help="Number of samples before the max TAC value.")
    
    verb_group = parser.add_argument_group("Additional information")
    verb_group.add_argument("-p", "--print", action="store_true", help="Print the resampled TAC values.",
                            required=False)
    verb_group.add_argument("-v", "--verbose", action="store_true", help="Print the sizes of the input and output TACs",
                            required=False)
    
    args = parser.parse_args()
    args.tac_path = os.path.abspath(args.tac_path)
    args.out_tac_path = os.path.abspath(args.out_tac_path)
    
    in_tac_times, in_tac_values = _safe_load_tac(args.tac_path)
    
    if args.samples_before_max is not None:
        interpolator = tac_intp.EvenlyInterpolateWithMax(tac_times=in_tac_times, tac_values=in_tac_values,
                                                         samples_before_max=args.samples_before_max)
    else:
        interpolator = tac_intp.EvenlyInterpolate(tac_times=in_tac_times, tac_values=in_tac_values,
                                                  delta_time=args.delta_time)
    
    resampled_times, resampled_values = interpolator.get_resampled_tac()
    
    _safe_write_tac(tac_times=resampled_times, tac_values=resampled_values, filename=args.out_tac_path)
    
    if args.verbose:
        print(f"Input TAC size:  {len(in_tac_values)}.")
        print(f"Output TAC size: {len(resampled_values)}.")
    
    if args.print:
        _print_tac_to_screen(tac_times=resampled_times, tac_values=resampled_values)


if __name__ == "__main__":
    main()
