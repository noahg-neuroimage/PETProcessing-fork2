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


def _safe_write_tac(tac_times: np.ndarray, tac_values : np.ndarray, filename:str) -> None:
    
    out_arr = np.array([tac_times, tac_values]).T
    np.savetxt(fname=filename, X=out_arr, delimiter="\t", header="Time\tActivity", fmt="%.6e")


def main():
    parser = argparse.ArgumentParser(prog="TAC Interpolation", description="Resample unevenly sampled TACs",
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
    
    args = parser.parse_args()
    
    tac_times, tac_values = _safe_load_tac(args.tac_path)
    
    if args.samples_before_max is not None:
        interpolator = tac_intp.EvenlyInterpolateWithMax(tac_times=tac_times, tac_values=tac_values,
                                                         samples_before_max=args.samples_before_max)
    else:
        interpolator = tac_intp.EvenlyInterpolate(tac_times=tac_times, tac_values=tac_values,
                                                  delta_time=args.delta_time)
    
    resampled_tac = interpolator.get_resampled_tac()
    
    _safe_write_tac(tac_times=resampled_tac[0], tac_values=resampled_tac[1], filename=args.out_tac_path)

if __name__ == "__main__":
    main()