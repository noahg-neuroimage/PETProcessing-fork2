import os
import argparse
from . import tac_fitting as pet_fit


def _generate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='pet-cli-tcm-fit',
                                     description='Command line interface for fitting Tissue Compartment Models (TCM) '
                                                 'to PET Time Activity Curves (TACs).',
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    # IO group
    grp_io = parser.add_argument_group('IO Paths and Prefixes')
    grp_io.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    grp_io.add_argument("-r", "--roi-tac-path", required=True, help="Path to the ROI TAC file.")
    grp_io.add_argument("-o", "--output-directory", required=True, help="Path to the output directory.")
    grp_io.add_argument("-p", "--output-filename-prefix", required=True, help="Prefix for the output filenames.")
    
    # Analysis group
    grp_analysis = parser.add_argument_group('Analysis Parameters')
    grp_analysis.add_argument("-t", "--input-fitting-threshold-in-mins", required=True, type=float,
                              help="Threshold in minutes for fitting the later half of the input function.")
    grp_analysis.add_argument("-m", "--model", required=True, choices=['1tcm', '2tcm-k4zero', 'serial-2tcm'],
                              help="Analysis method to be used.")
    grp_analysis.add_argument("-g", "--initial-guesses", required=False, nargs='+', type=float,
                              help="Initial guesses for each fitting parameter.")
    grp_analysis.add_argument("-l", "--lower-bounds", required=False, nargs='+', type=float,
                              help="Lower bounds for each fitting parameter.")
    grp_analysis.add_argument("-u", "--upper-bounds", required=False, nargs='+', type=float,
                              help="Upper bounds for each fitting parameter.")
    grp_analysis.add_argument("-f", "--max-fit-iterations", required=False, default=2500, type=int,
                              help="Maximum number of function iterations")
    
    
    # Printing arguments
    grp_verbose = parser.add_argument_group('Additional Options')
    grp_verbose.add_argument("--print", action="store_true", help="Whether to print the analysis results.")
    
    
    return parser.parse_args()

def main():
    args = _generate_args()