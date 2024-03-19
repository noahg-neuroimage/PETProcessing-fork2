import argparse
from . import graphical_plots as pet_plt


def main():
    parser = argparse.ArgumentParser(prog="Graphical Analysis Plots",
                                     description="Generate graphical analysis plots of PET TACs.",
                                     epilog="Example: --input-tac-path /path/to/input.tac --roi-tac-path "
                                            "/path/to/roi.tac --threshold-in-mins 30.0 --method-name patlak "
                                            "--output-directory /path/to/output --output-filename-prefix plot")
    parser.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    parser.add_argument("-r", "--roi-tac-path", required=True, help="Path to the Region of Interest (ROI) TAC file.")
    parser.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                        help="Threshold in minutes below which data points will be discarded.")
    parser.add_argument("-m", "--method-name", required=True, choices=['patlak', 'logan', 'alt-logan'],
                        help="Name of the method for generating the plot.")
    parser.add_argument("-o", "--output-directory", required=True,
                        help="Path to the directory where the plot output will be saved.")
    parser.add_argument("-p", "--output-filename-prefix", default="",
                        help="An optional prefix for the output filenames.")
    args = parser.parse_args()
    
    grph_plot = pet_plt.Plot(input_tac_path=args.input_tac_path, roi_tac_path=args.roi_tac_path,
                             threshold_in_mins=args.threshold_in_mins, method_name=args.method_name,
                             output_directory=args.output_directory, output_filename_prefix=args.output_filename_prefix)
    
    grph_plot.save_figure()


if __name__ == "__main__":
    main()