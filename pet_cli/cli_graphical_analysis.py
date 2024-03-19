import argparse
import pet_cli.graphical_analysis as pet_ga


def main():
    parser = argparse.ArgumentParser(description="Perform graphical analysis on TAC data.")
    
    # IO group
    grp_io = parser.add_argument_group('IO Paths and Prefixes')
    grp_io.add_argument("-i", "--input-tac-path", required=True, help="Path to the input TAC file.")
    grp_io.add_argument("-r", "--roi-tac-path", required=True, help="Path to the ROI TAC file.")
    grp_io.add_argument("-o", "--output-directory", required=True, help="Path to the output directory.")
    grp_io.add_argument("-p", "--output-filename-prefix", required=True, help="Prefix for the output filenames.")
    
    # Analysis group
    grp_analysis = parser.add_argument_group('Analysis Parameters')
    grp_analysis.add_argument("-t", "--threshold-in-mins", required=True, type=float,
                           help="Threshold in minutes for the analysis.")
    grp_analysis.add_argument("-m", "--method-name", required=True, choices=['patlak', 'logan', 'alt-logan'],
                           help="Analysis method to be used.")
    
    # Additional group arguments
    grp_verbose = parser.add_argument_group('Additional Options')
    grp_verbose.add_argument("--print", action="store_true", help="Whether to print the analysis results.")
    
    args = parser.parse_args()
    
    graphical_analysis = pet_ga.GraphicalAnalysis(input_tac_path=args.input_tac_path,
                                                  roi_tac_path=args.roi_tac_path,
                                                  output_directory=args.output_directory,
                                                  output_filename_prefix=args.output_filename_prefix)
    
    graphical_analysis.run_analysis(method_name=args.method_name, t_thresh_in_mins=args.threshold_in_mins)
    graphical_analysis.save_analysis()
    
    if args.print:
        # TODO: Print analyses. This feature will be implemented later.
        pass
    

if __name__ == "__main__":
    main()