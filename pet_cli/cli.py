import argparse
import image_derived_input_function as idif


def main():
    parser = argparse.ArgumentParser(prog="PET Processing",
                                     description="General purpose suite for processing PET images.",
                                     epilog="PET Processing complete.")
    
    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument("-m", "--mask", help="Path to mask image.", required=True)
    io_grp.add_argument("-i", "--images", help="Path to images to be processed.", required=True)
    io_grp.add_argument("-o", "--outfile", help="Path of output file.", required=True)
    
    stride_grp = parser.add_argument_group("Image strides")
    stride_grp.add_argument("-b", "--start", type=int, help="Index of first frame to process.", required=True)
    stride_grp.add_argument("-e", '--stop', type=int, help="Index of last frame to process.", required=True)
    stride_grp.add_argument("-s", "--step", type=int, help="Step size for striding over images.", required=True)
    
    verb_group = parser.add_argument_group("Additional information")
    verb_group.add_argument("-p", "--print", action="store_true", help="Print the calculated values to screen.",
                            required=False)
    verb_group.add_argument("-v", "--verbose", action="store_true",
                            help="Print the shape of the mask and images files.", required=False)
    
    args = parser.parse_args()
    
    idif.calculate_and_save_image_derived_input_function(mask_file=args.mask, pet_file=args.images,
                                                         out_file=args.outfile, start=args.start, step=args.step,
                                                         stop=args.stop, verbose=args.verbose,
                                                         print_to_screen=args.print)


if __name__ == '__main__':
    main()