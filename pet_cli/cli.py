import argparse
import pathlib
import image_derived_input_function as idif
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    
    assert pathlib.Path(args.mask).is_file(), f"Mask file path (${args.mask}) is incorrect or does not exist."
    assert pathlib.Path(args.images).is_file(), f"Images file path (${args.images}) is incorrect or does not exist."
    
    mask = idif.extract_from_nii_as_numpy(file_path=args.mask, verbose=args.verbose)
    images = idif.extract_from_nii_as_numpy(file_path=args.images, verbose=args.verbose)
    
    avg_vals = idif.compute_average_over_mask_of_multiple_frames(mask=mask, image_series=images, start=args.start,
                                                                 stop=args.stop, step=args.step)
    np.savetxt(fname=args.outfile, X=avg_vals, delimiter=', ')
    
    if args.print:
        print(avg_vals.shape)
        print(avg_vals)
