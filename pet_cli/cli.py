import argparse
import pathlib
import image_derived_input_function as idif
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--mask")
    parser.add_argument("-i", "--images")

    parser.add_argument("-b", "--start", type=int)
    parser.add_argument("-e", '--stop', type=int)
    parser.add_argument("-s", "--step", type=int)

    parser.add_argument("-o", "--outfile")
    parser.add_argument("-p", "--print", action="store_true")
    
    args = parser.parse_args()
    
    assert pathlib.Path(args.mask).is_file(), "Mask file path is incorrect or does not exist."
    assert pathlib.Path(args.images).is_file(), "Images file path is incorrect or does not exist."

    mask = idif.extract_from_nii_as_numpy(file_path=args.mask)
    # print(mask.shape)
    images = idif.extract_from_nii_as_numpy(file_path=args.images)
    # print(images.shape)

    avg_vals = idif.compute_average_over_mask_of_multiple_frames(mask=mask, image_series=images,
                                                                 start=args.start, stop=args.stop, step=args.step)

    np.savetxt(fname=args.outfile, X=avg_vals, delimiter=', ')
    
    if args.print:
        print(avg_vals.shape)
        print(avg_vals)