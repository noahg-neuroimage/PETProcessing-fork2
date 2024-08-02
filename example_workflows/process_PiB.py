""" Provides overall workflow for example PiB dataset, from 4D-PET to SUVR Parametric Images """

import os
import petpal.preproc as pp
import argparse

""" Argparse Configuration """


def options():
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: Ideally, this script can run on an entire BIDS project, correctly finding the relevant data.
    # parser.add_argument('-ib','--input_bids', required=True, help='Path to the top-level BIDS directory')

    parser.add_argument('-ip', '--input_pet', required=True, help='Path to the input PET image (.nii.gz)')
    parser.add_argument('-o', '--output_path', required=True,
                        help='Folder where all output and intermediate data will be stored')
    parser.add_argument('-im', '--input_mri', required=True, help='Path to the input mri image')
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # Unpack command line arguments
    args = options()
    if os.path.exists(args.input_pet):
        path_to_pet = args.input_pet
    else:
        # TODO: Actually handle this as an error
        print("PET File not found")
        return

    if os.path.exists(args.input_mri):
        path_to_mri = args.input_mri
    else:
        print("MRI File not found")
        return

    if not (os.path.exists(args.output_path)):
        os.mkdir(args.output_path)
        if args.verbose:
            print(f"Creating output directory at {args.output_path}")

    pib_half_life = 1221.84
    path_to_output = args.output_path
    path_to_moco_pet = os.path.join(path_to_output, "moco.nii.gz")
    # Step 1: Run Motion Correction on 4DPET image
    moco_4d_pet, params, framewise_displacement = pp.motion_corr.motion_corr(input_image_4d_path=path_to_pet,
                                                                             motion_target_option=(0, 600),
                                                                             out_image_path=path_to_moco_pet,
                                                                             verbose=True if args.verbose else False,
                                                                             half_life=pib_half_life)

    # Step 2: Register Moco'd PET to MRI space
    path_to_registered_pet = os.path.join(path_to_output, "registered_pet.nii.gz")
    registered_pet = pp.register.register_pet(input_reg_image_path=path_to_moco_pet,
                                              reference_image_path=path_to_mri,
                                              motion_target_option=(0, 600),
                                              out_image_path=path_to_registered_pet,
                                              verbose=True if args.verbose else False,
                                              half_life=pib_half_life)

    return


if __name__ == "__main__":
    main()
