""" Provides overall workflow for example PiB dataset, from 4D-PET to SUVR Parametric Images """

import os
import petpal as pp
import argparse
import glob

""" Argparse Configuration """


def options():
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: Ideally, this script can run on an entire BIDS project, correctly finding the relevant data.
    parser.add_argument('-ib', '--input_bids', required=True, help='Path to the top-level BIDS directory')

    parser.add_argument('-ip', '--input_pet', required=True, help='Path to the input PET image (.nii.gz)')
    parser.add_argument('-o', '--output_path', required=True,
                        help='Folder where all output and intermediate data will be stored')
    parser.add_argument('-im', '--input_mri', required=True, help='Path to the input mri image')
    parser.add_argument('-is', '--input_segmentation', required=True, help='Path to segmentation file (.nii.gz) '
                                                                           'containing all ROIs')
    parser.add_argument('-d', '--input_dseg')
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # Unpack command line arguments
    args = options()
    if os.path.exists(args.input_pet):
        path_to_pet = args.input_pet
    else:
        raise FileNotFoundError(f'No PET file found at {args.input_pet}')

    if os.path.exists(args.input_mri):
        path_to_mri = args.input_mri
    else:
        raise FileNotFoundError(f'No MRI file found at {args.input_mri}')

    if os.path.exists(args.input_segmentation):
        path_to_segmentation = args.input_segmentation
    else:
        raise FileNotFoundError(f'No segmentation file found at {args.input_segmentation}')

    if os.path.exists(args.input_dseg):
        path_to_dseg = args.input_dseg
    else:
        raise FileNotFoundError(f'No dseg.tsv file found at {args.input_segmentation}')

    if not (os.path.exists(args.output_path)):
        os.mkdir(args.output_path)
        if args.verbose:
            print(f"Creating output directory at {args.output_path}")

    if not (os.path.exists(args.input_bids)):
        raise FileNotFoundError(f'No such directory {args.input_bids}')

    # Parse through BIDS directory to define input and output paths
    bids_dir = args.input_bids
    subjects_list = glob.glob('sub-*')


    # mri_list = glob.glob('sub-*/*/anat/*T1w.nii.gz')
    # pet_list = glob.glob('sub-*/*/pet/*pet.nii.gz')

    pib_half_life = 1221.84
    path_to_output = args.output_path
    path_to_moco_pet = os.path.join(path_to_output, "moco.nii.gz")
    # Step 1: Run Motion Correction on 4DPET image
    if not os.path.exists(path_to_moco_pet):
        moco_4d_pet, params, framewise_displacement = pp.preproc.motion_corr.motion_corr(
            input_image_4d_path=path_to_pet,
            motion_target_option=(0, 600),
            out_image_path=path_to_moco_pet,
            verbose=args.verbose,
            half_life=pib_half_life)
    else:
        print(f'moco.nii.gz already exists in output dir {path_to_output}. Continuing...')

    # Step 2: Register Moco'd PET to MRI space
    path_to_registered_pet = os.path.join(path_to_output, "registered_pet.nii.gz")

    if not os.path.exists(path_to_registered_pet):
        registered_pet = pp.preproc.register.register_pet(input_reg_image_path=path_to_moco_pet,
                                                          reference_image_path=path_to_mri,
                                                          motion_target_option=(0, 600),
                                                          out_image_path=path_to_registered_pet,
                                                          verbose=args.verbose,
                                                          half_life=pib_half_life)
    else:
        print(f'registered_pet.nii.gz already exists in output dir {path_to_output}. Continuing...')

    # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input
    if not os.path.exists(os.path.join(path_to_output, 'tacs')):
        os.mkdir(os.path.join(path_to_output, 'tacs'))
        if args.verbose:
            print(f'Creating \"tacs\" directory inside output dir {path_to_output}')

    path_to_tacs = os.path.join(path_to_output, 'tacs')

    # TODO: Ensure that segmentation has the same image dimensions as MRI space
    try:
        tacs = pp.preproc.image_operations_4d.write_tacs(input_image_4d_path=path_to_registered_pet,
                                                         label_map_path=path_to_dseg,
                                                         segmentation_image_path=path_to_segmentation,
                                                         out_tac_dir=path_to_tacs,
                                                         verbose=args.verbose)
    except ValueError as e:
        print(e)

    # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region
    path_to_weighted_sum = os.path.join(path_to_output, 'wss.nii.gz')
    pp.preproc.image_operations_4d.weighted_series_sum(input_image_4d_path=path_to_registered_pet,
                                                       out_image_path=path_to_weighted_sum,
                                                       half_life=pib_half_life,
                                                       verbose=args.verbose)

    path_to_suvr = os.path.join(path_to_output, 'suvr.nii.gz')
    pp.preproc.image_operations_4d.suvr(input_image_path=path_to_weighted_sum,
                                        segmentation_image_path=path_to_segmentation,
                                        ref_region=8,  # This is just the right cerebellar cortex. Could also be 47
                                        # or, ideally, combine the two.
                                        out_image_path=path_to_suvr,
                                        verbose=args.verbose)

    # Step 5: Perform Partial Volume Correction on the SUVR image
    # TODO: Figure out how to access this petpvc docker image
    return


if __name__ == "__main__":
    main()
