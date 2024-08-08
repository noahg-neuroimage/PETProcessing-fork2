""" Provides overall workflow for example PiB dataset, from 4D-PET to SUVR Parametric Images """
import fnmatch
import glob
import os
import argparse
import re
from src.petpal.preproc import segmentation_tools

""" Argparse Configuration """


def options():
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: Ideally, this script can run on an entire BIDS project, correctly finding the relevant data.
    parser.add_argument('-ib', '--input_bids', required=True, help='Path to the top-level BIDS directory')

    # parser.add_argument('-ip', '--input_pet', required=True, help='Path to the input PET image (.nii.gz)')
    # parser.add_argument('-o', '--output_path', required=True,
    #                     help='Folder where all output and intermediate data will be stored')
    # parser.add_argument('-im', '--input_mri', required=True, help='Path to the input mri image')
    # parser.add_argument('-is', '--input_segmentation', required=True, help='Path to segmentation file (.nii.gz) '
    #                                                                        'containing all ROIs')
    parser.add_argument('-d', '--input_dseg', required=True, help='Path to dseg.tsv file containing LUT info')
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')
    parser.add_argument('-n', '--dry_run', help='Do not perform any computations - Used for testing',
                        action='store_true')
    args = parser.parse_args()

    return args


def process_single_subject(path_to_output: str,
                           path_to_segmentation: str,
                           path_to_pet: str,
                           path_to_mri: str,
                           path_to_dseg: str,
                           verbose: bool,
                           dry_run: bool,
                           half_life: float) -> None:
    """ Process one subject using the following steps:
        1. Motion Correct PET
        2. Register MoCo'd PET to MRI
        3. Extract TACs from each Freesurfer ROI
        4. Generate SUVR Parametric Image
        5. Perform Partial Volume Correction (PVC) on SUVR Image

    """
    # Step 1: Run Motion Correction on 4DPET image
    if verbose:
        print(f'Starting Motion Correction Process for {path_to_pet}')
    if not dry_run:
        path_to_moco_pet = os.path.join(path_to_output, "moco.nii.gz")

        if not os.path.exists(path_to_moco_pet):
            try:
                moco_4d_pet, params, _ = src.petpal.preproc.motion_corr.motion_corr(
                    input_image_4d_path=path_to_pet,
                    motion_target_option=(0, 600),
                    out_image_path=path_to_moco_pet,
                    verbose=verbose,
                    half_life=half_life)
            except KeyError as e:
                response = ''
                while response not in ('y', 'n'):
                    response = input(f'Error during motion correction on pet image {path_to_pet}: {e}\n'
                                     f'Do you want to skip this subject and continue? (y/n)\n').strip()
                if response == 'y':
                    print(f'Aborting Processing Subject with pet image {path_to_pet}...')
                    return
        else:
            print(f'moco.nii.gz already exists in output dir {path_to_output}. Continuing...')

    # Step 2: Register MoCo'd PET to MRI space
    if verbose:
        print(f'Starting Registration Process for motion-corrected {path_to_pet} to {path_to_mri}')
    if not dry_run:
        path_to_registered_pet = os.path.join(path_to_output, "registered_pet.nii.gz")

        if not os.path.exists(path_to_registered_pet):
            registered_pet = src.petpal.preproc.register.register_pet(input_reg_image_path=path_to_moco_pet,
                                                                      reference_image_path=path_to_mri,
                                                                      motion_target_option=(0, 600),
                                                                      out_image_path=path_to_registered_pet,
                                                                      verbose=verbose,
                                                                      half_life=half_life)
        else:
            print(f'registered_pet.nii.gz already exists in output dir {path_to_output}. Continuing...')

    # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input
    if verbose:
        print(f'Starting TAC Extraction Process for Registered PET using segmentation {path_to_segmentation}')
    if not dry_run:
        if not os.path.exists(os.path.join(path_to_output, 'tacs')):
            os.mkdir(os.path.join(path_to_output, 'tacs'))
            if verbose:
                print(f'Creating \"tacs\" directory inside output dir {path_to_output}')

            path_to_tacs = os.path.join(path_to_output, 'tacs')

            # TODO: Ensure that segmentation has the same image dimensions as MRI space
            try:
                tacs = src.petpal.preproc.image_operations_4d.write_tacs(input_image_4d_path=path_to_registered_pet,
                                                                         label_map_path=path_to_dseg,
                                                                         segmentation_image_path=path_to_segmentation,
                                                                         out_tac_dir=path_to_tacs,
                                                                         verbose=verbose)
            except ValueError as e:
                print(e)
                print(f'Resampling Segmentation {path_to_segmentation} to Registered PET Space')
                segmentation_tools.resample_segmentation(input_image_4d_path=path_to_registered_pet,
                                                         segmentation_image_path=path_to_segmentation,
                                                         out_seg_path=os.path.join(path_to_output,
                                                                                   'aparc+aseg_resampled.nii.gz'),
                                                         verbose=verbose)
                path_to_segmentation = os.path.join(path_to_output, 'aparc+aseg_resampled.nii.gz')
                tacs = src.petpal.preproc.image_operations_4d.write_tacs(input_image_4d_path=path_to_registered_pet,
                                                                         label_map_path=path_to_dseg,
                                                                         segmentation_image_path=path_to_segmentation,
                                                                         out_tac_dir=path_to_tacs,
                                                                         verbose=verbose)
        else:  # tacs directory already exists
            print(f'tacs/ dir already exists in output dir {path_to_output}. Continuing...')

    # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region
    if verbose:
        print(f'Starting SUVR Process for Registered and Motion-Corrected copy of {path_to_pet}')
    if not dry_run:
        if not os.path.exists(os.path.join(path_to_output, 'wss.nii.gz')):
            path_to_weighted_sum = os.path.join(path_to_output, 'wss.nii.gz')
            src.petpal.preproc.image_operations_4d.weighted_series_sum(input_image_4d_path=path_to_registered_pet,
                                                                       out_image_path=path_to_weighted_sum,
                                                                       half_life=half_life,
                                                                       verbose=verbose)
        else:
            print(f'weighted_series_sum file already exists in output dir {path_to_output}. Continuing...')

        if not os.path.exists(os.path.join(path_to_output, 'suvr.nii.gz')):
            path_to_suvr = os.path.join(path_to_output, 'suvr.nii.gz')
            src.petpal.preproc.image_operations_4d.suvr(input_image_path=path_to_weighted_sum,
                                                        segmentation_image_path=path_to_segmentation,
                                                        ref_region=8,  # This is just the right cerebellar cortex. Could also
                                                        # be 47 or, ideally, combine the two into one mask label.
                                                        out_image_path=path_to_suvr,
                                                        verbose=verbose)
        else:
            print(f'suvr image already exists in output dir {path_to_output}. Continuing...')

    # Step 5: Perform Partial Volume Correction on the SUVR image
    # TODO: Figure out how to access this petpvc docker image
    return


def main():
    # Unpack command line arguments
    args = options()
    # if os.path.exists(args.input_pet):
    #     path_to_pet = args.input_pet
    # else:
    #     raise FileNotFoundError(f'No PET file found at {args.input_pet}')
    #
    # if os.path.exists(args.input_mri):
    #     path_to_mri = args.input_mri
    # else:
    #     raise FileNotFoundError(f'No MRI file found at {args.input_mri}')
    #
    # if os.path.exists(args.input_segmentation):
    #     path_to_segmentation = args.input_segmentation
    # else:
    #     raise FileNotFoundError(f'No segmentation file found at {args.input_segmentation}')
    #
    if os.path.exists(args.input_dseg):
        path_to_dseg = args.input_dseg
    else:
        raise FileNotFoundError(f'No dseg.tsv file found at {args.input_dseg}')
    #
    # if not (os.path.exists(args.output_path)):
    #     os.mkdir(args.output_path)
    #     if args.verbose:
    #         print(f"Creating output directory at {args.output_path}")

    if not (os.path.exists(args.input_bids)):
        raise FileNotFoundError(f'No such directory {args.input_bids}')

    dry_run = args.dry_run
    verbose = args.verbose
    dseg_path = args.input_dseg

    # Parse through BIDS directory to define input and output paths
    bids_dir = os.path.abspath(args.input_bids)
    subjects_pattern = re.compile(r'sub-(\d{3})')
    # Create list (of dicts) for each subject's inputs' paths to be stored
    subject_inputs = []

    # TODO: Intelligently deal with multiple sessions or multiple images found for a given subject
    subject_dirs = fnmatch.filter([directory.name for directory in os.scandir(bids_dir)], 'sub-[0-9][0-9][0-9]')
    for subject_dir in subject_dirs:
        match = subjects_pattern.match(subject_dir)
        if match:
            subject_id = match.group(1)
            if verbose:
                print(f'Gathering inputs for subject {subject_id}')
            subject_dict = {
                'pet': glob.glob(f'{bids_dir}/{subject_dir}/**/pet/*_pet.nii.gz',
                                 recursive=True)[0],  # TODO: Don't assume only one image for this or the following two
                'mri': glob.glob(f'{bids_dir}/{subject_dir}/**/anat/*_T1w.nii.gz',
                                 recursive=True)[0],
                'segmentation': glob.glob(f'{bids_dir}/derivatives/sub-{subject_id}/aparc+aseg.nii.gz',
                                          recursive=True)[0],
                'output': os.path.join(bids_dir, 'derivatives', f'sub-{subject_id}')
            }
            subject_inputs.append(subject_dict)

    response = ''
    while response not in ('y', 'n'):
        response = input(
            f'Found {len(subject_inputs)} subject directories. Do you want to begin processing? (y/n)\n').strip()
    if response == 'n':
        return

    for subject_info in subject_inputs:
        process_single_subject(path_to_output=subject_info['output'],
                               path_to_segmentation=subject_info['segmentation'],
                               path_to_pet=subject_info['pet'],
                               path_to_mri=subject_info['mri'],
                               path_to_dseg=dseg_path,
                               verbose=verbose,
                               dry_run=dry_run,
                               half_life=1221.84)


if __name__ == "__main__":
    main()
