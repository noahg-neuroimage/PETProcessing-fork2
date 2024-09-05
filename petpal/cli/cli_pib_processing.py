"""
Command-line interface (CLI) for processing PET images collected using the PiB tracer.

This module provides a CLI to generate TACs and SUVR parametric images using input data from a single subject or a
BIDS-compliant directory. It assumes all input PET data was collected using Pittsburgh Compound B (PiB) as the
radiotracer.

The user must provide either ALL of the following:
    * Path to a 4D PET image file
    * Path to a MRI image file
    * Path to a freesurfer segmentation file generated from MRI image data
    * Path to desired output directory
    * Path to label map .tsv file

or:
    * Path to the top-level of a BIDS-compliant directory
    * Path to label map .tsv file

This script uses one instance of the class :class:`petpal.preproc.preproc.PreProc` for each subject-session pair found
in the BIDS directory.

Example:
    .. code-block:: bash

        TODO: update this once changes have been made

See Also:
    :mod:`petpal.preproc` - module for initiating and saving the graphical analysis of PET parametric images.

"""

import argparse
import glob
import logging
import os
import re
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import petpal.preproc

logger = logging.getLogger(__name__)


def parse_command_line_args() -> argparse.Namespace:
    """
    Retrieves arguments and flags from stdin using argparse.ArgumentParser().

    Returns:
        argsparse.Namespace: "args" object with key-value pairs for each argument passed to the script.
    """
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: Add Mutually Exclusive Groups for Using a BIDS directory or Manually Specifying Inputs
    parser.add_argument('-ib', '--input_bids', required=True, help='Path to the top-level BIDS directory')
    parser.add_argument('-d', '--input_dseg', required=True, help='Path to dseg.tsv file containing LUT info')
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')
    parser.add_argument('-n', '--dry_run', help='Do not perform any computations - Useful for sanity checks',
                        action='store_true')
    parser.add_argument('-f', '--force', help='Automatically overwrite any existing output files', action='store_true')
    args = parser.parse_args()

    return args


def process_single_subject_session(preproc_instance: petpal.preproc.preproc.PreProc,
                                   dry_run: bool,
                                   force: bool) -> None:
    """
    Processes one subject-session pair using an instance of the :class:`petpal.preproc.preproc.PreProc` class.

    Processing follows these steps:
        1. Motion Correct PET
        2. Register Motion-Corrected (AKA "MoCo'd") PET to MRI
        3. Extract TACs from each Freesurfer ROI
        4. Generate SUVR Parametric Image
        5. Perform Partial Volume Correction (PVC) on SUVR Image

    """

    # Generate all intermediate filepaths
    path_to_moco_pet = preproc_instance._generate_outfile_path('moco')
    path_to_registered_pet = preproc_instance._generate_outfile_path('reg')
    path_to_tacs = os.path.join(preproc_instance.output_directory, 'tacs')
    path_to_wss = preproc_instance._generate_outfile_path('wss')
    path_to_suvr = preproc_instance._generate_outfile_path('suvr')
    pvc_filename = f'{preproc_instance.output_filename_prefix}_pvc.nii.gz'
    path_to_pvc = os.path.join(preproc_instance.output_directory, pvc_filename)

    pbar = tqdm(total = 5)
    with logging_redirect_tqdm():

        # Step 1: Run Motion Correction on 4DPET image
        logger.debug(f'Starting Motion Correction Process for {preproc_instance.preproc_props["FilePathMocoInp"]}')
        if force or not os.path.exists(path_to_moco_pet):
            try:
                preproc_instance.run_preproc(method_name='motion_corr')
            except KeyError as e:
                print(
                    f'Error during motion correction on pet image {preproc_instance.preproc_props["FilePathMocoInp"]}:\n{e}\n'
                    f'Skipping this subject-session and continuing...\n')
                return
        else:
            logger.info(f'Motion-Corrected PET already exists at {path_to_moco_pet}. Skipping and continuing...')

        pbar.update(1)
        # Step 2: Register MoCo'd PET to MRI space
        logger.debug(
            f'Starting Registration Process for motion-corrected PET at {preproc_instance.preproc_props["FilePathRegInp"]} to {preproc_instance.preproc_props["FilePathAnat"]}')
        if force or not os.path.exists(path_to_registered_pet):
            preproc_instance.run_preproc(method_name='register_pet')
        else:
            logger.info(f'registered_pet.nii.gz already exists at {path_to_registered_pet}. Skipping and continuing...')

        pbar.update(1)
        # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input
        logger.debug(
            f'Starting TAC Extraction Process for Registered PET using segmentation {preproc_instance.preproc_props["FilePathSeg"]}')
        if force or not os.path.exists(path_to_tacs):
            try:
                preproc_instance.run_preproc(method_name='write_tacs')
            except ValueError as e:  # TODO: make this more specific to the different dimensions error
                if force:
                    logger.warning(
                        f'Segmentation is not in anatomical space. Resampling segmentation {preproc_instance.preproc_props["FilePathSeg"]} to anatomical space')
                    preproc_instance.run_preproc(method_name='resample_segmentation')
                else:
                    logger.error(
                        f'Encountered error {e}\nIf you want to automatically resample and overwrite this segmenation, use the -f/--force flag\nAborting subject...')
                    return

                tacs = petpal.preproc.image_operations_4d.write_tacs(input_image_4d_path=path_to_registered_pet,
                                                                     label_map_path=preproc_instance.preproc_props[
                                                                         "FilePathLabelMap"],
                                                                     segmentation_image_path=
                                                                     preproc_instance.preproc_props['FilePathSeg'],
                                                                     out_tac_dir=path_to_tacs,
                                                                     verbose=preproc_instance.preproc_props["Verbose"])
        else:  # tacs directory already exists
            logger.info(f'tacs/ dir already exists in output dir {path_to_tacs}. Continuing...')

        pbar.update(1)
        # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region
        logger.debug(
            f'Starting SUVR Process for Registered and Motion-Corrected copy of {preproc_instance.preproc_props["FilePathRegInp"]}')
        if force or not os.path.exists(path_to_wss):
            preproc_instance.run_preproc(method_name='weighted_series_sum')
        else:
            logger.info(f'weighted_series_sum file already exists at {path_to_wss}. Continuing...')

        if force or not os.path.exists(path_to_suvr):
            petpal.preproc.image_operations_4d.suvr(input_image_path=path_to_wss,
                                                    segmentation_image_path=preproc_instance.preproc_props[
                                                        "FilePathSeg"],
                                                    ref_region=preproc_instance.preproc_props["RefRegion"],
                                                    # TODO: Write a function to combine an arbitrary number of mask label into one.
                                                    out_image_path=path_to_suvr,
                                                    verbose=preproc_instance.preproc_props["Verbose"])
        else:
            logger.info(f'SUVR image already exists at {path_to_suvr}. Continuing...')

        pbar.update(1)
        # Step 5: Perform Partial Volume Correction on the SUVR image

        pvc_runner = petpal.preproc.partial_volume_corrections.PetPvc()
        # pvc_runner.run_petpvc(pet_4d_filepath=path_to_registered_pet,
        #                       output_filepath=path_to_pvc,
        #                       pvc_method="SGTM",
        #                       psf_dimensions=8.0,
        #                       mask_filepath=)

    return


def main():
    """
    PiB Pipeline Command Line Interface.
    """
    # Unpack command line arguments
    args = parse_command_line_args()

    logger.setLevel(level=logging.INFO)
    verbose = args.verbose
    if verbose:
        logger.setLevel(level=logging.DEBUG)


    if not (os.path.exists(args.input_bids)):
        raise FileNotFoundError(f'No such directory {args.input_bids}')

    if os.path.exists(args.input_dseg):
        dseg_path = args.input_dseg
    else:
        raise FileNotFoundError(f'No such file {args.input_dseg}')

    force = args.force
    dry_run = args.dry_run
    half_life = 1224  # C-11 Half-Life TODO: Make a shared global dict with Half-Lives of common Radiotracers

    # Parse through BIDS directory to define input and output paths
    bids_dir = os.path.abspath(args.input_bids)
    subjects_pattern = re.compile(r'sub-(\d+)')
    sessions_pattern = re.compile(r'ses-(\d+)')

    # List to store an instance of PreProc for each subject-session pair found
    subject_inputs = []

    subject_dirs = [directory for directory in os.scandir(bids_dir) if
                    directory.is_dir() and subjects_pattern.match(directory.name)]
    for subject_dir in subject_dirs:
        subject_id = subjects_pattern.match(subject_dir.name).group(1)  # Retrieve subject id
        logger.debug(f'Gathering inputs for subject {subject_id}')
        session_dirs = [directory for directory in os.scandir(subject_dir) if
                        directory.is_dir() and sessions_pattern.match(directory.name)]
        if session_dirs:
            for session_dir in session_dirs:
                session_id = sessions_pattern.match(session_dir.name).group(1)  # Retrieve session id
                logger.debug(f'MocoPaths: {glob.glob(f"{session_dir.path}/pet/*_pet.nii.gz")}')
                sub_ses_props = {
                    'FilePathMocoInp': glob.glob(f'{session_dir.path}/pet/*_pet.nii.gz')[0],
                    'FilePathAnat': glob.glob(f'{session_dir.path}/anat/*_T1w.nii.gz')[0],
                    'FilePathSeg':
                        glob.glob(f'{bids_dir}/derivatives/freesurfer/sub-{subject_id}/aparc+aseg.nii.gz')[0]
                }
                output = f'{bids_dir}/derivatives/petpal/sub-{subject_id}/ses-{session_id}'
                prefix = f'sub-{subject_id}'
                subject = petpal.preproc.preproc.PreProc(output_directory=output,
                                                         output_filename_prefix=prefix)
                subject.update_props(sub_ses_props)
                subject_inputs.append(subject)

        else:
            logger.debug(f'Possible MocoInputs: {glob.glob(os.path.join(subject_dir.path, "pet", "*_pet.nii.gz"))}')
            sub_props = {
                'FilePathMocoInp': glob.glob(f'{subject_dir.path}/pet/*_pet.nii.gz')[0],
                'FilePathAnat': glob.glob(f'{subject_dir.path}/anat/*_T1w.nii.gz')[0],
                'FilePathSeg':
                    glob.glob(f'{bids_dir}/derivatives/freesurfer/sub-{subject_id}/sub-{subject_id}_aparc+aseg.nii.gz')[0],
            }
            output = os.path.join(bids_dir, 'derivatives/petpal', f'sub-{subject_id}')
            prefix = f'sub-{subject_id}'
            subject = petpal.preproc.preproc.PreProc(output_directory=output,
                                                     output_filename_prefix=prefix)
            subject.update_props(sub_props)
            subject_inputs.append(subject)

    logger.info(f'Found {len(subject_inputs)} subject-session pairs.')

    for subject_info in subject_inputs:
        # Update props with everything necessary that's not found in BIDS input directory.
        properties = {
            'FilePathWSSInput': subject_info._generate_outfile_path(method_short='reg'),
            'HalfLife': half_life,
            'Verbose': verbose,
            'MotionTarget': (0, 600),  # Use the summed first 10 minutes as motion target
            'FilePathRegInp': subject_info._generate_outfile_path(method_short='moco'),
            'FilePathTACInput': subject_info._generate_outfile_path(method_short='reg'),
            'FilePathLabelMap': dseg_path,
            'TimeFrameKeyword': 'FrameTimesStart',
            'FilePathSUVRInput': subject_info._generate_outfile_path(method_short='wss'),
            'RefRegion': 8,
        }
        subject_info.update_props(properties)
        process_single_subject_session(subject_info, dry_run=dry_run, force=force)


if __name__ == "__main__":
    main()
