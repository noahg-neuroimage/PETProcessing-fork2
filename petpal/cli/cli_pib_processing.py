"""
Command-line interface (CLI) for processing PET images collected using the PiB tracer.

This module provides a CLI to generate TACs and SUVR parametric images using input PET and MRI data from a
BIDS-compliant directory. It assumes all input PET data was collected using Pittsburgh Compound B (PiB) as the
radiotracer.

The user must provide either ALL of the following:
    * Path to the top-level of a BIDS-compliant directory
    * Path to label map .tsv file

This script uses one instance of the class :class:`petpal.preproc.preproc.PreProc` for each subject-session pair found
in the BIDS directory.

Assumptions regarding input data:
    * If generating tacs or SUVR, segmentation is assumed to exist at
        "{bids_dir}/derivatives/freesurfer/{prefix}-aparc+aseg.nii.gz"
        * In this case, "prefix" should exactly match a corresponding prefix in your bids_dir (e.g. "sub-01")



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
        argsparse.Namespace: "args" object with key-value pairs for each argument passed from stdin.
    """
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-ib', '--input_bids', required=True, help='Path to the top-level BIDS directory')
    parser.add_argument('-d', '--input_dseg', required=True, help='Path to dseg.tsv file containing LUT info')
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')
    parser.add_argument('-n', '--dry_run', help='Do not perform any computations - Useful for sanity checks; '
                                                'equivalent to using all --skip-{command} flags',
                        action='store_true')
    parser.add_argument('--skip-motion-correct', help='Do not perform motion correction', action='store_true')
    parser.add_argument('--skip-register', help='Do not perform registration', action='store_true')
    parser.add_argument('--skip-tacs', help='Do not generate TACs', action='store_true')
    parser.add_argument('--skip-suvr', help='Do not generate SUVR images', action='store_true')
    args = parser.parse_args()

    return args


def process_single_subject_session(preproc_instance: petpal.preproc.preproc.PreProc,
                                   skip_motion_correct: bool = False,
                                   skip_register: bool = False,
                                   skip_tacs: bool = False,
                                   skip_suvr: bool = False) -> None:
    """
    Processes one subject-session pair using an instance of the :class:`petpal.preproc.preproc.PreProc` class.

    Processing follows these steps:
        1. Motion Correct PET
        2. Register Motion-Corrected (AKA "MoCo'd") PET to MRI
        3. Extract TACs from each Freesurfer ROI
        4. Generate SUVR Parametric Image

    """

    progress_bar = tqdm(total=4)
    with logging_redirect_tqdm():

        # Step 1: Run Motion Correction on 4DPET image
        logger.debug(f'Starting Motion Correction Process for {preproc_instance.preproc_props["FilePathMocoInp"]}')
        if not skip_motion_correct:
            try:
                preproc_instance.run_preproc(method_name='motion_corr')
            except KeyError as e:
                logger.error(
                    f'Error during motion correction on pet image {preproc_instance.preproc_props["FilePathMocoInp"]}:'
                    f'\n{e}\n'
                    f'Skipping this subject-session and continuing...\n')
                return

        # Step 2: Register MoCo'd PET to MRI space
        progress_bar.update(1)
        logger.debug(
            f'Starting Registration Process for motion-corrected PET at '
            f'{preproc_instance.preproc_props["FilePathRegInp"]} to {preproc_instance.preproc_props["FilePathAnat"]}')
        if not skip_register:
            preproc_instance.run_preproc(method_name='register_pet')

        # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input
        progress_bar.update(1)
        logger.debug(
            f'Starting TAC Extraction Process for Registered PET using segmentation '
            f'{preproc_instance.preproc_props["FilePathSeg"]}')
        if not skip_tacs:
            try:
                preproc_instance.run_preproc(method_name='write_tacs')
            except ValueError:
                logger.info(
                    f'Segmentation is not in anatomical space. Resampling segmentation '
                    f'{preproc_instance.preproc_props["FilePathSeg"]} to anatomical space')
                preproc_instance.run_preproc(method_name='resample_segmentation')
                preproc_instance.run_preproc(method_name='write_tacs')


        # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region
        progress_bar.update(1)
        logger.debug(
            f'Starting SUVR Process for Registered and Motion-Corrected copy of '
            f'{preproc_instance.preproc_props["FilePathRegInp"]}')
        if not skip_suvr:
            preproc_instance.run_preproc(method_name='weighted_series_sum')
            preproc_instance.run_preproc(method_name='suvr')


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
                prefix = f'sub-{subject_id}_ses-{session_id}'
                sub_ses_props = {
                    'FilePathMocoInp': glob.glob(f'{session_dir.path}/pet/sub-{subject_id}_ses-{session_id}_pet.nii.gz')[0],
                    'FilePathAnat': glob.glob(f'{session_dir.path}/anat/sub-{subject_id}_ses-{session_id}_T1w.nii.gz')[0],
                    'FilePathSeg':
                        glob.glob(f'{bids_dir}/derivatives/freesurfer/sub-{subject_id}/ses-{session_id}/sub-{subject_id}_ses-{session_id}_aparc+aseg.nii.gz')[0]
                }
                output = f'{bids_dir}/derivatives/petpal/sub-{subject_id}/ses-{session_id}'
                subject = petpal.preproc.preproc.PreProc(output_directory=output,
                                                         output_filename_prefix=prefix)
                subject.update_props(sub_ses_props)
                subject_inputs.append(subject)

        else:
            prefix = f'sub-{subject_id}'
            sub_props = {
                'FilePathMocoInp': glob.glob(f'{subject_dir.path}/pet/sub-{subject_id}_pet.nii.gz')[0],
                'FilePathAnat': glob.glob(f'{subject_dir.path}/anat/sub-{subject_id}_T1w.nii.gz')[0],
                'FilePathSeg':
                    glob.glob(f'{bids_dir}/derivatives/freesurfer/sub-{subject_id}/sub-{subject_id}_aparc+aseg.nii.gz')[0],
            }
            output = os.path.join(bids_dir, 'derivatives/petpal', f'sub-{subject_id}')
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
