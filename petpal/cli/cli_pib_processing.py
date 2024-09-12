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
    * MRI and PET input files are assumed to have ONLY sub-{}, (optionally) ses-{}, and a suffix (e.g. 'pet'). No
        run-{}, desc-{}, etc... are allowed currently.



Example:
    .. code-block:: bash
        petpal-pib-proc -i bids_dir/ -l labelmap.tsv -v

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
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def _parse_command_line_args() -> argparse.Namespace:
    """
    Retrieves arguments and flags from stdin using argparse.ArgumentParser().

    Returns:
        argsparse.Namespace: "args" object with key-value pairs for each argument passed from stdin.
    """
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input_bids', required=True, help='Path to the top-level BIDS directory')
    parser.add_argument('-l', '--label_map', required=True, help='Path to .tsv file containing segmentation LUT info')
    parser.add_argument('-v', '--verbose', help='Display more information while running (up to -vv for debug mode)',
                        action='count', default=0)
    parser.add_argument('-n', '--dry_run', help='Do not perform any computations - Useful for sanity checks; '
                                                'equivalent to using all --skip-{command} flags',
                        action='store_true')
    parser.add_argument('-t', '--threads',
                        help='Number of threads to use for parallel processing; no parallel processing occurs if -t/--threads not specified',
                        default=1, type=int)
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
                                   skip_suvr: bool = False,) -> None:
    """
    Processes one subject-session pair using an instance of the :class:`petpal.preproc.preproc.PreProc` class.

    Processing follows these steps:
        1. Motion Correct PET
        2. Register Motion-Corrected (AKA "MoCo'd") PET to MRI
        3. Extract TACs from each Freesurfer ROI
        4. Generate SUVR Parametric Image

    """

    # Adjust progress bar size to account for skipped steps
    num_steps = (skip_motion_correct, skip_register, skip_tacs, skip_suvr).count(False)
    if num_steps > 0:
        progress_bar = tqdm(total=num_steps, desc=f'{preproc_instance.output_filename_prefix}')

    with logging_redirect_tqdm():

        # Step 1: Run Motion Correction on 4DPET image
        if not skip_motion_correct:
            logger.debug(f'Performing Motion Correction on {preproc_instance.preproc_props["FilePathMocoInp"]}')
            preproc_instance.run_preproc(method_name='thresh_crop', modality='pet')
            try:
                preproc_instance.run_preproc(method_name='motion_corr', modality='pet')
            except KeyError as e:
                logger.error(
                    f'Error during motion correction on pet image {preproc_instance.preproc_props["FilePathMocoInp"]}:'
                    f'\n{e}\n'
                    f'Skipping this subject-session and continuing...\n')
                return
            progress_bar.update(1)

        # Step 2: Register MoCo'd PET to MRI space
        if not skip_register:
            logger.debug(
                f'Registering {preproc_instance.preproc_props["FilePathRegInp"]} to '
                f'{preproc_instance.preproc_props["FilePathAnat"]} space')
            preproc_instance.run_preproc(method_name='register_pet', modality='pet')
            progress_bar.update(1)

        # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input

        if not skip_tacs:
            logger.debug(
                f'Extracting TACs from registered PET using segmentation '
                f'{preproc_instance.preproc_props["FilePathSeg"]}')
            try:
                preproc_instance.run_preproc(method_name='write_tacs')
            except ValueError:
                logger.info(
                    f'Segmentation is not in anatomical space. Resampling segmentation '
                    f'{preproc_instance.preproc_props["FilePathSeg"]} to anatomical space')
                preproc_instance.run_preproc(method_name='resample_segmentation')
                preproc_instance.run_preproc(method_name='write_tacs')
            progress_bar.update(1)

        # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region

        if not skip_suvr:
            logger.debug(
                f'Starting SUVR Process for Registered and Motion-Corrected copy of '
                f'{preproc_instance.preproc_props["FilePathRegInp"]}')
            preproc_instance.run_preproc(method_name='weighted_series_sum', modality='pet')
            preproc_instance.run_preproc(method_name='suvr')
            progress_bar.update(1)

    return


def main():
    """
    PiB Pipeline Command Line Interface.
    """

    # Unpack command line arguments
    args = _parse_command_line_args()

    bids_dir = os.path.abspath(args.input_bids)
    label_map_path = os.path.abspath(args.label_map)
    dry_run = args.dry_run
    verbose = args.verbose
    skip_motion_correct = args.skip_motion_correct
    skip_register = args.skip_register
    skip_tacs = args.skip_tacs
    skip_suvr = args.skip_suvr
    threads = args.threads
    if dry_run:
        skip_motion_correct, skip_register, skip_tacs, skip_suvr = True, True, True, True

    # Python logging levels are ints (0, 10, 20, 30, 40, 50) with 10 being debug, 20 being info, etc...
    logger.setLevel(level=logging.WARNING - (10 * verbose)) if (verbose < 3) else logger.setLevel(level=logging.DEBUG)

    if not (os.path.exists(args.input_bids)):
        raise FileNotFoundError(f'No such directory {args.input_bids}')

    if not os.path.exists(args.label_map):
        raise FileNotFoundError(f'No such file {args.label_map}')

    half_life = 1224  # C-11 Half-Life TODO: Make a shared global dict with Half-Lives of common Radiotracers

    # Parse through BIDS directory to define input and output paths

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
                    'FilePathCropInput':
                        glob.glob(f'{session_dir.path}/pet/sub-{subject_id}_ses-{session_id}_pet.nii.gz')[0],
                    'FilePathAnat': glob.glob(f'{session_dir.path}/anat/sub-{subject_id}_ses-{session_id}_T1w.nii.gz')[
                        0],
                    'FilePathSeg':
                        glob.glob(
                            f'{bids_dir}/derivatives/freesurfer/sub-{subject_id}/ses-{session_id}/sub-{subject_id}_ses-{session_id}_aparc+aseg.nii.gz')[
                            0]
                }
                output = f'{bids_dir}/derivatives/petpal/sub-{subject_id}/ses-{session_id}'
                subject = petpal.preproc.preproc.PreProc(output_directory=output,
                                                         output_filename_prefix=prefix)
                subject.update_props(sub_ses_props)
                subject_inputs.append(subject)

        else:
            prefix = f'sub-{subject_id}'
            sub_props = {
                'FilePathCropInput': glob.glob(f'{subject_dir.path}/pet/sub-{subject_id}_pet.nii.gz')[0],
                'FilePathAnat': glob.glob(f'{subject_dir.path}/anat/sub-{subject_id}_T1w.nii.gz')[0],
                'FilePathSeg':
                    glob.glob(f'{bids_dir}/derivatives/freesurfer/sub-{subject_id}/sub-{subject_id}_aparc+aseg.nii.gz')[
                        0],
            }
            output = os.path.join(bids_dir, 'derivatives/petpal', f'sub-{subject_id}')
            subject = petpal.preproc.preproc.PreProc(output_directory=output,
                                                     output_filename_prefix=prefix)
            subject.update_props(sub_props)
            subject_inputs.append(subject)

    logger.debug(f'Found {len(subject_inputs)} subject-session pairs.')

    for subject_info in subject_inputs:
        # Update props with everything necessary that's not found in BIDS input directory.
        properties = {
            'FilePathWSSInput': subject_info._generate_outfile_path(method_short='reg', modality='pet'),
            'FilePathMocoInp': subject_info._generate_outfile_path(method_short='threshcropped', modality='pet'),
            'HalfLife': half_life,
            'CropThreshold': .01,
            'Verbose': True if (verbose > 0) else False,
            'MotionTarget': (0, 600),  # Use the summed first 10 minutes as motion target
            'FilePathRegInp': subject_info._generate_outfile_path(method_short='moco', modality='pet'),
            'FilePathTACInput': subject_info._generate_outfile_path(method_short='reg', modality='pet'),
            'FilePathLabelMap': label_map_path,
            'TimeFrameKeyword': 'FrameTimesStart',
            'FilePathSUVRInput': subject_info._generate_outfile_path(method_short='wss', modality='pet'),
            'RefRegion': 8,
        }
        subject_info.update_props(properties)

    if threads != 1:
        with Pool(threads) as p:
            p.starmap(process_single_subject_session,
                      map(lambda subject_obj: (subject_obj, skip_motion_correct, skip_register, skip_tacs, skip_suvr),
                          subject_inputs))
    else:
        for subject_info in subject_inputs:
            process_single_subject_session(subject_info, skip_motion_correct, skip_register, skip_tacs, skip_suvr,)


if __name__ == "__main__":
    main()
