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
import petpal.preproc

logger = logging.getLogger(__name__)


def _parse_command_line_args() -> argparse.Namespace:
    """
    Retrieves arguments and flags from stdin using argparse.ArgumentParser().

    Returns:
        argsparse.Namespace: "args" object with key-value pairs for each argument passed from stdin.
    """
    parser = argparse.ArgumentParser(description='Script to process PiB PET data from 4D-PET to SUVR Parametric Image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--pet', required=True, help='Path to input 4D PET File')
    parser.add_argument('-m', '--mri', required=True, help='Path to input T1 MRI File')
    parser.add_argument('-l', '--label_map', required=True, help='Path to .tsv file containing segmentation LUT info')
    parser.add_argument('-s', '--seg', required=True, help='Path to segmentation file (.nii.gz)')
    parser.add_argument('-o', '--outdir', required=True,
                        help="Directory to store all outputs. Directory will be created if it doesn't exist")
    parser.add_argument('-v', '--verbose', help='Display more information while running', action='store_true')
    parser.add_argument('--skip-moco', help='Do not perform motion correction', action='store_true')
    parser.add_argument('--skip-register', help='Do not perform registration', action='store_true')
    parser.add_argument('--skip-tacs', help='Do not generate TACs', action='store_true')
    parser.add_argument('--skip-suvr', help='Do not generate SUVR images', action='store_true')
    args = parser.parse_args()

    return args


def process_single_subject_session(preproc_instance: petpal.preproc.preproc.PreProc,
                                   skip_motion_correct: bool = False,
                                   skip_register: bool = False,
                                   skip_tacs: bool = False,
                                   skip_suvr: bool = False, ) -> None:
    """
    Processes one subject-session pair using an instance of the :class:`petpal.preproc.preproc.PreProc` class.

    Processing follows these steps:
        1. Motion Correct PET
        2. Register Motion-Corrected (AKA "MoCo'd") PET to MRI
        3. Extract TACs from each Freesurfer ROI
        4. Generate SUVR Parametric Image

    """

    # Step 1: Run Motion Correction on 4DPET image
    if not skip_motion_correct:
        logger.info(f'Performing Motion Correction on {preproc_instance.preproc_props["FilePathMocoInp"]}...')
        preproc_instance.run_preproc(method_name='thresh_crop', modality='pet')
        preproc_instance.run_preproc(method_name='motion_corr', modality='pet')

    # Step 2: Register MoCo'd PET to MRI space
    if not skip_register:
        logger.info(
            f'Registering {preproc_instance.preproc_props["FilePathRegInp"]} to '
            f'{preproc_instance.preproc_props["FilePathAnat"]} space...')
        preproc_instance.run_preproc(method_name='register_pet', modality='pet')

    # Step 3: Extract TACs from each Brain ROI using Freesurfer Segmentation Input
    if not skip_tacs:
        logger.info(
            f'Extracting TACs from registered PET using segmentation '
            f'{preproc_instance.preproc_props["FilePathSeg"]}...')
        try:
            preproc_instance.run_preproc(method_name='write_tacs', modality='pet')
        except ValueError:
            logger.info(
                f'Segmentation is not in anatomical space. Resampling segmentation '
                f'{preproc_instance.preproc_props["FilePathSeg"]} to anatomical space...')
            preproc_instance.run_preproc(method_name='resample_segmentation')
            preproc_instance.run_preproc(method_name='write_tacs')

    # Step 4: Compute SUVR with Cerebellar Gray Matter as Reference Region
    if not skip_suvr:
        logger.info(
            f'Starting SUVR Process for Registered and Motion-Corrected copy of '
            f'{preproc_instance.preproc_props["FilePathRegInp"]}...')
        preproc_instance.run_preproc(method_name='weighted_series_sum', modality='pet')
        preproc_instance.run_preproc(method_name='suvr')

    return


def main():
    """
    PiB Pipeline Command Line Interface.
    """

    # Unpack command line arguments
    args = _parse_command_line_args()

    label_map_path = os.path.abspath(args.label_map)
    path_to_pet = os.path.abspath(args.pet)
    path_to_mri = os.path.abspath(args.mri)
    path_to_seg = os.path.abspath(args.segmentation)
    outdir = os.path.abspath(args.outdir)

    verbose = args.verbose
    skip_motion_correct = args.skip_motion_correct
    skip_register = args.skip_register
    skip_tacs = args.skip_tacs
    skip_suvr = args.skip_suvr

    logger.setLevel(level=logging.INFO if verbose else logging.WARNING)

    if not os.path.exists(args.label_map):
        raise FileNotFoundError(f'No such file {args.label_map}')

    if not os.path.exists(path_to_mri):
        raise FileNotFoundError(f'No such file {path_to_mri}')

    if not os.path.exists(path_to_seg):
        raise FileNotFoundError(f'No such file {path_to_seg}')

    if not os.path.exists(path_to_pet):
        raise FileNotFoundError(f'No such file {path_to_pet}')

    # TODO: Make a shared global dict with Half-Lives of common Radiotracers
    half_life = 1221.66  # C-11 Half-Life According to NIST

    pattern = re.compile(r'sub-(\d+)')
    prefix = re.match(pattern, path_to_pet.split(os.sep)[-1])[0] # There will only be one match if BIDS-compliant

    subject = petpal.preproc.preproc.PreProc(output_directory=outdir,
                                             output_filename_prefix=prefix)

    # Update props with everything necessary.
    properties = {
        'FilePathCropInput': path_to_pet,
        'FilePathAnat': path_to_mri,
        'FilePathSeg': path_to_seg,
        'FilePathWSSInput': subject._generate_outfile_path(method_short='reg', modality='pet'),
        'FilePathMocoInp': subject._generate_outfile_path(method_short='threshcropped', modality='pet'),
        'HalfLife': half_life,
        'CropThreshold': .01,
        'Verbose': verbose,
        'MotionTarget': (0, 600),  # Use the summed first 10 minutes as motion target
        'FilePathRegInp': subject._generate_outfile_path(method_short='moco', modality='pet'),
        'FilePathTACInput': subject._generate_outfile_path(method_short='reg', modality='pet'),
        'FilePathLabelMap': label_map_path,
        'TimeFrameKeyword': 'FrameTimesStart',
        'FilePathSUVRInput': subject._generate_outfile_path(method_short='wss', modality='pet'),
        'RefRegion': 8,  # TODO: Make a function for combining ROIs
        'StartTimeWSS': 1800,
        'EndTimeWSS': 3600
    }
    subject.update_props(properties)

    process_single_subject_session(subject, skip_motion_correct, skip_register, skip_tacs, skip_suvr)


if __name__ == "__main__":
    main()
