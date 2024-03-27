"""
CLI - Preprocessing
------------------------

The `cli_preproc` module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * Path to PET input data in NIfTI format. This can be source data, or with some preprocessing
        such as registration or motion correction, depending on the chosen operation.
    * Directory to which the output is written.
    * The name of the subject being processed, for the purpose of naming output files.
    * 3D imaging data, such as anatomical, segmentation, or PET sum, depending on the desired
        preprocessing operation.
    * Additional information needed for preprocessing, such as color table or half-life.
    * The operation to be performed on input data. Options: `weighted_sum`, `motion_correct`,
        `register`, or `write_tacs`.

Example:
    .. code-block:: bash

        pet-cli-preproc --pet /path/to/pet.nii --anatomical /path/to/mri.nii --pet_reference /path/to/pet_sum.nii --out_dir /path/to/output --operation register

See Also:
    * :mod:`pet_cli.image_operations_4d` - module used to preprocess PET imaging data.

"""
import os
import argparse
from . import image_operations_4d


def generate_args():
    parser = argparse.ArgumentParser(
        prog='PET Preprocessing',
        description='Command line interface for running PET preprocessing steps.',
        epilog='Example: pet-cli-preproc '
            '--pet /path/to/pet.nii '
            '--anatomical /path/to/mri.nii '
            '--pet_reference /path/to/pet_sum.nii '
            '--out_dir /path/to/output '
            '--operation register'
    )
    io_grp = parser.add_argument_group('I/O')
    io_grp.add_argument('--pet',required=True,help='Path to PET series')
    io_grp.add_argument('--anatomical',required=False,help='Path to 3D anatomical image')
    io_grp.add_argument('--segmentation',required=False,help='Path to segmentation image\
        in anatomical space')
    io_grp.add_argument('--pet_reference',required=False,help='Path to reference image\
        for motion correction, if not weighted_sum.')
    io_grp.add_argument('--color_table_path',required=False,help='Path to color table')
    io_grp.add_argument('--half_life',required=False,help='Half life of radioisotope in seconds.',
        type=float)
    io_grp.add_argument('--out_dir',required=True,help='Directory to write results to')
    io_grp.add_argument('--subject_id',required=False,help='Subject ID to name files with',
        default='sub')

    ops_grp = parser.add_argument_group('Operations')
    ops_grp.add_argument('--operation',required=True,help='Preprocessing operation to perform',
        choices=['weighted_sum','motion_correct','register','write_tacs'])

    verb_group = parser.add_argument_group('Additional information')
    verb_group.add_argument('-v', '--verbose', action='store_true',
        help='Print processing information during computation.', required=False)

    args = parser.parse_args()
    return args


def create_dirs(main_dir,ops_dir,sub_id,ops_ext):
    inter_dir = os.path.join(main_dir,ops_dir)
    os.makedirs(inter_dir,exist_ok=True)
    image_write = os.path.join(main_dir,ops_dir,f'{sub_id}_{ops_ext}.nii.gz')
    return image_write


def check_ref(args):
    if args.pet_reference is not None:
        ref_image = args.pet_reference
    else:
        ref_image = create_dirs(args.out_dir,'sum_image',args.subject_id,'sum')
    return ref_image


def main():
    args = generate_args()

    if args.operation == 'weighted_sum':
        image_write = create_dirs(args.out_dir,'sum_image',args.subject_id,'sum')
        image_operations_4d.weighted_series_sum(
            input_image_4d_path=args.pet,
            out_image_path=image_write,
            half_life=args.half_life,
            verbose=args.verbose
        )


    if args.operation == 'motion_correct':
        image_write = create_dirs(args.out_dir,'motion-correction',args.subject_id,'moco')
        ref_image = check_ref(args)
        image_operations_4d.motion_correction(
            input_image_4d_path=args.pet,
            reference_image_path=ref_image,
            out_image_path=image_write,
            verbose=args.verbose
        )


    if args.operation == 'register':
        image_write = create_dirs(args.out_dir,'registration',args.subject_id,'reg')
        ref_image = check_ref(args)
        image_operations_4d.register_pet(
            input_calc_image_path=ref_image,
            input_reg_image_path=args.pet,
            reference_image_path=args.anatomical,
            out_image_path=image_write,
            verbose=args.verbose
        )


    if args.operation == 'write_tacs':
        tac_write = os.path.join(args.out_dir,'tacs')
        os.makedirs(tac_write,exist_ok=True)
        image_write = create_dirs(args.out_dir,'segmentation',args.subject_id,'seg')
        image_operations_4d.resample_segmentation(
            input_image_4d_path=args.pet,
            segmentation_image_path=args.segmentation,
            out_seg_path=image_write,
            verbose=args.verbose
        )
        image_operations_4d.write_tacs(
            input_image_4d_path=args.pet,
            color_table_path=args.color_table_path,
            segmentation_image_path=image_write,
            out_tac_path=tac_write,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
