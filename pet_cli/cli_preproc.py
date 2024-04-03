"""
This module provides a Command-line interface (CLI) for preprocessing imaging data to
produce regional PET Time-Activity Curves (TACs) and prepare data for parametric imaging analysis.

The user must provide:
    * The sub-command. Options: 'weighted-sum', 'motion-correct', 'register', or 'write-tacs'.
    * Path to PET input data in NIfTI format. This can be source data, or with some preprocessing
      such as registration or motion correction, depending on the chosen operation.
    * Directory to which the output is written.
    * The name of the subject being processed, for the purpose of naming output files.
    * 3D imaging data, such as anatomical, segmentation, or PET sum, depending on the desired
      preprocessing operation.
    * Additional information needed for preprocessing, such as color table or half-life.
    

Examples:
    * Half-life Weighted Sum:
    
        .. code-block:: bash
    
            pet-cli-preproc weighted-sum --pet /path/to/pet.nii --out-dir /path/to/output --half-life 6600.0
    
    * Image Registration:
    
        .. code-block:: bash
    
            pet-cli-preproc register --pet /path/to/pet.nii --anatomical /path/to/mri.nii --pet-reference /path/to/pet_sum.nii --out-dir /path/to/output
            
    * Motion Correction:
    
        .. code-block:: bash
            
            pet-cli-preproc motion-correct --pet /path/to/pet.nii --pet-reference /path/to/sum.nii --out-dir /path/to/output
            
    * Extracting TACs Using A Mask And Color-Table:
    
        .. code-block:: bash
            
            pet-cli-preproc write-tacs --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --color-table-path /path/to/color_table.json --out-dir /path/to/output

See Also:
    * :mod:`pet_cli.image_operations_4d` - module used to preprocess PET imaging data.

"""
import os
import argparse
from . import image_operations_4d


def _generate_image_path_and_directory(main_dir, ops_dir_name, file_prefix, ops_desc) -> str:
    """
    Generates the full path of an image file based on given parameters and creates the necessary directories.

    This function takes in four arguments: the main directory (main_dir), the operations directory (ops_dir),
    the subject ID (sub_id), and the operations extension (ops_ext). It joins these to generate the full path
    for an image file. The generated directories are created if they do not already exist.

    Args:
        main_dir (str): The main directory path.
        ops_dir_name (str): The operations (ops) directory. This is a directory inside `main_dir`.
        file_prefix (str): The prefix for the file name. Usually sub-XXXX if following BIDS.
        ops_desc (str): The operations (ops) extension to append to the filename.

    Returns:
        str: The full path of the image file with '.nii.gz' extension.

    Side Effects:
        Creates directories denoted by `main_dir`/`ops_dir_name` if they do not exist.

    Example:
        
        .. code-block:: python
        
            _generate_image_path_and_directory('/home/images', 'ops', '123', 'preprocessed')
            # '/home/images/ops/123_desc-preprocessed.nii.gz'
            # Directories '/home/images/ops' are created if they do not exist.
            
    """
    image_dir = os.path.join(main_dir, ops_dir_name)
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(f'{image_dir}', f'{file_prefix}_desc-{ops_desc}.nii.gz')
    return str(image_path)




_PREPROC_EXAMPLES_ = (r"""
Examples:
  - Weighted Sum:
    pet-cli-preproc weighted-sum --pet /path/to/pet.nii --out-dir /path/to/output --half-life 6600.0
  - Registration:
    pet-cli-preproc register --pet /path/to/pet.nii --anatomical /path/to/mri.nii --pet-reference /path/to/pet_sum.nii --out-dir /path/to/output
  - Motion Correction:
    pet-cli-preproc motion-correct --pet /path/to/pet.nii --pet-reference /path/to/sum.nii --out-dir /path/to/output
  - Writing TACs From Segmentation Masks:
    pet-cli-preproc write-tacs --pet /path/to/pet.nii --segmentation /path/to/seg_masks.nii --color-table-path /path/to/color_table.json --out-dir /path/to/output
  - Verbose:
    pet-cli-preproc -v [sub-command] [arguments]
""")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds common arguments ('--pet', '--out-dir', and '--prefix') to a provided ArgumentParser object.

    This function modifies the passed ArgumentParser object by adding three arguments commonly used in the script.
    It uses the add_argument method of the ArgumentParser class. After running this function, the parser will
    be able to accept and parse these additional arguments from the command line when run.

    .. note::
        This function modifies the passed `parser` object in-place and does not return anything.

    Args:
        parser (argparse.ArgumentParser): The argument parser object to which the arguments are added.

    Raises:
        argparse.ArgumentError: If a duplicate argument tries to be added.

    Side Effects:
        Modifies the ArgumentParser object by adding new arguments.

    Example:
        .. code-block:: python

            parser = argparse.ArgumentParser()
            _add_common_args(parser)
            args = parser.parse_args(['--pet', 'pet_file', '--out-dir', './', '--prefix', 'prefix'])
            print(args.pet)
            print(args.out_dir)
            print(args.prefix)
            
    """
    parser.add_argument('-p', '--pet', required=True, help='Path to PET file')
    parser.add_argument('-o', '--out-dir', default='./', help='Output directory')
    parser.add_argument('-f', '--prefix', default="sub_XXXX", help='Output file prefix')


def _generate_args() -> argparse.Namespace:
    """
    Generates command line arguments for method :func:`main`.

    Returns:
        args (argparse.Namespace): Arguments used in the command line and their corresponding values.
    """
    parser = argparse.ArgumentParser(prog='pet-cli-preproc',
                                     description='Command line interface for running PET pre-processing steps.',
                                     epilog=_PREPROC_EXAMPLES_, formatter_class=argparse.RawTextHelpFormatter)
    
    # create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help.")
    
    # create parser for "weighted-sum" command
    parser_sum = subparsers.add_parser('weighted-sum', help='Half-life weighted sum of 4D PET series.')
    _add_common_args(parser_sum)
    parser_sum.add_argument('-l', '--half-life', required=True, help='Half life of radioisotope in seconds.',
                            type=float)
    
    # create parser for "register" command
    parser_reg = subparsers.add_parser('register', help='Register 4D PET to MRI anatomical space.')
    _add_common_args(parser_reg)
    parser_reg.add_argument('-a', '--anatomical', required=True, help='Path to 3D anatomical image (T1w or T2w).',
                            type=str)
    parser_reg.add_argument('-r', '--pet-reference', default=None,
                            help='Path to reference image for motion correction, if not weighted_sum.')
    
    # create parser for the "motion-correct" command
    parser_moco = subparsers.add_parser('motion-correct', help='Motion correction for 4D PET using ANTS')
    _add_common_args(parser_moco)
    parser_moco.add_argument('-r', '--pet-reference', default=None,
                             help='Path to reference image for motion correction, if not weighted_sum.')
    
    # create parser for the "write-tacs" command
    parser_tac = subparsers.add_parser('write-tacs', help='Write ROI TACs from 4D PET using segmentation masks.')
    _add_common_args(parser_tac)
    parser_tac.add_argument('-s', '--segmentation', required=True,
                            help='Path to segmentation image in anatomical space.')
    parser_tac.add_argument('-c', '--color-table-path', required=True, help='Path to color table in JSON format')
    parser_tac.add_argument('-r', '--resample-segmentation', default=False, action='store_true',
                            help='Resample segmentation.')
    
    verb_group = parser.add_argument_group('Additional information')
    verb_group.add_argument('-v', '--verbose', action='store_true',
                            help='Print processing information during computation.', required=False)
    
    args = parser.parse_args()
    return args


def _check_ref(args) -> str:
    """
    Checks if the 'pet-reference' command-line argument was provided. If it was, this function
    will return its value. If 'pet-reference' argument was not provided, the function will
    generate an image path and directory using :func:`_generate_image_path_and_directory` and return this path.

    This function is used to determine the reference image for PET processing. If a reference
    image has been explicitly provided using `--pet-reference`, it's utilized. Otherwise, the
    function assumes the reference image file will be found in path produced by
    :func:`_generate_image_path_and_directory` with 'sum_image' as operations directory name and
    'sum' as operations description.

    Args:
        args: a namespace contains all command-line arguments. It's the result of ArgumentParser.parse_args().

    Returns:
        str: The path of the reference image.

    """
    if args.pet_reference is not None:
        ref_image = args.pet_reference
    else:
        ref_image = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                       ops_dir_name='sum_image',
                                                       file_prefix=args.prefix,
                                                       ops_desc='sum')
    return ref_image


def main():
    """
    Preprocessing command line interface
    """
    args = _generate_args()
    
    args.out_dir = os.path.abspath(args.out_dir)
    args.pet = os.path.abspath(args.pet)
    
    if args.command == 'weighted-sum':
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='sum_image',
                                                         file_prefix=args.prefix,
                                                         ops_desc='sum')
        image_operations_4d.weighted_series_sum(input_image_4d_path=args.pet,
                                                out_image_path=image_write,
                                                half_life=args.half_life,
                                                verbose=args.verbose)
    
    if args.command == 'motion-correct':
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='motion-correction',
                                                         file_prefix=args.prefix,
                                                         ops_desc='moco')
        ref_image = _check_ref(args=args)
        image_operations_4d.motion_correction(input_image_4d_path=args.pet,
                                              reference_image_path=ref_image,
                                              out_image_path=image_write,
                                              verbose=args.verbose)
    
    if args.command == 'register':
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='registration',
                                                         file_prefix=args.prefix,
                                                         ops_desc='reg')
        ref_image = _check_ref(args=args)
        image_operations_4d.register_pet(input_calc_image_path=ref_image,
                                         input_reg_image_path=args.pet,
                                         reference_image_path=args.anatomical,
                                         out_image_path=image_write,
                                         verbose=args.verbose)
    
    if args.command == 'write-tacs':
        tac_write_path = os.path.join(args.out_dir, 'tacs')
        os.makedirs(tac_write_path, exist_ok=True)
        image_write = _generate_image_path_and_directory(main_dir=args.out_dir,
                                                         ops_dir_name='segmentation',
                                                         file_prefix=args.prefix,
                                                         ops_desc='seg')
        if args.resample_segmentation:
            image_operations_4d.resample_segmentation(input_image_4d_path=args.pet,
                                                      segmentation_image_path=args.segmentation,
                                                      out_seg_path=image_write,
                                                      verbose=args.verbose)
        
        image_operations_4d.write_tacs(input_image_4d_path=args.pet,
                                       color_table_path=args.color_table_path,
                                       segmentation_image_path=image_write,
                                       out_tac_dir=tac_write_path,
                                       verbose=args.verbose)


if __name__ == "__main__":
    main()
