import argparse
from pet_cli.image_operations_4d import ImageOps4D

def main():
    """
    Parses command-line arguments and invokes the appropriate functions from the image_operations_4d
    package.

    The script supports various operations, including creating a BIDS scaffold,
    validating BIDS datasets (future), and converting user-provided files to BIDS format.
    """
    parser = argparse.ArgumentParser(prog="PET Processing",
                                     description="General purpose suite for processing PET images.",
                                     epilog="PET Processing complete.")

    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument('--subject', required=True, help="Name of the subject.")
    io_grp.add_argument('--out_path',required=True,help='Path to which results are output.')
    io_grp.add_argument('--pet', required=True, help="Path to the PET image. Required format .nii \
        or .nii.gz")
    io_grp.add_argument('--anatomical', required=False, help="Path to the anatomical image. \
        Required format .nii or .nii.gz")
    io_grp.add_argument('--segmentation', required=False, help="Path to the segmentation image. \
        Required format .nii or .nii.gz")
    io_grp.add_argument('--pet_sum_image', required=False, help="Path to a weighted sum PET image. \
        Required format .nii or .nii.gz")
    io_grp.add_argument('--pet_moco', required=False, help="Path to motion corrected PET image. \
        Required format .nii or .nii.gz")
    io_grp.add_argument('--color_table', required=False, help="Path to the color table.")
    io_grp.add_argument('--half_life', required=False, help="Half life of tracer radioisotope \
        in seconds.")

    comp_group = parser.add_argument_group("Computations")
    comp_group.add_argument('--weighted_sum', required=False, help="Compute a weighted sum PET \
        image.")
    comp_group.add_argument('--motion_correct', required=False, help="Motion correct PET image \
        series.")
    comp_group.add_argument('--register', required=False, help='Register PET to anatomical imaging \
        data.')
    comp_group.add_argument('--tacs', required=False, help='Compute TACs for regions in color \
        table.')

    verb_group = parser.add_argument_group("Additional information")
    verb_group.add_argument("-v", "--verbose", action="store_true",
                            help="Print the shape of the mask and images files.", required=False)

    args = parser.parse_args()

    sum_image = None

    if args.weighted_sum:
        image_paths = {'pet': args.pet}
        operations = ImageOps4D(
            sub_id=args.subject,
            image_paths=image_paths,
            half_life=args.half_life,
            out_path=args.out_path,
            verbose=args.verbose
        )
        operations.weighted_series_sum()
        sum_image = f'{args.out_path}/sum_image/{args.subject}-sum.nii.gz'

    if args.motion_correct:
        image_paths = {'pet': args.pet}
        if sum_image is None:
            image_paths['pet_sum_image'] = args.pet_sum_image
        else:
            image_paths['pet_sum_image'] = sum_image
        operations = ImageOps4D(
            sub_id=args.subject,
            image_paths=image_paths,
            out_path=args.out_path,
            verbose=args.verbose
        )
        operations.motion_correction()
        motion_corrected = f'{args.out_path}/pet_moco/{args.subject}-moco.nii.gz'

    if args.register:
        image_paths = {'pet': args.pet,'mri': args.anatomical}
        if sum_image is None:
            image_paths['pet_sum_image'] = args.pet_sum_image
        else:
            image_paths['pet_sum_image'] = sum_image
        if motion_corrected is None:
            image_paths['pet_moco'] = args.pet_moco
        else:
            image_paths['pet_moco'] = motion_corrected
        operations = ImageOps4D(
            sub_id=args.subject,
            image_paths=image_paths,
            out_path=args.out_path,
            verbose=args.verbose
        )
        operations.register_pet()
        pet_reg = f'{args.out_path}/registration/{args.subject}-moco-reg.nii.gz'

    if args.tacs:
        image_paths = {'pet_moco_reg': pet_reg, 'seg': args.segmentation}
        operations = ImageOps4D(
            sub_id=args.subject,
            image_paths=image_paths,
            half_life=args.half_life,
            color_table_path=args.color_table,
            out_path=args.out_path,
            verbose=args.verbose
        )
        operations.write_tacs()

if __name__ == "__main__":
    main()
