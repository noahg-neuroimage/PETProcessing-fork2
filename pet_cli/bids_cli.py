import argparse
import BIDS_utils


def main():
    parser = argparse.ArgumentParser(description="Client interface for BIDS utilities.")
    parser.add_argument('--project_path', required=True, help="Path to the BIDS project")
    parser.add_argument('--subject', required=True, help="Subject ID")
    parser.add_argument('--session', required=True, help="Session ID")
    # parser.add_argument('--modality', required=True, help="Modality (E.g. 'anat', 'pet')")
    # parser.add_argument('--image_type', required=True, help="Image type (E.g. 'T1w', 'FLAIR', 'FDG')")
    parser.add_argument('--PET_image', required=True, help="Input 4D PET image file")
    parser.add_argument('--PET_label', required=True, help="Label of PET type (E.g. 'FDG', 'CS1P1')")
    parser.add_argument('--T1w_image', help="Input T1-weighted MR image file")
    parser.add_argument('--T1w_label', help="Label of T1-weighted MRI type (E.g. 'T1w', 'MPRAGE')")
    parser.add_argument('--T2w_image', help="Input T2-weighted MR image file")
    parser.add_argument('--T2w_label', help="Label of T2-weighted MRI type (E.g. 'T2w', 'FLAIR')")
    # parser.add_argument('--acquisition', help="Acquisition details", default=None)
    # parser.add_argument('--contrast_enhancing', help="Contrast enhancing agent ID", default=None)
    # parser.add_argument('--reconstruction', help="Reconstruction method", default=None)
    # parser.add_argument('--space', help="Reference image space", default=None)
    # parser.add_argument('--description', help="Other descriptions", default=None)

    args = parser.parse_args()

    if args.T1w_image and not args.T1w_label:
        parser.error("--T1w_label is required when --T1w_image is used.")
    if args.T2w_image and not args.T2w_label:
        parser.error("--T2w_label is required when --T2w_image is used.")

    bids_instance = BIDS_utils.BidsInstance(project_path=args.project_path, subject=args.subject)
    bids_instance.create_filepath(session=args.session,
                                  modality="pet",
                                  image_type=args.PET_label)
    bids_instance.write_symbolic_link(input_filepath=args.PET_image)
    bids_instance.cache_filepath(name="raw_PET")
    if args.T1w_image:
        bids_instance.change_parts(modality="anat",
                                   image_type=args.T1w_label)




if __name__ == "__main__":
    main()