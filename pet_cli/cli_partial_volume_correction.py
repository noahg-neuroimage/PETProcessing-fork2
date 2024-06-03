"""
Command-line interface (CLI) for Partial Volume Correction (PVC) using SGTM and PETPVC methods.

This module provides a CLI to apply PVC to PET images using either the SGTM method or the PETPVC package.
It uses argparse to handle command-line arguments and chooses the appropriate method based on the provided input.

The user must provide:
    * PET image file path
    * ROI image file path
    * FWHM for Gaussian blurring
    * PVC method ('SGTM' or any other method for PETPVC)
    * Additional options for PETPVC

Example usage:
    Using SGTM method:
        .. code-block:: bash

            pvc_cli.py --method SGTM --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 8.0

    Using PETPVC method:
        .. code-block:: bash

            pvc_cli.py --method RBV --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 6.0 --output-path /path/to/output_image.nii

See Also:
    SGTM and PETPVC methods implementation modules.

"""

import argparse
import nibabel as nib
from symmetric_geometric_transfer_matrix import sgtm
from partial_volume_corrections import PetPvc


def main():
    """
    Main function to handle command-line arguments and apply the appropriate PVC method.
    """
    parser = argparse.ArgumentParser(
        prog="PVC CLI",
        description="Apply Partial Volume Correction (PVC) to PET images using SGTM or PETPVC methods.",
        epilog="Example of usage: pvc_cli.py --method SGTM --pet-path /path/to/pet_image.nii --roi-path /path/to/roi_image.nii --fwhm 8.0"
    )

    parser.add_argument("--method", required=True, help="PVC method to use (SGTM or PETPVC method).")
    parser.add_argument("--pet-path", required=True, help="Path to the PET image file.")
    parser.add_argument("--roi-path", required=True, help="Path to the ROI image file.")
    parser.add_argument("--fwhm", required=True, type=float, help="Full Width at Half Maximum for Gaussian blurring.")
    parser.add_argument("--output-path", help="Path to the output image file (for PETPVC method).")
    parser.add_argument("--verbose", action="store_true", help="Print additional information.")
    parser.add_argument("--mask-path", help="Path to the mask image file (optional, for PETPVC method).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (for PETPVC method).")

    args = parser.parse_args()

    pet_img = nib.load(args.pet_path)
    roi_img = nib.load(args.roi_path)

    if args.method.lower() == "sgtm":
        """
        Apply the SGTM method for Partial Volume Correction.

        Args:
            pet_img (nib.Nifti1Image): The 3D PET image.
            roi_img (nib.Nifti1Image): The 3D ROI image.
            fwhm (float): Full Width at Half Maximum for Gaussian blurring.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Labels, corrected PET values, and condition number of the omega matrix.
        """
        labels, corrected_values, cond_number = sgtm(pet_img, roi_img, args.fwhm)
        if args.verbose:
            print("Labels:", labels)
            print("Corrected values:", corrected_values)
            print("Condition number:", cond_number)
    else:
        """
        Apply the PETPVC method for Partial Volume Correction.

        Args:
            pet_img (nib.Nifti1Image): The 3D PET image.
            roi_img (nib.Nifti1Image): The 3D ROI image.
            fwhm (float): Full Width at Half Maximum for Gaussian blurring.
            output_path (str): The path where the output image will be saved.
            mask_path (str, optional): The path to the mask image.
            debug (bool, optional): Enable debug mode for detailed logs.
            verbose (bool, optional): Print additional information.

        """
        petpvc_handler = PetPvc()
        petpvc_handler.run_petpvc(
            pet_4d_filepath=args.pet_path,
            output_filepath=args.output_path,
            pvc_method=args.method,
            psf_dimensions=args.fwhm,
            mask_filepath=args.mask_path,
            verbose=args.verbose,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
