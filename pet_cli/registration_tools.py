import ants
import nibabel.nifti1
from nibabel.filebasedimages import FileBasedHeader, FileBasedImage
import numpy as np
import SimpleITK as sITK


def load_nii(file_path: str, verbose: bool) -> FileBasedImage:
    """Wrapper to load nifti from file path.

    Args:
        file_path: The full file path to the nifti file.
        verbose:

    Returns:
        The nifti FileBasedImage.

    """
    image = nibabel.load(file_path)

    if verbose:
        print(f"(fileIO): {file_path} loaded")

    return image


def extract_image_from_nii_as_numpy(image: nibabel.nifti1.Nifti1Image, verbose: bool) -> np.ndarray:
    """Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array.

    Args:
        image: The nifti file itself.
        verbose:

    Returns:
        The data contained in the .nii or .nii.gz file as a numpy array.

    """
    image_data: np.ndarray = image.get_fdata()

    if verbose:
        print(f"(fileIO): Image has shape {image_data.shape}")

    return image_data


def extract_header_from_nii(image: nibabel.nifti1.Nifti1Image, verbose: bool) -> FileBasedHeader:
    """Convenient wrapper to extract header information from a .nii or .nii.gz file as a nibabel file-based header.

    Args:
        image: The nifti file itself.
        verbose:

    Returns:
        The nifti header.

    """
    image_header: FileBasedHeader = image.header

    if verbose:
        print(f"(fileIO): Image header is: {image_header}")

    return image_header


def reorient_to_ras(image: nibabel.nifti1.Nifti1Image, verbose: bool) -> nibabel.nifti1.Nifti1Image:
    """Wrapper for the RAS reorientation used to ensure images are oriented the same.

    Args:
        image: The nifti file itself.
        verbose:

    Returns:
        The reoriented nifti file.

    """
    reoriented_image: nibabel.nifti1.Nifti1Image = nibabel.as_closest_canonical(image)

    if verbose:
        print(f"(fileIO): Image has been reoriented to RAS")

    return reoriented_image


def apply_3d_mr_bias_correction(image: nibabel.nifti1.Nifti1Image, verbose: bool) -> nibabel.nifti1.Nifti1Image:
    """Wrapper for bias field correction on 3-dimensional MR images, uses N4BiasFieldCorrection

    Args:
        image: The nifti file itself.
        verbose:

    Returns:
        N4BiasField-corrected 3D MR nifti image.

    """
    image_array = extract_image_from_nii_as_numpy(image=image, verbose=False)
    image_header = extract_header_from_nii(image=image, verbose=False)
    sitk_image = sITK.GetImageFromArray(image_array.astype(np.float32))
    sitk_image.SetSpacing(image_header.get_zooms()[:3])

    corrector = sITK.N4BiasFieldCorrectionImageFilter()
    corrected_sitk_image = corrector.Execute(sitk_image)

    corrected_image_array = sITK.GetArrayFromImage(corrected_sitk_image)

    # noinspection PyTypeChecker
    corrected_nibabel_image: nibabel.nifti1.Nifti1Image = nibabel.Nifti1Image(dataobj=corrected_image_array,
                                                                              affine=image.affine)

    if verbose:
        print(f"(fileIO): 3D MR image has been bias corrected")

    return corrected_nibabel_image


def convert_nib_to_ants(nib_image: nibabel.nifti1.Nifti1Image) -> ants.ANTsImage:
    """Converts nibabel image format (nibabel.nifti1.Nifti1Image) to ants image format (ANTsImage).

    Args:
        nib_image: Image in nibabel.nifti1.Nifti1Image format.

    Returns:
        Image in ANTsImage format.

    """
    nib_image_data: np.ndarray = nib_image.get_fdata()
    nib_image_affine = nib_image.affine
    ants_image = ants.from_numpy(nib_image_data)
    ants_image.set_spacing(tuple(np.diag(nib_image_affine)[:3]))

    return ants_image


def rigid_3d_registration(moving_image: nibabel.nifti1.Nifti1Image, fixed_image: nibabel.nifti1.Nifti1Image,
                          verbose: bool) -> nibabel.nifti1.Nifti1Image:
    """

    Args:
        moving_image:
        fixed_image:
        verbose:

    Returns:

    """
    moving_image_ants = convert_nib
