import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter


def sgtm(pet_3d: np.ndarray,
         roi_3d: np.ndarray,
         fwhm: float,
         resolution: float,
         zeroth_roi: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply Symmetric Geometric Transfer Matrix (SGTM) method for Partial Volume Correction (PVC) to PET images based on ROI labels.

    This method involves using a matrix-based approach to adjust the PET signal intensities for the effects of partial volume averaging.

    Args:
        pet_3d (np.ndarray): The 3D PET image array.
        roi_3d (np.ndarray): The 3D ROI image array, should have the same dimensions as `pet_3d`.
        fwhm (float): Full width at half maximum of the Gaussian blurring kernel.
        resolution (float): Spatial resolution of the PET image (same units as FWHM).
        zeroth_roi (bool): If False, ignores the zero label in calculations, often used to exclude background or non-ROI regions.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - np.ndarray: Array of unique ROI labels.
            - np.ndarray: Corrected PET values after applying PVC.
            - float: Condition number of the omega matrix, indicating the numerical stability of the inversion.

    Raises:
        AssertionError: If `pet_3d` and `roi_3d` do not have the same dimensions.

    Examples:
        .. code-block:: python

            >>> pet_image = np.random.rand(64, 64, 64)
            >>> roi_image = np.random.randint(0, 4, size=(64, 64, 64))
            >>> fwhm, resolution = 8.0, 2.5
            >>> labels, corrected_values, cond_number = sgtm(pet_image, roi_image, fwhm, resolution)
            >>> labels.shape
            (3,)
            >>> corrected_values.shape
            (3,)
            >>> type(cond_number)
            <class 'float'>

    Notes:
        The SGTM method uses the matrix :math:`\Omega` (omega), defined as:

        .. math::
            \Omega = V^T V

        where :math:`V` is the matrix obtained by applying Gaussian filtering to each ROI, converting each ROI into a vector. The element :math:`\Omega_{ij}` of the matrix :math:`\Omega` is the dot product of vectors corresponding to the i-th and j-th ROIs, representing the spatial overlap between these ROIs after blurring.

        The vector :math:`t` is calculated as:

        .. math::
            t = V^T p

        where :math:`p` is the vectorized PET image. The corrected values, :math:`t_{corrected}`, are then obtained by solving the linear system:

        .. math::
            \Omega t_{corrected} = t

        This provides the estimated activity concentrations corrected for partial volume effects in each ROI.
    """
    assert pet_3d.shape == roi_3d.shape, "PET and ROI images must be the same dimensions"
    unique_labels = np.unique(roi_3d)
    if not zeroth_roi:
        unique_labels = unique_labels[unique_labels != 0]

    sigma = (fwhm / 2.355) / resolution

    flattened_size = pet_3d.size
    voxel_by_roi_matrix = np.zeros((flattened_size, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        masked_roi = (roi_3d == label).astype(float)
        blurred_roi = gaussian_filter(masked_roi, sigma=sigma)
        voxel_by_roi_matrix[:, i] = blurred_roi.ravel()

    omega = voxel_by_roi_matrix.T @ voxel_by_roi_matrix

    t_vector = voxel_by_roi_matrix.T @ pet_3d.ravel()
    t_corrected = np.linalg.solve(omega, t_vector)
    condition_number = np.linalg.cond(omega)

    return unique_labels, t_corrected, condition_number
