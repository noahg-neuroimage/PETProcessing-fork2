"""
Image registration tools for PET processing
"""
import numpy as np
import ants
import nibabel


def run_image_registration(input_image: np.ndarray, input_affine: np.ndarray,
                           transform: np.ndarray) -> np.ndarray:
    """
    Given an input image and a transform, will perform
    linear registration based on the supplied transform.

    Arguments:
        input_image (np.ndarray):
        input_affine (np.ndarray):
        transform (np.ndarray): A 4x4 linear transformation matrix, following FSL convention.

    Returns:
        transformed_image (np.ndarray): Input image after being transformed based on the 
                                        supplied transform matrix

    """
    transformed_image = 1
    return transformed_image