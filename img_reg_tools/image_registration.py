"""
Image registration tools for PET processing
"""
import numpy as np
import ants
import nibabel
import fsl


def run_image_registration(pet_image_series: np.ndarray, 
                           reference_image: np.ndarray,
                           input_affine: np.ndarray,
                           reference_affine: np.ndarray,
                           transform: np.ndarray) -> np.ndarray:
    """
    Given an input image and a transform, will perform
    linear registration based on the supplied transform.

    Arguments:
        pet_image_series (np.ndarray):
        reference_image (np.ndarray):
        input_affine (np.ndarray): 4x4
        reference_affine (np.ndarray): 4x4
        transform (np.ndarray): A 4x4 linear transformation matrix, following FSL convention.

    Returns:
        transformed_image (np.ndarray): Input image after being transformed based on the 
                                        supplied transform matrix

    """
    transformed_image = 1
    return transformed_image


def pet_to_atlas_registration(pet_image_series: np.ndarray,
                              atlas_image: np.ndarray,
                              init_transform=None,
                              compute_transform: bool=True) -> np.ndarray:
    """
    Warps pet image to atlas

    Arguments:
        pet_image_series (np.ndarray): 4D PET image to be resampled onto atlas.
        atlas_image (np.ndarray): Atlas image to which the PET image is warped onto.
        compute_transform (bool): Set to True to run the registration from PET to atlas space.
                                  Set to False if supplying a pre-computed transform.
        init_transform: Transformation matrix or list of transformation matrices to apply to
                        PET image if compute_transform is False; i.e. when user supplies
                        their own transforms.


    Returns:

    """
    pet_image_ants = ants.from_numpy(pet_image_series)
    atlas_image_ants = ants.from_numpy(atlas_image)

    if compute_transform:
        pet_on_atlas, _atlas_on_pet, _pet_to_atlas_xfm, _atlas_to_pet_xfm  = ants.registration(
            atlas_image_ants,
            pet_image_ants,
            type_of_transform='SyN')
        return pet_on_atlas

    pet_on_atlas = ants.apply_transforms(
        atlas_image_ants,
        pet_image_ants,
        transformlist=init_transform
    )
    return pet_on_atlas
