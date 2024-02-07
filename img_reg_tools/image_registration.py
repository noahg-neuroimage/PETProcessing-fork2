"""
Image registration tools for PET processing
"""
import ants
import nibabel


def pet_to_atlas_registration(pet_image_series: nibabel.nifti1,
                              atlas_image: nibabel.nifti1,
                              init_transform=None,
                              compute_transform: bool=True) -> nibabel.nifti1:
    """
    Warps pet image to atlas

    Arguments:
        pet_image_series (nibabel.nifti1): 4D PET image to be resampled onto atlas.
        atlas_image (nibabel.nifti1): Atlas image to which the PET image is warped onto.
        compute_transform (bool): Set to True to run the registration from PET to atlas space.
                                  Set to False if supplying a pre-computed transform.
        init_transform: Transformation matrix or list of transformation matrices to apply to
                        PET image if compute_transform is False; i.e. when user supplies
                        their own transforms.


    Returns:

    """
    pet_image_ants = ants.from_nibabel(pet_image_series)
    atlas_image_ants = ants.from_nibabel(atlas_image)

    if compute_transform:
        # TODO: save pet_to_atlas_xfm to file
        pet_on_atlas_ants, _atlonpet, _pet_to_atlas_xfm, _atlas_to_pet_xfm  = ants.registration(
            atlas_image_ants,
            pet_image_ants,
            type_of_transform='SyN')
    else:
        pet_on_atlas_ants = ants.apply_transforms(
            atlas_image_ants,
            pet_image_ants,
            transformlist=init_transform
        )

    pet_on_atlas = ants.utils.convert_nibabel.to_nibabel(pet_on_atlas_ants)
    return pet_on_atlas


def pet_to_subject_registration(pet_image_series: nibabel.nifti1,
                                mpr_image: nibabel.nifti1,
                                init_transform=None,
                                compute_transform: bool=True):
    """
    Rigid transform pet -> subject T1 space

    Arguments:
        pet_image_series (nibabel.nifti1): 4D PET image to be resampled onto MPRAGE.
        atlas_image (nibabel.nifti1): MPRAGE image to which the PET image is warped onto.
        compute_transform (bool): Set to True to run the registration from PET to subject MPRAGE space.
                                  Set to False if supplying a pre-computed transform.
        init_transform: Transformation matrix or list of transformation matrices to apply to
                        PET image if compute_transform is False; i.e. when user supplies
                        their own transforms.

    Returns:
        pet_on_mpr (nibabel.nifti1): 4D PET image resampled onto the MPRAGE.
    """
    pet_image_ants = ants.from_nibabel(pet_image_series)
    mpr_image_ants = ants.from_nibabel(mpr_image)

    if compute_transform:
        pet_on_mpr_ants, _mpr_on_pet, _pet_to_mpr_xfm, _mpr_to_pet_xfm  = ants.registration(
            mpr_image_ants,
            pet_image_ants,
            type_of_transform='Rigid')
    else:
        pet_on_mpr_ants = ants.apply_transforms(
            mpr_image_ants,
            pet_image_ants,
            transformlist=init_transform
        )
    
    pet_on_mpr = ants.utils.convert_nibabel.to_nibabel(pet_on_mpr_ants)
    return pet_on_mpr
