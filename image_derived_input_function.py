import numpy as np
import nibabel


def extract_from_nii_as_numpy(file_path: str) -> np.ndarray:
    """
    
    :param file_path:
    :return:
    """
    
    return nibabel.load(filename=file_path).get_fdata()
    


def compute_average_over_mask(mask: np.ndarray, image: np.ndarray) -> float:
    """
    
    :param mask: 3D numpy array that contains 1s and 0s.
    :param image: 3D numpy array corresponding to an image.
    :return: np.sum(mask * image) / np.sum(mask). The average value of image over the provided mask. We multiply the mask
    and the image element-wise, which gives us the image pixels corresponding to the mask. We then take the standard mean
    over those values by summing and dividing by the number of values.
    """
    # TODO: Move this assert to a higher level function if speed is greatly affected by this when iterating over many frames
    assert mask.shape == image.shape, "The mask and the image should have the same dimensions."
    
    return np.sum(mask * image) / np.sum(mask)
