import numpy as np
import nibabel


def extract_from_nii_as_numpy(file_path: str) -> np.ndarray:
    """
    Convenient wrapper to extract data from a .nii or .nii.gz file as a numpy array
    :param file_path: The full path to the file.
    :return: The data contained in the .nii or .nii.gz file as a numpy array
    """
    
    return nibabel.load(filename=file_path).get_fdata()
    

def compute_average_over_mask(mask: np.ndarray, image: np.ndarray) -> float:
    """
    We compute the average value of `image` over the provided `mask`. We multiply the mask
    and the image element-wise, which gives us the image pixels corresponding to the mask. We then take the standard mean
    over those values by summing and dividing by the number of values. Note that the implementation does not care about
    the dimensions of mask and image; just that they be the same.
    :param mask: 3D numpy array that contains 1s and 0s.
    :param image: 3D numpy array corresponding to an image.
    :return: Average value of the image over the mask.
    """
    #TODO: Move this assert to a higher level function if speed is greatly affected by this when iterating over many frames
    assert mask.shape == image.shape, "The mask and the image should have the same dimensions."
    
    return np.sum(mask * image) / np.sum(mask)


def compute_average_over_mask_of_multiple_frames(mask: np.ndarray,
                                                 image_series: np.ndarray,
                                                 start: int,
                                                 step: int,
                                                 stop: int):
    """
    Given a mask, and multiple images, we compute the average value of the image over the mask. It is assumed that the last
    index for the images corresponds to different images or time-points. A simple and explicit iterator to stride over the
    images is included.
    :param mask:
    :param image_series:
    :param start:
    :param step:
    :param stop:
    :return:
    """
    #TODO: A smarter way to stride over images so that negative indecies can be used.
    assert start >= 0, "`start` has to be >= 0."
    assert stop <= image_series.shape[-1], "`end` has to be smaller than the number of frames in the image array."
    assert step >= 0, "`stride` has to be >= 0."
    
    frame_it = range(start, stop, step)
    num_frames = len(frame_it)
    avg_values = np.zeros(num_frames, float)
    
    for t, frmID in enumerate(frame_it):
        avg_values[t] = compute_average_over_mask(mask=mask, image=image_series[:, :, :, frmID])
    
    return avg_values
    