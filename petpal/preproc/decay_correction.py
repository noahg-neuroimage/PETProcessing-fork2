"""Decay Correction Module.

Provides functions for undo-ing decay correction and recalculating it."""

import numpy as np
from petpal.utils.image_io import ImageIO

def undo_decay_correction(input_image_path: str,
                          output_image_path: str,
                          verbose: bool = False) -> np.ndarray:
    """Uses decay factors from the .json sidecar file for an image to remove decay correction for each frame."""
    image_loader = ImageIO(verbose=verbose)

    nifti_image = image_loader.load_nii(image_path=input_image_path)

    frame_data = frame_data / decay_factor

    return frame_data

