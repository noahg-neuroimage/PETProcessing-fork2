import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nibabel


class NiftiGifCreator:
    def __init__(self,
                 path_to_image: str,
                 view: str,
                 output_directory: str,
                 output_filename_prefix: str = ""):
        
        self.view = view.lower()
        self._validate_view()
        
        self.path_to_image = os.path.abspath(path_to_image)
        self.output_directory = os.path.abspath(output_directory)
        self.prefix = output_filename_prefix
        
        self.image = nibabel.load(self.path_to_image).get_fdata()
        
        self.vmax = max(np.max(self.image), np.abs(np.min(self.image)))
        
        self.ani_image = None
        
        self.fig, self.ax = plt.subplots(1,1, constrained_layout=True)
        
        self.imKW = {'origin': 'lower',
                     'cmap': 'bwr',
                     'vmin': -self.vmax / 3.,
                     'vmax': self.vmax / 3.,
                     'interpolation': 'none'}
    
    
    def _validate_view(self):
        if self.view not in ['coronal', 'sagittal', 'axial', 'x', 'y', 'z']:
            raise ValueError("Invalid view. Please choose from 'coronal', 'sagittal', 'axial', 'x', 'y', 'z'.")