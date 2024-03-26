import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from . import parametric_images as pet_pim

nifty_loader = pet_pim._safe_load_4dpet_nifty

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
        
        self.image_obj = nifty_loader(self.path_to_image)
        self.image = self.image_obj.get_fdata()
        
        self.vmax = max(np.max(self.image), np.abs(np.min(self.image)))
        
        self.fig, self.ax = plt.subplots(1, 1, constrained_layout=True)
        
        self.imKW = {'origin': 'lower',
                     'cmap': 'bwr',
                     'vmin': -self.vmax / 3.,
                     'vmax': self.vmax / 3.,
                     'interpolation': 'none'}
        
        self.ani_image = self.make_first_frame(axis=self.view)
    
    def _validate_view(self):
        if self.view not in ['coronal', 'sagittal', 'axial', 'x', 'y', 'z']:
            raise ValueError("Invalid view. Please choose from 'coronal', 'sagittal', 'axial', 'x', 'y', 'z'.")
    
    def make_first_frame(self, axis):
        if axis in ['x', 'coronal']:
            img = self.image[0, :, :].T
        elif axis == ['y', 'sagittal']:
            img = self.image[:, 0, :].T
        else:
            img = self.image[:, :, 0].T
        out_im = self.ax.imshow(img, **self.imKW)
        
        cbar = self.fig.colorbar(out_im, ax=self.ax, shrink=1.0)
        cbar.set_label('$K_i$ (Infusion Rate)', rotation=270)
        self.fig.suptitle("Patlak-$K_i$ Parametric Image")
        out_im.axes.get_xaxis().set_visible(False)
        out_im.axes.get_yaxis().set_visible(False)
        
        return out_im
        
        
    def update_frame(self, i, axis):
        if axis in ['x', 'coronal']:
            img = self.image[i, :, :].T
        elif axis == ['y', 'sagittal']:
            img = self.image[:, i, :].T
        else:
            img = self.image[:, :, i].T
        
        self.ani_image.set_data(img)
        
        return self.ani_image,
        