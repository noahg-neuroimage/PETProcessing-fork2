import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from . import parametric_images as pet_pim
from typing import Iterable, Tuple

nifty_loader = pet_pim._safe_load_4dpet_nifty

class NiftiGifCreator:
    def __init__(self,
                 path_to_image: str,
                 view: str,
                 output_directory: str,
                 output_filename_prefix: str = "",
                 fig_title: str = "Patlak-$K_i$ Parametric Image",
                 cbar_label: str = "$K_i$ (Infusion Rate)"):
        
        self.view = view.lower()
        self._validate_view()
        
        self.path_to_image = os.path.abspath(path_to_image)
        self.output_directory = os.path.abspath(output_directory)
        self.prefix = output_filename_prefix
        
        self.image = nifty_loader(self.path_to_image).get_fdata()
        
        self.vmax = max(np.max(self.image), np.abs(np.min(self.image)))
        
        self.fig, self.ax = plt.subplots(1, 1, constrained_layout=True)
        self.ani = None
        self.cbar = None
        
        self.imKW = {'origin': 'lower',
                     'cmap': 'bwr',
                     'vmin': -self.vmax / 3.,
                     'vmax': self.vmax / 3.,
                     'interpolation': 'none'}
        
        self.ani_image = self.make_first_frame()
        self.set_figure_title_and_labels(title=fig_title, cbar_label=cbar_label)
    
    def _validate_view(self):
        if self.view not in ['coronal', 'sagittal', 'axial', 'x', 'y', 'z']:
            raise ValueError("Invalid view. Please choose from 'coronal', 'sagittal', 'axial', 'x', 'y', or 'z'.")
    
    def make_first_frame(self):
        if self.view in ['x', 'sagittal']:
            img = self.image[0, :, :].T
        elif self.view in ['y', 'axial']:
            img = self.image[:, 0, :].T
        else:
            img = self.image[:, :, 0].T
        out_im = self.ax.imshow(img, **self.imKW)
        
        return out_im
    
    def set_figure_title_and_labels(self, title: str, cbar_label: str):
        self.cbar = self.fig.colorbar(self.ani_image, ax=self.ax, shrink=1.0)
        self.cbar.set_label(cbar_label, rotation=270)
        self.fig.suptitle(title)
        self.ani_image.axes.get_xaxis().set_visible(False)
        self.ani_image.axes.get_yaxis().set_visible(False)
        
    def update_frame(self, i):
        if self.view in ['x', 'sagittal']:
            img = self.image[i, :, :].T
        elif self.view in ['y', 'axial']:
            img = self.image[:, i, :].T
        else:
            img = self.image[:, :, i].T
        
        self.ani_image.set_data(img)
        
        return self.ani_image,
    
    def make_gif(self, frames: Iterable = None):
        
        if frames is None:
            tot_dims = self.image.shape
            if self.view in ['x', 'sagittal']:
                num_frames = tot_dims[0]
            elif self.view in ['y', 'axial']:
                num_frames = tot_dims[1]
            else:
                num_frames = tot_dims[2]
            frames = range(1, num_frames, 10)
        
        self.ani = mpl_animation.FuncAnimation(fig=self.fig,
                                               func=self.update_frame,
                                               frames=frames,
                                               blit=True)
        
    def write_gif(self):
        out_path = os.path.join(self.output_directory, f'{self.prefix}_view-{self.view}.gif')
        self.ani.save(f"{out_path}", fps=45, writer='pillow', dpi=100)
        plt.close(self.fig)
        