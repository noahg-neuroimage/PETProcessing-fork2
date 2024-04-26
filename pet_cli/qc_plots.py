"""
Module for generating quality control plots.
"""
import numpy as np
import pandas as pd
import seaborn as sns

def motion_plot(framewise_displacement: np.ndarray,
                output_plot: str=None):
    """
    Plots the quantity of motion estimated by :meth:`ants.motion_correction`.
    Takes the framewise displacement returned by ANTs, and plots this quantity
    against frame number.

    Args:
        framewise_displacement (np.ndarray): Total movement, or displacement,
            estimated between consecutive frames.
        output_plot (str): File to which plot is saved. If `None`, write to
            a temporary file. Default value `None`.
    
    Returns:
        movement_plot (sns.lineplot): Plot of total movement between frames.
    """
    movement_dataframe = pd.DataFrame(columns=['Framewise Displacement (mm)'])
    movement_dataframe['Framewise Displacement (mm)'] = framewise_displacement
    movement_plot = sns.lineplot(movement_dataframe,y='Framewise Displacement (mm)')
    return movement_plot

