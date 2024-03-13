from matplotlib import pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import seaborn as sns
from . import graphical_analysis


class GraphicalAnalysisPlot(ABC):
    
    def __init__(self, pTAC: np.ndarray, tTAC: np.ndarray, t_thresh_in_mins: float, figObj: plt.Figure = None):
        self.pTAC = pTAC[:]
        self.tTAC = tTAC[:]
        self.t_thres = t_thresh_in_mins
        self.fig, self.ax_list = self.generate_figure_and_axes(figObj=figObj)
    
    @staticmethod
    def generate_figure_and_axes(figObj: plt.Figure = None):
        if figObj is None:
            fig, ax_list = plt.subplots(1, 2,
                                        constrained_layout=True, figsize=[8, 4],
                                        linewidth=3.0, edgecolor='k')
            ax_list = ax_list.flatten()
        else:
            fig = figObj
            ax_list = fig.get_axes()
        return fig, ax_list


