from matplotlib import pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import seaborn as sns
from typing import Tuple, Dict
from . import graphical_analysis as pet_grph


class GraphicalAnalysisPlot(ABC):
    
    def __init__(self, pTAC: np.ndarray, tTAC: np.ndarray, t_thresh_in_mins: float, figObj: plt.Figure = None):
        self.pTAC = pTAC[:]
        self.tTAC = tTAC[:]
        self.t_thresh_in_mins = t_thresh_in_mins
        self.fig, self.ax_list = self.generate_figure_and_axes(figObj=figObj)
        self.x, self.y, self.fit_params = self.calculate_x_and_y()
        
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

    def add_data_plots(self):
        for ax in self.ax_list:
            ax.plot(self.x, self.y, lw=1, alpha=0.9, ms=8, marker='.', zorder=1, color='black')

    # TODO: Refactor so that the `good_points` and `t_thresh` calculation is only done once.
    def add_shading_plots(self):
        good_points = np.argwhere(self.pTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][good_points],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)
        x_lo, x_hi = self.x[t_thresh], self.x[-1]
        for ax in self.ax_list:
            ax.axvspan(x_lo, x_hi, color='gray', alpha=0.2, zorder=0)

    def add_fit_points(self):
        good_points = np.argwhere(self.pTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][good_points],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)
        for ax in self.ax_list:
            ax.plot(self.x[t_thresh:], self.y[t_thresh:], 'o', alpha=0.9, ms='5', zorder=2, color='blue')
    
    def add_fit_lines(self):
        y = self.x * self.fit_params['slope'] + self.fit_params['intercept']
        for ax in self.ax_list:
            ax.plot(self.x, y, '-', color='orange', lw=2.5, zorder=3, label=self.generate_label_from_fit_params())
    
    def add_plots(self, plot_data: bool = True,
                        plot_fit_points: bool = True,
                        plot_fit_lines: bool = True,
                        fit_shading: bool = True):
        for ax in self.ax_list:
            if plot_data:
                self.add_data_plots()
            if plot_fit_points:
                self.add_fit_points()
            if plot_fit_lines:
                self.add_fit_lines()
            if fit_shading:
                self.add_shading_plots()
        
    
    @abstractmethod
    def calculate_x_and_y(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        pass
    
    @abstractmethod
    def generate_label_from_fit_params(self) -> str:
        pass