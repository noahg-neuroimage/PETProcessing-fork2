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
            ax.plot(self.x, y, '-', color='orange', lw=2.5,
                    zorder=3, label=self.generate_label_from_fit_params())
    
    def add_plots(self,
                  plot_data: bool = True,
                  plot_fit_points: bool = True,
                  plot_fit_lines: bool = True,
                  fit_shading: bool = True):
        if plot_data:
            self.add_data_plots()
        if plot_fit_points:
            self.add_fit_points()
        if plot_fit_lines:
            self.add_fit_lines()
        if fit_shading:
            self.add_shading_plots()
    
    def generate_figure(self,
                        plot_data: bool = True,
                        plot_fit_points: bool = True,
                        plot_fit_lines: bool = True,
                        fit_shading: bool = True):
        self.add_plots(plot_data=plot_data,
                       plot_fit_points=plot_fit_points,
                       plot_fit_lines=plot_fit_lines,
                       fit_shading=fit_shading)
        self.add_figure_labels_and_legend()
    
    @abstractmethod
    def calculate_x_and_y(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        pass
    
    @abstractmethod
    def generate_label_from_fit_params(self) -> str:
        pass
    
    @abstractmethod
    def add_figure_labels_and_legend(self):
        pass
    

class PatlakPlot(GraphicalAnalysisPlot):
    def calculate_x_and_y(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        good_points = np.argwhere(self.pTAC[1] != 0.0).T[0]
        
        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])[good_points] / self.pTAC[1][good_points]
        y = self.tTAC[1][good_points] / self.pTAC[1][good_points]
        
        fit_params = pet_grph.patlak_analysis_with_rsquared(input_tac_values=self.pTAC[1],
                                                            region_tac_values=self.tTAC[1],
                                                            tac_times_in_minutes=self.pTAC[0],
                                                            t_thresh_in_minutes=self.t_thresh_in_mins)
        
        fit_params = {'slope': fit_params[0],
                      'intercept': fit_params[1],
                      'r_squared':fit_params[2]}
        return x, y, fit_params

    def generate_label_from_fit_params(self) -> str:
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']
        
        return f"$K_1=${slope:<5.3f}\n$V_T=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"

    def add_figure_labels_and_legend(self):
        patlak_x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        patlak_y_label = r"$\frac{R(t)}{C_\mathrm{P}(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(patlak_x_label)
            ax.set_ylabel(patlak_y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(),
                        bbox_to_anchor=(1.0, 0.8),
                        loc='upper left',
                        title='Patlak Analysis')
        self.fig.suptitle("Patlak Plots")
        