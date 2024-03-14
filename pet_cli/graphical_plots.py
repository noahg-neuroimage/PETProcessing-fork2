from matplotlib import pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import seaborn as sns
from typing import Tuple, Dict
from . import graphical_analysis as pet_grph


class GraphicalAnalysisPlot(ABC):
    """
    This is an abstract base class designed for creating customizable plots for graphical analysis.
    It takes Time-Activity Curves (TACs) as input and creates various plots (x vs. y, fit lines, fit area
    shading plots, and fit points) based on user input. This class also calculates fit parameters for the plotting data
    and determines relevant indices based on given thresholds.

    Attributes:
        pTAC (np.ndarray): The Input/Plasma TAC, an array containing time points and corresponding activity.
        tTAC (np.ndarray): The Tissue/Region TAC, an array with time points and corresponding activity.
        t_thresh_in_mins (float): The threshold time, in minutes, to consider in the analysis. Points are fit after this
        threshold.
        fig (plt.Figure): A matplotlib Figure object where the plots will be drawn.
        ax_list (list): A list of matplotlib Axes associated with `fig` where the plots will be drawn.
        non_zero_idx (np.ndarray): Indexes of non-zero values in appropriate TAC (calculated in specific implementations).
        t_thresh_idx (int): The index at which the time threshold is crossed in the TACs (calculated in specific
        implementations).
        x (np.ndarray): The "x" values for plotting (calculated in specific implementations).
        y (np.ndarray): The "y" values for plotting (calculated in specific implementations).
        fit_params (dict): The parameters fit to the data using least squares.
                           Contains 'slope', 'intercept', and 'r_squared' (calculated in specific implementations).

    Note:
        This is an abstract class and should be inherited by a concrete class that implements the following methods:
        * :func:`calculate_valid_indicies_and_x_and_y`
        * :func:`generate_label_from_fit_params`
        * :func:`add_figure_labels_and_legend`
    """
    def __init__(self, pTAC: np.ndarray, tTAC: np.ndarray, t_thresh_in_mins: float, figObj: plt.Figure = None):
        """
        Initialize an instance of the GraphicalAnalysisPlot class.

        The instance is initialized with two Time-Activity Curves (TACs), a threshold time, and an optional matplotlib
        Figure. It calculates valid indices (where the denominator is non-zero for the particular analysis), 'x' and 'y'
        values for plotting based on the TACs and the threshold time, and also analyzes the TACs to generate the fits.

        Args:
            pTAC (np.ndarray): The input Time Activity Curve, an array containing time points and corresponding activity.
            tTAC (np.ndarray): The Tissue or Region Time Activity Curve, an array with time points and corresponding
            activity.
            t_thresh_in_mins (float): The threshold time in minutes to consider when performing calculations for the
            plots.
            figObj (plt.Figure, optional): An optional matplotlib Figure object. If not provided, a new Figure object is
             created.

        Raises:
            matplotlib error: Error handling for the plot generation is managed by matplotlib. Any exceptions thrown
            during plotting are handled by the matplotlib library internally.
        """
        self.pTAC: np.ndarray = pTAC[:]
        self.tTAC: np.ndarray = tTAC[:]
        self.t_thresh_in_mins: float = t_thresh_in_mins
        self.fig, self.ax_list = self.generate_figure_and_axes(figObj=figObj)
        self.non_zero_idx: np.ndarray = None
        self.t_thresh_idx: int = None
        self.x: np.ndarray = None
        self.y: np.ndarray = None
        self.fit_params: Dict = None
        self.calculate_valid_indicies_and_x_and_y()
        self.calculate_fit_params()
        
    @staticmethod
    def generate_figure_and_axes(figObj: plt.Figure = None):
        """
        Generate a matplotlib Figure and Axes for plotting.

        A new Figure and Axes are created if no Figure object is provided. If a Figure object is provided, the method
        retrieves the existing axes from the Figure object. In either case, the method returns the Figure and a list of
        its Axes.

        Args:
            figObj (plt.Figure, optional): An optional matplotlib Figure object. If not provided, a new Figure object is
             created with 2 subplots arranged in 1 row, a figure size of 8x4, line width of 3.0, and edge color 'k'.

        Returns:
            fig (plt.Figure): The resulting matplotlib Figure object.
            ax_list (list): A list of Axes objects associated with 'fig'.

        Raises:
            None
        """
        if figObj is None:
            fig, ax_list = plt.subplots(1, 2,
                                        constrained_layout=True, figsize=[8, 4],
                                        linewidth=3.0, edgecolor='k')
            ax_list = ax_list.flatten()
        else:
            fig = figObj
            ax_list = fig.get_axes()
        return fig, ax_list

    def add_data_plots(self, pl_kwargs: dict = None):
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.plot(self.x, self.y, lw=1, alpha=0.9, ms=8, marker='.', zorder=1, color='black')
        else:
            for ax in self.ax_list:
                ax.plot(self.x, self.y, **pl_kwargs)
                
    def add_shading_plots(self, pl_kwargs: dict = None):
        x_lo, x_hi = self.x[self.t_thresh_idx], self.x[-1]
        
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.axvspan(x_lo, x_hi, color='gray', alpha=0.2, zorder=0)
        else:
            for ax in self.ax_list:
                ax.axvspan(x_lo, x_hi, **pl_kwargs)
                
    def add_fit_points(self, pl_kwargs: dict = None):
        t_thresh = self.t_thresh_idx
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.plot(self.x[t_thresh:], self.y[t_thresh:], 'o', alpha=0.9, ms='5', zorder=2, color='blue')
        else:
            for ax in self.ax_list:
                ax.plot(self.x[t_thresh:], self.y[t_thresh:], **pl_kwargs)
                
    def add_fit_lines(self, pl_kwargs: dict = None):
        y = self.x * self.fit_params['slope'] + self.fit_params['intercept']
        if pl_kwargs is None:
            for ax in self.ax_list:
                ax.plot(self.x, y, '-', color='orange', lw=2.5,
                        zorder=3, label=self.generate_label_from_fit_params())
        else:
            for ax in self.ax_list:
                ax.plot(self.x, y,  **pl_kwargs)
    
    def add_plots(self,
                  plot_data: bool = True,
                  plot_fit_points: bool = True,
                  plot_fit_lines: bool = True,
                  fit_shading: bool = True,
                  data_kwargs: dict = None,
                  points_kwargs: dict = None,
                  line_kwargs: dict = None,
                  shading_kwargs: dict = None):
        if plot_data:
            self.add_data_plots(pl_kwargs=data_kwargs)
        if plot_fit_points:
            self.add_fit_points(pl_kwargs=points_kwargs)
        if plot_fit_lines:
            self.add_fit_lines(pl_kwargs=line_kwargs)
        if fit_shading:
            self.add_shading_plots(pl_kwargs=shading_kwargs)
    
    def generate_figure(self,
                        plot_data: bool = True,
                        plot_fit_points: bool = True,
                        plot_fit_lines: bool = True,
                        fit_shading: bool = True,
                        data_kwargs: dict = None,
                        points_kwargs: dict = None,
                        line_kwargs: dict = None,
                        shading_kwargs: dict = None):
        self.add_plots(plot_data=plot_data,
                       plot_fit_points=plot_fit_points,
                       plot_fit_lines=plot_fit_lines,
                       fit_shading=fit_shading,
                       data_kwargs=data_kwargs,
                       points_kwargs=points_kwargs,
                       line_kwargs=line_kwargs,
                       shading_kwargs=shading_kwargs)
        self.add_figure_labels_and_legend()
        self.ax_list[0].set_title("Linear Plot")
        self.ax_list[1].set_title("LogLog Plot")
        
        self.ax_list[1].set(yscale='log', xscale='log')
    
    def calculate_fit_params(self):
        t_thresh = self.t_thresh_idx
        fit_params = pet_grph.fit_line_to_data_using_lls_with_rsquared(xdata=self.x[t_thresh:], ydata=self.y[t_thresh:])
        
        fit_params = {
            'slope': fit_params[0], 'intercept': fit_params[1], 'r_squared': fit_params[2]
            }
        self.fit_params = fit_params

    @abstractmethod
    def calculate_valid_indicies_and_x_and_y(self) -> None:
        pass
    
    @abstractmethod
    def generate_label_from_fit_params(self) -> str:
        pass
    
    @abstractmethod
    def add_figure_labels_and_legend(self):
        pass
    

class PatlakPlot(GraphicalAnalysisPlot):
    def calculate_valid_indicies_and_x_and_y(self) -> None:
        non_zero_indices = np.argwhere(self.pTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][non_zero_indices],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)
        
        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])
        x = x[non_zero_indices] / self.pTAC[1][non_zero_indices]
        y = self.tTAC[1][non_zero_indices] / self.pTAC[1][non_zero_indices]
        
        self.x = x[:]
        self.y = y[:]
        self.non_zero_idx = non_zero_indices[:]
        self.t_thresh_idx = t_thresh
        return None
    
    def generate_label_from_fit_params(self) -> str:
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']
        
        return f"$K_1=${slope:<5.3f}\n$V_T=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"

    def add_figure_labels_and_legend(self):
        x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        y_label = r"$\frac{R(t)}{C_\mathrm{P}(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(),
                        bbox_to_anchor=(1.0, 0.8),
                        loc='upper left',
                        title='Patlak Analysis')
        self.fig.suptitle("Patlak Plots")


class LoganPlot(GraphicalAnalysisPlot):
    def calculate_valid_indicies_and_x_and_y(self) -> None:
        non_zero_indices = np.argwhere(self.tTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][non_zero_indices],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)
        
        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])
        y = pet_grph.cumulative_trapezoidal_integral(xdata=self.tTAC[0], ydata=self.tTAC[1])
        
        x = x[non_zero_indices] / self.tTAC[1][non_zero_indices]
        y = y[non_zero_indices] / self.tTAC[1][non_zero_indices]
        
        self.x = x[:]
        self.y = y[:]
        self.non_zero_idx = non_zero_indices[:]
        self.t_thresh_idx = t_thresh
        return None
    
    def generate_label_from_fit_params(self) -> str:
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']
        
        return f"$m=${slope:<5.3f}\n$b=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"
    
    def add_figure_labels_and_legend(self):
        x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{R(t)}$"
        y_label = r"$\frac{\int_{0}^{t}R(s)\mathrm{d}s}{R(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(), bbox_to_anchor=(1.0, 0.8), loc='upper left',
                        title='Logan Analysis')
        self.fig.suptitle("Logan Plots")


class AltLoganPlot(GraphicalAnalysisPlot):
    def calculate_valid_indicies_and_x_and_y(self) -> None:
        non_zero_indices = np.argwhere(self.pTAC[1] != 0.0).T[0]
        t_thresh = pet_grph.get_index_from_threshold(times_in_minutes=self.pTAC[0][non_zero_indices],
                                                     t_thresh_in_minutes=self.t_thresh_in_mins)
        
        x = pet_grph.cumulative_trapezoidal_integral(xdata=self.pTAC[0], ydata=self.pTAC[1])
        y = pet_grph.cumulative_trapezoidal_integral(xdata=self.tTAC[0], ydata=self.tTAC[1])
        
        x = x[non_zero_indices] / self.pTAC[1][non_zero_indices]
        y = y[non_zero_indices] / self.pTAC[1][non_zero_indices]
        
        self.x = x[:]
        self.y = y[:]
        self.non_zero_idx = non_zero_indices[:]
        self.t_thresh_idx = t_thresh
        return None
    
    def generate_label_from_fit_params(self) -> str:
        slope = self.fit_params['slope']
        intercept = self.fit_params['intercept']
        r_sq = self.fit_params['r_squared']
        
        return f"$m=${slope:<5.3f}\n$b=${intercept:<5.3f}\n$R^2=${r_sq:<5.3f}"
    
    def add_figure_labels_and_legend(self):
        x_label = r"$\frac{\int_{0}^{t}C_\mathrm{P}(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        y_label = r"$\frac{\int_{0}^{t}R(s)\mathrm{d}s}{C_\mathrm{P}(t)}$"
        for ax in self.ax_list:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        self.fig.legend(*self.ax_list[0].get_legend_handles_labels(), bbox_to_anchor=(1.0, 0.8), loc='upper left',
                        title='Alt-Logan Analysis')
        self.fig.suptitle("Alt-Logan Plots")
