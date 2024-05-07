import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


_TEXT_BOX_ = {'facecolor': 'lightblue', 'edgecolor': 'black', 'lw': 2.0, 'alpha': 0.2}


def scatter_with_regression_figure(axes,
                                   fit_values: np.ndarray,
                                   true_values: np.ndarray,
                                   ax_titles: list[str],
                                   sca_kwargs: dict = None,
                                   reg_kwargs: dict = None):
    
    if sca_kwargs is None:
        sca_kwargs = dict(s=10, marker='.', color='red')
    
    if sca_kwargs is None:
        sca_kwargs = dict(s=10, color='black', alpha=0.8, lw=3, ls='-')
    
    fax = axes.flatten()
    for ax_id, (xAr, yAr, title) in enumerate((zip(true_values.T, fit_values.T, ax_titles))):
        x = xAr[~np.isnan(yAr)]
        y = yAr[~np.isnan(yAr)]
        lin_reg = linregress(x, y)
        
        fax[ax_id].scatter(x, y, **sca_kwargs)
        fax[ax_id].plot(x, x * lin_reg.slope + lin_reg.intercept, **reg_kwargs)
        
        fax[ax_id].text(0.05, 0.95, fr"$r^2={lin_reg.rvalue:<5.3f}$",
                        fontsize=20, transform=fax[ax_id].transAxes,
                        ha='left', va='top', bbox=_TEXT_BOX_)
        fax[ax_id].set_title(f"{title} Fits", fontweight='bold')
        fax[ax_id].set(xlabel=fr'True Values', ylabel=fr'Fit Values')