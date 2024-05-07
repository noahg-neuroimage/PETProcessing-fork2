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
    
    if reg_kwargs is None:
        reg_kwargs = dict(s=10, color='black', alpha=0.8, lw=3, ls='-')
    
    fax = axes.flatten()
    for ax_id, (xAr, yAr, title) in enumerate((zip(true_values.T, fit_values.T, ax_titles))):
        x = xAr[~np.isnan(yAr)]
        y = yAr[~np.isnan(yAr)]
        fax[ax_id].scatter(x, y, **sca_kwargs)
        
        lin_reg = linregress(x, y)
        fax[ax_id].plot(x, x * lin_reg.slope + lin_reg.intercept, **reg_kwargs)
        
        fax[ax_id].text(0.05, 0.95, fr"$r^2={lin_reg.rvalue:<5.3f}$",
                        fontsize=20, transform=fax[ax_id].transAxes,
                        ha='left', va='top', bbox=_TEXT_BOX_)
        fax[ax_id].set_title(f"{title} Fits", fontweight='bold')
        fax[ax_id].set(xlabel=fr'True Values', ylabel=fr'Fit Values')


def bland_atlman_figure(axes,
                        fit_values: np.ndarray,
                        true_values: np.ndarray,
                        ax_titles: list[str],
                        sca_kwargs: dict = None,
                        bland_kwargs: dict = None):
    
    if sca_kwargs is None:
        sca_kwargs = dict(s=10, marker='.', color='red')
    
    if bland_kwargs is None:
        bland_kwargs = dict(s=10, color='red', alpha=0.8, lw=1)
    
    fax = axes.flatten()
    for ax_id, (xAr, yAr, title) in enumerate((zip(fit_values.T, true_values.T, ax_titles))):
        x = (xAr + yAr) / 2.0
        y = xAr - yAr
        
        fax[ax_id].scatter(x, y, **sca_kwargs)
        
        mean_diff = np.nanmean(y)
        std_dev = np.nanstd(y)
        mid = mean_diff
        hi = mean_diff + 1.96 * std_dev
        lo = mean_diff - 1.96 * std_dev
        fax[ax_id].axhline(hi, ls='--', zorder=0, color=bland_kwargs['color'])
        fax[ax_id].axhline(lo, ls='--', zorder=0, color=bland_kwargs['color'])
        fax[ax_id].axhline(mid, ls='-', zorder=0, color=bland_kwargs['color'])
        
        fax[ax_id].set_title(f"{title} Fits", fontweight='bold')
        fax[ax_id].set(xlabel=fr'$\frac{{S_1+S_2}}{{2}}$ (Mean)', ylabel=fr'$S_1-S_2$ (Diff.)')