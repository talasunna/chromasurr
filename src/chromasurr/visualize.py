from __future__ import annotations

"""
visualize.py — plotting utilities for sensitivity analysis and Bayesian posterior.

This module provides functions to visualize Sobol sensitivity indices and
posterior distributions for surrogate model calibration.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns
import pandas as pd
from typing import Callable, Sequence


def sobol_indices(
    sobol_results: dict[str, dict[str, np.ndarray]],
    metric: str,
    param_names: list[str] | None = None,
    figsize: tuple[int, int] = (8, 4),
    ax: plt.Axes | None = None,
    sort: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of first-order (S1) and total-order (ST) Sobol indices.

    Parameters
    ----------
    sobol_results : dict
        Mapping from metric names to sensitivity results dict containing
        keys 'S1', 'ST', 'S1_conf', and 'ST_conf' (each array-like of length n_params).
    metric : str
        Metric key whose indices to plot.
    param_names : list of str, optional
        Human-readable parameter names; if None, generic labels are used.
    figsize : tuple of int, optional
        Figure size (width, height) in inches; ignored if `ax` is provided.
    ax : matplotlib.axes.Axes, optional
        Existing axes object to draw into; if None, a new figure and axes are created.
    sort : bool, default True
        Whether to sort parameters by descending total-order index ST.
    show : bool, default True
        If True, calls `plt.show()` before returning.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object with the bar plot.
    """
    Si = sobol_results[metric]
    S1 = np.clip(np.array(Si["S1"]), 0.0, None)
    ST = np.clip(np.array(Si["ST"]), 0.0, None)
    conf1 = np.array(Si["S1_conf"])
    confT = np.array(Si["ST_conf"])

    if param_names is None:
        param_names = [f"Param {i+1}" for i in range(len(S1))]

    if sort:
        order = np.argsort(ST)[::-1]
        S1, ST = S1[order], ST[order]
        conf1, confT = conf1[order], confT[order]
        param_names = [param_names[i] for i in order]

    x = np.arange(len(param_names))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(x - width/2, S1, width, yerr=conf1,
           label="First-order $S_1$", capsize=4)
    ax.bar(x + width/2, ST, width, yerr=confT,
           label="Total $S_T$", capsize=4, alpha=0.8)

    ax.set_ylabel("Sobol index")
    ax.set_title(f"Sensitivity — {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def posterior(
    posterior_samples: np.ndarray,
    param_names: list[str],
    param: str,
    bins: int = 30,
    kde: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (6, 4),
) -> None:
    """
    Plot posterior distribution for a single parameter.

    Parameters
    ----------
    posterior_samples : numpy.ndarray
        Posterior samples array of shape (n_samples, n_params).
    param_names : list of str
        Names of parameters corresponding to columns of posterior_samples.
    param : str
        Parameter name to plot.
    bins : int, default 30
        Number of histogram bins.
    kde : bool, default True
        Whether to overlay a kernel density estimate.
    xlim : tuple of float, optional
        Limits for x-axis.
    ylim : tuple of float, optional
        Limits for y-axis.
    figsize : tuple of int, optional
        Figure size in inches (width, height).

    Raises
    ------
    ValueError
        If `param` is not in `param_names`.
    """
    if param not in param_names:
        raise ValueError(f"'{param}' not in param_names.")

    idx = param_names.index(param)
    data = posterior_samples[:, idx]

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data, bins=bins, kde=kde, ax=ax, stat="density")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.set_title(f"Posterior of {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("Density")

    median = np.median(data)
    ax.axvline(median, linestyle="--", label=f"median = {median:.3g}")
    ax.legend()

    plt.tight_layout()
    plt.show()


def corner(
    posterior_samples: np.ndarray,
    param_names: Sequence[str],
    figsize: tuple[int, int] = (8, 8),
) -> None:
    """
    Plot a corner (pairplot) to visualize joint parameter uncertainty.

    Parameters
    ----------
    posterior_samples : numpy.ndarray
        Posterior samples array of shape (n_samples, n_params).
    param_names : sequence of str
        Parameter names for labeling axes.
    figsize : tuple of int, optional
        Figure size in inches (width, height).
    """
    df = pd.DataFrame(posterior_samples, columns=param_names)
    sns.pairplot(df, corner=True, kind="kde", diag_kind="kde")
    plt.suptitle("Posterior Pairwise Distributions", y=1.02)
    plt.gcf().set_size_inches(figsize)
    plt.show()


def summarize_posterior(
    posterior_samples: np.ndarray,
    param_names: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize posterior samples with mean, std, and 95% credible intervals.

    Parameters
    ----------
    posterior_samples : numpy.ndarray
        Posterior samples array of shape (n_samples, n_params).
    param_names : sequence of str
        Names of parameters corresponding to columns.

    Returns
    -------
    pandas.DataFrame
        Table with columns ['mean', 'std', '2.5%', '97.5%'] indexed by parameter name.
    """
    df = pd.DataFrame(posterior_samples, columns=param_names)
    summary = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "2.5%": df.quantile(0.025),
        "97.5%": df.quantile(0.975)
    }).round(5)

    print("\nPosterior Summary (mean ± std, 95% CI):")
    for name in param_names:
        μ = summary.loc[name, "mean"]
        σ = summary.loc[name, "std"]
        lo = summary.loc[name, "2.5%"]
        hi = summary.loc[name, "97.5%"]
        print(f"{name} = {μ:.5g} ± {σ:.2g}  (95% CI: [{lo:.3g}, {hi:.3g}])")

    return summary


def summarize_results(
    surrogate, metric: str, x_opt: np.ndarray,
    posterior_df: pd.DataFrame,
    uq_result: dict
) -> None:
    """Print summary of calibration + UQ results."""
    print("\n" + "="*40)
    print(f"Summary for metric: {metric}")
    print("="*40)

    print(f"\nPoint Calibration:\n  x_opt = {x_opt}")

    print("\nPosterior Summary (mean ± std, 95% CI):")
    for param in posterior_df.columns:
        mean = posterior_df[param].mean()
        std = posterior_df[param].std()
        low, high = posterior_df[param].quantile([0.025, 0.975])
        print(f"{param: <10} = {mean:.5f} ± {std:.5f}  (95% CI: [{low:.5f}, {high:.5f}])")

    mean = uq_result["mean"]
    ci = uq_result["quantiles"]
    print(f"\nUQ Result for '{metric}':")
    print(f"  Mean: {mean:.5f}")
    print(f"  95% CI: [{ci['2.5%']:.5f}, {ci['97.5%']:.5f}]")
