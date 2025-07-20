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


def sobol_indices(
    sobol_results: dict[str, dict[str, np.ndarray]],
    metric: str,
    param_names: list[str] | None = None,
    figsize: tuple[int, int] = (10, 9),
    ax: plt.Axes | None = None,
    sort: bool = True,
    show: bool = True,
    param_config: dict[str, str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot with error bars for first-order (S1) & total-order (ST) Sobol indices.

    Parameters
    ----------
    sobol_results : dict[str, dict[str, np.ndarray]]
        The Sobol analysis results, which are dictionaries returned from the
        `run_sensitivity_analysis`.
    metric : str
        The metric name whose Sobol indices to plot.
    param_names : list[str], optional
       Paremeter names. If None, the parameter names are extracted from the
       Sobol results.
    figsize : tuple[int, int], optional
        Figure size (ignored if `ax` is given).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on; if None, a new figure and axis are created.
    sort : bool, optional
        Whether to sort the indices (default is True).
    show : bool, optional
        Whether to show the plot (default is True).
    param_config : dict[str, str], optional
        Mapping from parameter names to their full descriptions for proper labeling.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects for the plot.
    """
    if metric not in sobol_results:
        raise ValueError(f"Metric '{metric}' not found in Sobol results.")

    Si = sobol_results[metric]

    S1 = np.clip(np.array(Si["S1"]), 0.0, None)
    ST = np.clip(np.array(Si["ST"]), 0.0, None)
    S1_conf = np.array(Si["S1_conf"])
    ST_conf = np.array(Si["ST_conf"])

    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(S1))]
        if param_config:
            param_names = list(param_config.keys())

    if sort:

        sorted_indices = np.argsort(ST)[::-1]
        S1 = S1[sorted_indices]
        ST = ST[sorted_indices]
        param_names = [param_names[i] for i in sorted_indices]
        S1_conf = S1_conf[sorted_indices]
        ST_conf = ST_conf[sorted_indices]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.errorbar(
        range(len(param_names)),
        S1,
        yerr=S1_conf,
        fmt="ro",
        label="First-order (S1)",
        markersize=5,
        capsize=5,
    )
    ax.errorbar(
        range(len(param_names)),
        ST,
        yerr=ST_conf,
        fmt="bs",
        label="Total-order (ST)",
        markersize=5,
        capsize=5,
    )

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Sobol Indices")
    ax.set_title(f"Aggregated Sobol Indices for {metric}")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def uq_distribution(
    uq_result: dict,
    metric: str,
    bins: int = 40,
    figsize: tuple[int, int] = (10, 7),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram + KDE + 95 % interval for one UQ output metric."""
    data = uq_result["y_means"]
    ci = uq_result["quantiles"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.histplot(data, bins=bins, kde=True, stat="density", ax=ax)
    ax.axvline(ci["2.5%"], color="k", ls="--", lw=1, label="95 % CI")
    ax.axvline(ci["97.5%"], color="k", ls="--", lw=1)
    ax.set_title(f"Uncertainty distribution: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def summarize_results(
    surrogate,
    metric: str,
    uq_result: dict,
) -> None:
    """Print summary of calibration + UQ results."""
    print("\n" + "=" * 40)
    print(f"Summary for metric: {metric}")
    print("=" * 40)

    mean = uq_result["mean"]
    ci = uq_result["quantiles"]
    print(f"\nUQ Result for '{metric}':")
    print(f"  Mean: {mean:.5f}")
    print(f"  95% CI: [{ci['2.5%']:.5f}, {ci['97.5%']:.5f}]")

def uq_variance_bar(
    uq_result: dict,
    metric: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    between = uq_result["var_between"]
    within  = uq_result["var_within"]
    ax.bar([0], [between], label="Between‑sample", width=0.6)
    ax.bar([0], [within],  bottom=[between], label="Emulator", width=0.6)

    ax.set_xticks([0])
    ax.set_xticklabels([metric])
    ax.set_ylabel("Variance")
    ax.set_title(f"Variance breakdown – {metric}")
    ax.legend()
    plt.tight_layout()
    return fig, ax
