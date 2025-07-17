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
    figsize: tuple[int, int] = (8, 6),
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
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_title(f"Posterior of {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("Density")

    median = np.median(data)
    ax.axvline(median, linestyle="--", label=f"median = {median:.3g}")
    ax.legend()

    plt.tight_layout()
    plt.show()


def summarize_results(
    surrogate,
    metric: str,
    x_opt: np.ndarray,
    posterior_df: pd.DataFrame,
    uq_result: dict,
) -> None:
    """Print summary of calibration + UQ results."""
    print("\n" + "=" * 40)
    print(f"Summary for metric: {metric}")
    print("=" * 40)

    print(f"\nPoint Calibration:\n  x_opt = {x_opt}")

    print("\nPosterior Summary (mean ± std, 95% CI):")
    for param in posterior_df.columns:
        mean = posterior_df[param].mean()
        std = posterior_df[param].std()
        low, high = posterior_df[param].quantile([0.025, 0.975])
        print(
            f"{param: <10} = {mean:.5f} ± {std:.5f}  (95% CI: [{low:.5f}, {high:.5f}])"
        )

    mean = uq_result["mean"]
    ci = uq_result["quantiles"]
    print(f"\nUQ Result for '{metric}':")
    print(f"  Mean: {mean:.5f}")
    print(f"  95% CI: [{ci['2.5%']:.5f}, {ci['97.5%']:.5f}]")
