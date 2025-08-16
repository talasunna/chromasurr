from __future__ import annotations
from typing import Any, Mapping, Sequence, cast
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

"""
visualize.py — plotting utilities for sensitivity analysis and Bayesian posterior.

This module provides functions to visualize Sobol sensitivity indices and
posterior distributions for surrogate model calibration.
"""


ArrayF = NDArray[np.float64]


def sobol_indices(
    sobol_results: Mapping[str, Mapping[str, ArrayF]],
    metric: str,
    param_names: Sequence[str] | None = None,
    figsize: tuple[int, int] = (10, 9),
    ax: Axes | None = None,
    sort: bool = True,
    show: bool = True,
    param_config: Mapping[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """
    Scatter plot with error bars for first-order (S1) & total-order (ST) Sobol indices.

    Parameters
    ----------
    sobol_results : Mapping[str, Mapping[str, ndarray]]
        Sobol analysis output (e.g., from SALib), keyed by metric name, with
        arrays under keys like ``"S1"``, ``"ST"``, ``"S1_conf"``, ``"ST_conf"``.
    metric : str
        The metric name whose Sobol indices to plot.
    param_names : Sequence[str] or None, optional
        Parameter labels; if ``None``, derive from ``param_config`` or fall back
        to ``["param_0", "param_1", ...]``.
    figsize : tuple[int, int], default=(10, 9)
        Figure size (ignored if ``ax`` is given).
    ax : Axes or None, optional
        Axis to plot on; if ``None``, a new figure and axis are created.
    sort : bool, default=True
        Sort parameters by decreasing ``ST``.
    show : bool, default=True
        Call :func:`matplotlib.pyplot.show` after plotting.
    param_config : Mapping[str, str] or None, optional
        Mapping from parameter names to their full path/description for labeling.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes objects for the plot.

    Raises
    ------
    ValueError
        If ``metric`` is not present in ``sobol_results``.
    """
    if metric not in sobol_results:
        raise ValueError(f"Metric '{metric}' not found in Sobol results.")

    Si = sobol_results[metric]

    S1 = np.clip(np.asarray(Si["S1"], dtype=float), 0.0, None)
    ST = np.clip(np.asarray(Si["ST"], dtype=float), 0.0, None)
    S1_conf = np.asarray(Si["S1_conf"], dtype=float)
    ST_conf = np.asarray(Si["ST_conf"], dtype=float)

    if param_names is None:
        names: list[str] = (
            list(param_config.keys())
            if param_config is not None
            else [f"param_{i}" for i in range(len(S1))]
        )
    else:
        names = list(param_names)

    if sort:
        order = np.argsort(ST)[::-1]
        S1 = S1[order]
        ST = ST[order]
        S1_conf = S1_conf[order]
        ST_conf = ST_conf[order]
        names = [names[i] for i in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    ax.errorbar(
        range(len(names)),
        S1,
        yerr=S1_conf,
        fmt="ro",
        label="First-order (S1)",
        markersize=5,
        capsize=5,
    )
    ax.errorbar(
        range(len(names)),
        ST,
        yerr=ST_conf,
        fmt="bs",
        label="Total-order (ST)",
        markersize=5,
        capsize=5,
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Sobol Indices")
    ax.set_title(f"Aggregated Sobol Indices for {metric}")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def uq_distribution(
    uq_result: Mapping[str, Any],
    metric: str,
    bins: int = 40,
    figsize: tuple[int, int] = (10, 7),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Histogram + KDE + 95% interval for one UQ output metric.

    Parameters
    ----------
    uq_result : Mapping[str, Any]
        Output containing keys like ``"y_means"`` (array-like) and
        ``"quantiles"`` (mapping with ``"2.5%"`` and ``"97.5%"``).
    metric : str
        Label for axes/title.
    bins : int, default=40
        Number of histogram bins.
    figsize : tuple[int, int], default=(10, 7)
        Figure size (ignored if ``ax`` is given).
    ax : Axes or None, optional
        Axis to plot on; if ``None``, a new figure and axis are created.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and axes objects for the plot.
    """
    data = np.asarray(uq_result["y_means"], dtype=float)
    ci = uq_result["quantiles"]  # Mapping[str, float] at runtime

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    sns.histplot(data, bins=bins, kde=True, stat="density", ax=ax)
    ax.axvline(float(ci["2.5%"]), color="k", ls="--", lw=1, label="95% CI")
    ax.axvline(float(ci["97.5%"]), color="k", ls="--", lw=1)
    ax.set_title(f"Uncertainty distribution: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def summarize_results(
    surrogate: Any,
    metric: str,
    uq_result: Mapping[str, Any],
) -> None:
    """
    Print a concise summary of calibration and UQ results.

    Parameters
    ----------
    surrogate : Any
        Surrogate object (unused here; kept for API symmetry).
    metric : str
        Metric name being summarized.
    uq_result : Mapping[str, Any]
        Result dictionary with keys like ``"mean"`` and ``"quantiles"``.
    """
    print("\n" + "=" * 40)
    print(f"Summary for metric: {metric}")
    print("=" * 40)

    mean = float(uq_result["mean"])
    ci = uq_result["quantiles"]
    print(f"\nUQ Result for '{metric}':")
    print(f"  Mean: {mean:.5f}")
    print(f"  95% CI: [{float(ci['2.5%']):.5f}, {float(ci['97.5%']):.5f}]")
