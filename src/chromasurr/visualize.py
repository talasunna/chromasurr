import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def plot_sobol_indices(
    sobol_results: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    param_names: List[str] | None = None,
    figsize: Tuple[int, int] = (8, 4),
    ax: plt.Axes | None = None,
    sort: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of first-order (S1) and total-order (ST) Sobol indices with
    optional sorting and custom axis.

    Parameters
    ----------
    sobol_results : dict
        Output of `run_sensitivity_analysis`.
    metric : str
        Metric key whose indices to plot.
    param_names : list[str], optional
        Human-readable names; if None they are generated.
    figsize : tuple[int, int], optional
        Figure size (ignored if `ax` is given).
    ax : matplotlib.axes.Axes, optional
        Plot into an existing axis; if None a new figure/axis is created.
    sort : bool, default True
        Sort bars in descending order of ST (makes “important” inputs pop out).
    show : bool, default True
        Call `plt.show()`.  Disable when the caller wants to compose a figure.
    Returns
    -------
    fig, ax : the Matplotlib figure and axis objects.
    """
    Si = sobol_results[metric]
    S1, ST = np.array(Si["S1"]), np.array(Si["ST"])
    conf1, confT = np.array(Si["S1_conf"]), np.array(Si["ST_conf"])

    # tiny negatives arise from sampling noise
    S1, ST = np.clip(S1, 0.0, None), np.clip(ST, 0.0, None)

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

    # first- and total-order bars in contrasting colours
    ax.bar(x - width/2, S1, width, yerr=conf1,
           label="First-order $S_1$", capsize=4, color="#1f77b4")
    ax.bar(x + width/2, ST, width, yerr=confT,
           label="Total $S_T$", capsize=4, color="#ff7f0e", alpha=0.8)

    ax.set_ylabel("Sobol index")
    ax.set_title(f"Sensitivity – {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
