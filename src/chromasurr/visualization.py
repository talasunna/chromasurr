import matplotlib.pyplot as plt
import numpy as np


def plot_sobol_indices(
    sobol_results: dict[str, dict[str, np.ndarray]],
    metric: str,
    param_names: list[str] | None = None,
    figsize: tuple[int, int] = (8, 4)
) -> None:
    """
    Plot first-order and total-order Sobol sensitivity indices for a given metric.

    Parameters
    ----------
    sobol_results : dict
        Dictionary of results returned by `run_sensitivity_analysis`.
    metric : str
        The name of the metric to plot (e.g., "retention_time").
    param_names : list of str, optional
        The names of the parameters (optional if already in the sobol result).
    figsize : tuple, optional
        Size of the plot.
    """
    Si = sobol_results[metric]
    S1 = Si["S1"]
    ST = Si["ST"]
    conf1 = Si["S1_conf"]
    confT = Si["ST_conf"]

    if param_names is None:
        param_names = [f"Param {i+1}" for i in range(len(S1))]

    x = np.arange(len(param_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, S1, width, yerr=conf1, label="S1", capsize=4)
    ax.bar(x + width/2, ST, width, yerr=confT, label="ST", capsize=4)

    ax.set_ylabel("Sobol Index")
    ax.set_title(f"Sensitivity Analysis for '{metric}'")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_distributions(
    self,
    predictions: Dict[str, np.ndarray],
    bins: int = 30
) -> None:
    """
    Plot histograms of the propagated metric distributions.

    Parameters
    ----------
    predictions : dict
        Mapping metric name to numpy array of predictions.
    bins : int, default=30
        Number of histogram bins.

    Returns
    -------
    None
    """
    for metric, arr in predictions.items():
        plt.figure()
        plt.hist(arr, bins=bins, edgecolor='k', alpha=0.7)
        plt.title(f"Uncertainty distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
