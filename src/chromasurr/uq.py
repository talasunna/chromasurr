from __future__ import annotations

"""
uq.py — forward uncertainty propagation with a trained surrogate.

This module provides utilities for Latin hypercube sampling and Monte Carlo
uncertainty quantification of surrogate model predictions.
"""

import logging
from typing import Callable, Sequence

import numpy as np

from chromasurr.surrogate import Surrogate

_logger = logging.getLogger(__name__)

__all__ = [
    "latin_hypercube_sampler",
    "perform_monte_carlo_uq",
]


def latin_hypercube_sampler(
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int | None = None,
) -> Callable[[int], np.ndarray]:
    """
    Create a Latin hypercube sampler for given parameter bounds.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        Sequence of (min, max) pairs for each model input dimension.
    seed : int, optional
        Seed for random number generator to ensure reproducibility.

    Returns
    -------
    Callable[[int], np.ndarray]
        Function that takes an integer n and returns an (n, len(bounds))
        array of sampled inputs.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    span = np.asarray([b[1] - b[0] for b in bounds], dtype=float)

    def _sample(n: int) -> np.ndarray:
        u = (rng.random((n, len(bounds))) + rng.permutation(n)[:, None]) / n
        return lo + u * span

    return _sample


def _to_list(x: str | Sequence[str]) -> list[str]:
    """
    Ensure the input is a list of strings.

    Parameters
    ----------
    x : str or Sequence[str]
        Single metric name or sequence of metric names.

    Returns
    -------
    list[str]
        List of metric names.
    """
    return [x] if isinstance(x, str) else list(x)


def perform_monte_carlo_uq(
    *,
    surrogate: Surrogate,
    sample_input: Callable[[int], np.ndarray],
    metrics: str | Sequence[str] | None = None,
    metric: str | None = None,
    n_samples: int = 10_000,
) -> (
    dict[str, dict[str, float | np.ndarray | dict[str, float]]]
    | dict[str, float | np.ndarray | dict[str, float]]
):
    """
    Propagate input uncertainty via Monte Carlo sampling using a surrogate model.

    Parameters
    ----------
    surrogate : Surrogate
        Trained surrogate model providing `predict` and `predict_var` methods.
    sample_input : Callable[[int], np.ndarray]
        Function that generates an array of shape (n_samples, num_vars) of input samples.
    metrics : str or Sequence[str], optional
        Metric name or list of metric names to evaluate. If None, uses all surrogate.metrics.
    metric : str, optional
        Legacy single-metric alias; if provided, `metrics` must be None.
    n_samples : int, default 10000
        Number of Monte Carlo samples to draw.

    Returns
    -------
    dict or dict of dict
        If `metric` is specified or a single metric, returns a dict with keys:
        'mean', 'variance', 'var_between', 'var_within', 'quantiles', 'y_means', 'y_vars'.
        If multiple metrics, returns a mapping from each metric name to its statistics dict.

    Raises
    ------
    ValueError
        If both `metric` and `metrics` are provided, or if the sampled input array
        does not match the expected shape (n_samples, num_vars).
    """
    if metric is not None and metrics is not None:
        raise ValueError("Pass either 'metric' or 'metrics', not both.")
    if metric is not None:
        metrics_list = [metric]
        flatten = True
    else:
        metrics_list = _to_list(metrics) if metrics is not None else surrogate.metrics
        flatten = len(metrics_list) == 1

    X = sample_input(n_samples)
    expected_shape = (n_samples, surrogate.problem["num_vars"])  # type: ignore[index]
    if X.shape != expected_shape:
        raise ValueError("sample_input returned array of wrong shape")

    results: dict[str, dict[str, float | np.ndarray | dict[str, float]]] = {}

    for m in metrics_list:
        y_means = surrogate.predict(m, X)
        y_vars = surrogate.predict_var(m, X)

        mu = float(np.mean(y_means))
        var_between = float(np.var(y_means, ddof=1))
        var_within = float(np.mean(y_vars))
        var_total = var_between + var_within
        q025, q50, q975 = np.percentile(y_means, [2.5, 50, 97.5])

        results[m] = {
            "mean": mu,
            "variance": var_total,
            "var_between": var_between,
            "var_within": var_within,
            "quantiles": {"2.5%": float(q025), "50%": float(q50), "97.5%": float(q975)},
            "y_means": y_means,
            "y_vars": y_vars,
        }

    if flatten:
        return results[metrics_list[0]]
    return results
