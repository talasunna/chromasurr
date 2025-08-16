from __future__ import annotations
from typing import Callable, Sequence

"""
uq.py — forward uncertainty propagation with a trained surrogate.

This module provides utilities for Latin hypercube sampling and Monte Carlo
uncertainty quantification of surrogate model predictions.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from chromasurr.surrogate import Surrogate

_logger = logging.getLogger(__name__)

__all__ = [
    "latin_hypercube_sampler",
    "perform_monte_carlo_uq",
]
ArrayF = NDArray[np.float64]


def latin_hypercube_sampler(
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int | None = None,
) -> Callable[[int], ArrayF]:
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
    Callable[[int], ndarray of float]
        Function that takes an integer n and returns an (n, len(bounds))
        array of sampled inputs.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    span = np.asarray([b[1] - b[0] for b in bounds], dtype=float)

    def _sample(n: int) -> ArrayF:
        # Latin hypercube in (0,1): stratify each dimension then shuffle rows
        d = len(bounds)
        u = (rng.random((n, d)) + rng.permutation(n)[:, None]) / n
        return np.asarray(lo + u * span, dtype=float)

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
    sample_input: Callable[[int], ArrayF],
    metrics: str | Sequence[str] | None = None,
    metric: str | None = None,
    n_samples: int = 10_000,
) -> (
    dict[str, dict[str, float | ArrayF | dict[str, float]]]
    | dict[str, float | ArrayF | dict[str, float]]
):
    """
    Propagate input uncertainty via Monte Carlo sampling using a surrogate model.

    Parameters
    ----------
    surrogate : Surrogate
        Trained surrogate model providing ``predict`` and ``predict_var``.
    sample_input : Callable[[int], ndarray]
        Function that generates an array of shape ``(n_samples, num_vars)``
        of input samples.
    metrics : str or Sequence[str], optional
        Metric name or list of metric names to evaluate. If ``None``, uses
        ``surrogate.metrics``.
    metric : str, optional
        Legacy single-metric alias; if provided, ``metrics`` must be ``None``.
    n_samples : int, default 10000
        Number of Monte Carlo samples to draw.

    Returns
    -------
    dict or dict of dict
        If a single metric, returns a dict with keys:
        ``'mean'``, ``'variance'``, ``'var_between'``, ``'var_within'``,
        ``'quantiles'``, ``'y_means'``, ``'y_vars'``.
        If multiple metrics, returns a mapping from metric name to its stats.
    """
    if metric is not None and metrics is not None:
        raise ValueError("Pass either 'metric' or 'metrics', not both.")
    if metric is not None:
        metrics_list = [metric]
        flatten = True
    else:
        metrics_list = _to_list(metrics) if metrics is not None else surrogate.metrics
        flatten = len(metrics_list) == 1

    X: ArrayF = sample_input(n_samples)
    expected_shape = (n_samples, surrogate.problem["num_vars"])
    if X.shape != expected_shape:
        raise ValueError("sample_input returned array of wrong shape")

    results: dict[str, dict[str, float | ArrayF | dict[str, float]]] = {}

    for m in metrics_list:
        y_means: ArrayF = surrogate.predict(m, X)
        y_vars: ArrayF = surrogate.predict_var(m, X)

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
