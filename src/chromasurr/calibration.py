from __future__ import annotations

"""calibration.py – deterministic and Bayesian calibration of a Surrogate.
"""

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import emcee
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from chromasurr.surrogate import Surrogate

_logger = logging.getLogger(__name__)

__all__ = ["CalibrationResult", "calibrate_surrogate", "bayesian_calibration"]

EPS: float = 1e-12


@dataclass(slots=True)
class CalibrationResult:
    """Container for calibration outputs."""

    x_opt: np.ndarray
    success: bool
    fun: float
    extra: dict[str, object]


def _sigma_total(sigma_pred: float, sigma_obs: float) -> float:
    """Combine observation and emulator uncertainties."""
    return max(math.hypot(sigma_pred, sigma_obs), EPS)


def _residual_sum_squares(
    params: np.ndarray,
    surrogate: Surrogate,
    y_obs: Mapping[str, float],
    sigma_obs: Mapping[str, float],
    metric_weights: Mapping[str, float],
) -> float:
    """Compute weighted residual sum-of-squares across metrics."""
    ssq = 0.0
    X = params.reshape(1, -1)
    for m, y_true in y_obs.items():
        y_pred = surrogate.predict(m, X).item()
        sigma_pred = math.sqrt(surrogate.predict_var(m, X).item())
        sigma = _sigma_total(sigma_pred, sigma_obs.get(m, 0.0))
        w = metric_weights.get(m, 1.0)
        ssq += w * ((y_pred - y_true) / sigma) ** 2
    return ssq


def _log_posterior(
    params: np.ndarray,
    surrogate: Surrogate,
    y_obs: Mapping[str, float],
    sigma_obs: Mapping[str, float],
    metric_weights: Mapping[str, float],
) -> float:
    """Evaluate Gaussian log-likelihood (including normalization constants)."""
    lp = 0.0
    X = params.reshape(1, -1)
    for m, y_true in y_obs.items():
        y_pred = surrogate.predict(m, X).item()
        sigma_pred = math.sqrt(surrogate.predict_var(m, X).item())
        sigma = _sigma_total(sigma_pred, sigma_obs.get(m, 0.0))
        w = metric_weights.get(m, 1.0)
        lp -= (
            0.5
            * w
            * (
                ((y_pred - y_true) / sigma) ** 2
                + 2.0 * math.log(sigma)
                + math.log(2.0 * math.pi)
            )
        )
    return lp


def calibrate_surrogate(
    surrogate: Surrogate,
    *,
    y_obs: Union[Mapping[str, float], float],
    metric: str | None = None,
    sigma_obs: Mapping[str, float] | None = None,
    metric_weights: Mapping[str, float] | None = None,
    x0: np.ndarray | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    method: str = "L-BFGS-B",
    tol: float | None = 1e-8,
    maxiter: int = 500,
    disp: bool = False,
) -> CalibrationResult:
    """Perform point-estimate calibration via weighted least squares.

    Handles both new mapping API and legacy single-metric API.

    Parameters
    ----------
    surrogate : Surrogate
        The trained surrogate model to calibrate.
    y_obs : mapping or float
        Observed metric values. If `metric` is provided, `y_obs` can be a single float;
        otherwise it must be a mapping from metric names to observed values.
    metric : str, optional
        Name of the metric corresponding to `y_obs` when `y_obs` is a single float.
    sigma_obs : mapping, optional
        Observation uncertainties by metric.
    metric_weights : mapping, optional
        Weights for each metric in the loss function.
    x0 : array-like, optional
        Initial guess for parameters.
    bounds : sequence of (low, high) pairs, optional
        Bounds on parameters.
    method : str, default "L-BFGS-B"
        Optimization method passed to `scipy.optimize.minimize`.
    tol : float, optional
        Tolerance for optimization termination.
    maxiter : int, default 500
        Maximum number of iterations.
    disp : bool, default False
        If True, prints convergence messages.

    Returns
    -------
    CalibrationResult
        *x_opt* best-fit parameters, *fun* RSS value, *extra* holds OptimizeResult.
    """
    if metric is not None:
        if isinstance(y_obs, Mapping):
            y_map = y_obs
        else:
            y_map = {metric: float(y_obs)}
    else:
        if not isinstance(y_obs, Mapping):
            raise ValueError(
                "When `metric` is not provided, `y_obs` must be a mapping of metric names to values."
            )
        y_map = y_obs  # type: ignore[assignment]

    sigma_obs = sigma_obs or {}
    metric_weights = metric_weights or {}
    bounds = bounds or surrogate.problem["bounds"]  # type: ignore[index]

    if x0 is None:
        x0 = np.mean(np.asarray(bounds, dtype=float), axis=1)

    obj = lambda p: _residual_sum_squares(
        p, surrogate, y_map, sigma_obs, metric_weights
    )
    res: OptimizeResult = minimize(
        obj,
        x0=x0,
        bounds=bounds,
        method=method,
        options={"maxiter": maxiter, "disp": disp},
        tol=tol,
    )

    return CalibrationResult(
        x_opt=res.x.copy(),
        success=res.success,
        fun=float(res.fun),
        extra={"opt_result": res},
    )


def bayesian_calibration(
    surrogate: Surrogate,
    *,
    y_obs: Mapping[str, float],
    sigma_obs: Mapping[str, float] | None = None,
    metric_weights: Mapping[str, float] | None = None,
    n_walkers: int | None = None,
    n_steps: int = 5000,
    burn_in: int | None = None,
    initial_radius: float = 1e-2,
    seed: int | None = None,
    progress: bool = True,
) -> CalibrationResult:
    """
    Perform Bayesian calibration via the emcee EnsembleSampler.

    This function calibrates the surrogate model by running an MCMC (Markov Chain Monte Carlo) simulation using the
    emcee EnsembleSampler. The posterior distribution is used to estimate the optimal parameters for the surrogate.

    Parameters
    ----------
    surrogate : Surrogate
        The surrogate model that will be calibrated using the MCMC method.
    y_obs : Mapping[str, float]
        A mapping of the observed values for each metric to compare against the
        surrogate predictions.
    sigma_obs : Mapping[str, float] | None, optional
        The standard deviation of the observed values, used to calculate the likelihood.
    metric_weights : Mapping[str, float] | None, optional
        A mapping of weights to assign to each metric in the likelihood function.
    n_walkers : int | None, optional
        The number of walkers (MCMC chains). If not provided, the default is
        4 * number of parameters or 20, whichever is larger.
    n_steps : int, optional
        The number of steps (iterations) for the MCMC sampler. Default is 5000.
    burn_in : int | None, optional
        The number of burn-in steps to discard before starting to collect samples.
        Default is None, which will use 20% of `n_steps`.
    initial_radius : float, optional
        The radius for initializing the walkers' positions around the initial point.
        Default is 1e-2.
    seed : int | None, optional
        A random seed to ensure reproducibility of the results.
    progress : bool, optional
        Whether to display a progress bar during the MCMC sampling. Default is True.

    Returns
    -------
    CalibrationResult
        An object containing the optimal parameters (`x_opt`), success status, the
        log-probability of the optimal parameters, and additional information such as
        the sampler object and the MCMC chain.
    """
    sigma_obs = sigma_obs or {}
    metric_weights = metric_weights or {}

    ndim = surrogate.problem["num_vars"]
    bounds = np.asarray(surrogate.problem["bounds"], dtype=float)

    det = calibrate_surrogate(
        surrogate,
        y_obs=y_obs,
        sigma_obs=sigma_obs,
        metric_weights=metric_weights,
    )
    x0 = det.x_opt

    rng = np.random.default_rng(seed)
    n_w = n_walkers or max(4 * ndim, 20)
    p0 = x0 + initial_radius * rng.standard_normal((n_w, ndim))
    p0 = np.clip(p0, bounds[:, 0], bounds[:, 1])

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
            return -np.inf
        return _log_posterior(theta, surrogate, y_obs, sigma_obs, metric_weights)

    sampler = emcee.EnsembleSampler(n_w, ndim, log_prob)

    if seed is not None:
        np.random.seed(seed)

    sampler.run_mcmc(p0, n_steps, progress=progress)

    bi = burn_in or int(0.2 * n_steps)
    chain = sampler.get_chain(discard=bi, flat=True)
    lp = sampler.get_log_prob(discard=bi, flat=True)

    idx = int(np.argmax(lp))
    return CalibrationResult(
        x_opt=chain[idx].copy(),
        success=True,
        fun=float(lp[idx]),
        extra={"sampler": sampler, "chain": chain, "logp": lp},
    )
