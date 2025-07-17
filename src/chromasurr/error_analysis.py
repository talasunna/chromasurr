"""Diagnostic tools for surrogate or calibrated model.

This module collects lightweight helpers that assess how well a surrogate
(e.g., a Gaussian‑process emulator) reproduces reference data.  In addition to
the classic *RMSE*, *MAE* and coefficient of determination (*R²*), we provide

* **NRMSE** – Normalised root‑mean‑square error (normalised by either the data
  range, mean or standard deviation)
* **MAPE** – Mean absolute percentage error

A convenience wrapper :pyfunc:`evaluate` prints a traffic‑light style report so
users can immediately gauge surrogate quality in notebooks or scripts.

"""
from __future__ import annotations

from typing import Dict, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "basic_stats",
    "nrmse",
    "mape",
    "evaluate",
    "parity_plot",
]


def basic_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return basic point‑wise error statistics.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Arrays of equal length containing reference and predicted values.

    Returns
    -------
    dict
        *RMSE*, *MAE* and *R²* in a dictionary.
    """
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    norm: Literal["range", "mean", "std"] = "range",
) -> float:
    """Compute the *normalised* RMSE.

    The RMSE is divided by one of three scale factors:

    * ``"range"`` – max(y_true) − min(y_true)
    * ``"mean"`` – mean(y_true)
    * ``"std"``  – population standard deviation of y_true

    Normalising puts error into perspective and enables comparison across
    metrics with different units.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Reference and predicted data.
    norm : {'range', 'mean', 'std'}, default='range'
        Scaling to use.

    Returns
    -------
    float
        NRMSE in relative units (0 ⟹ perfect fit, higher is worse).
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if norm == "range":
        denom = float(np.ptp(y_true))  # max − min
    elif norm == "mean":
        denom = float(np.mean(y_true))
    elif norm == "std":
        denom = float(np.std(y_true, ddof=0))
    else:
        raise ValueError("norm must be 'range', 'mean' or 'std'")

    if denom == 0:
        return float("nan")
    return rmse / denom


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error.

    ``MAPE = mean(|y_true − y_pred| / |y_true|)`` expressed as a fraction.

    Returns *nan* if any true value is zero, as the percentage error would be
    undefined.
    """
    if np.any(y_true == 0):
        return float("nan")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    nrmse_threshold: float = 0.10,
    mape_threshold: float = 0.05,
    norm: Literal["range", "mean", "std"] = "range",
) -> Dict[str, Tuple[float, bool]]:
    """Comprehensive error report with *traffic‑light* flags.

    Any metric that meets the respective threshold is marked ✅ (good), others
    ❌ (poor).  The verdict is the conjunction of all individual flags.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Arrays with reference and predicted values.
    nrmse_threshold : float, default=0.10
        Acceptable NRMSE (relative units).
    mape_threshold : float, default=0.05
        Acceptable MAPE (fractional).
    norm : {'range', 'mean', 'std'}, default='range'
        Scaling used for NRMSE.

    Returns
    -------
    dict
        Mapping from metric name to *(value, flag)* where *flag* is *True* if
        the surrogate passes the criterion.  An additional *verdict* key holds
        a human‑readable string.
    """
    stats = basic_stats(y_true, y_pred)
    stats["NRMSE"] = nrmse(y_true, y_pred, norm=norm)
    stats["MAPE"] = mape(y_true, y_pred)

    flags: Dict[str, Tuple[float, bool]] = {}
    for name, value in stats.items():
        if name == "NRMSE":
            ok = value <= nrmse_threshold
        elif name == "MAPE":
            ok = value <= mape_threshold
        else:  # heuristic: lower error metrics good, higher R2 good
            ok = value <= 0.1 if name in {"RMSE", "MAE"} else value >= 0.9
        flags[name] = (value, ok)

    all_good = all(ok for _, ok in flags.values())
    flags["verdict"] = (
        "✅ Good: surrogate error comfortably below target\n"
        if all_good
        else "❌ Warning: surrogate error above acceptable limits\n"
    )

    return flags


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Scatter plot of predicted vs. observed values.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Reference and predicted data.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.  If *None*, a new figure/axes is created.
    label : str, optional
        Legend label for the scatter points.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the data was plotted on.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", label=label)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "--", lw=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    if label is not None:
        ax.legend()

    return ax
