"""surrogate.py – Gaussian‑process replacement of CADET simulations.

This module provides the :class:`Surrogate` class that can

1. sample the CADET parameter space with Saltelli’s method,
2. train one Gaussian‑process (GP) emulator **per KPI** in log‑space,
3. perform *cheap* Sobol sensitivity analysis *on* the GP,
4. keep only the most important parameters (user‑defined criterion), and
5. retrain **without rerunning CADET**.

It also exposes :py:meth:`Surrogate.predict` and
:py:meth:`Surrogate.predict_var` so you can propagate uncertainty later
(e.g. with ``chromasurr.uq.perform_monte_carlo_uq``).

All public methods carry Numpy‑style docstrings and full Python ≥3.10
static type hints.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Union, overload

import numpy as np
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample import saltelli as sobol_sample
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from CADETProcess.processModel.process import Process
from CADETProcess.simulator import Cadet

from .metrics import extract  # noqa: F401 – relative import inside package
from .sensitivity import set_nested_attr

_logger = logging.getLogger(__name__)

__all__ = ["Surrogate"]

EPS: float = 1e-12  # lower bound to avoid log(0)


class Surrogate:
    """Gaussian‑process surrogate for a CADET chromatography workflow.

    Parameters
    ----------
    process
        Fully configured :class:`CADETProcess.processModel.process.Process`.
    param_config
        Mapping from **human‑readable** parameter names to *attribute paths*
        on *process*, e.g. ``{"ads_rate_A": "flow_sheet.column.binding_model.adsorption_rate[0]"}``.
    bounds
        Lower/upper bounds for each parameter – same keys/order as
        *param_config*.
    metrics
        Keys returned by :func:`chromasurr.metrics.extract` to emulate.
    n_train
        Saltelli *base* sample size.  Total sims ≈ ``n_train × (2D + 2)``.
    kernel
        scikit‑learn kernel or class.  Defaults to ``Matern(1.5)+White``.
    seed
        RNG seed (NumPy *and* scikit‑learn).
    """

    def __init__(
        self,
        process: Process,
        param_config: Dict[str, str],
        bounds: Dict[str, Sequence[float]],
        metrics: List[str],
        n_train: int = 128,
        *,
        kernel: Kernel | type | None = None,
        seed: int = 0,
    ) -> None:
        self.process: Process = process
        self.param_config: Dict[str, str] = param_config
        self.bounds: Dict[str, Sequence[float]] = bounds
        self.metrics: List[str] = metrics
        self.n_train: int = n_train
        self.seed: int = seed

        if kernel is None:
            self.kernel: Kernel = (
                Matern(nu=1.5, length_scale_bounds=(1e-3, 1e3))
                + WhiteKernel(noise_level_bounds=(1e-6, 1e1))
            )
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            self.kernel = kernel()

        self.problem: Dict[str, Any] = {
            "num_vars": len(param_config),
            "names": list(param_config.keys()),
            "bounds": [list(bounds[name]) for name in param_config],
        }

        self.models: Dict[str, GaussianProcessRegressor] = {}
        self._scaler: Optional[StandardScaler] = None  # set in .train()
        self.X: Optional[np.ndarray] = None  # Saltelli design used for train
        self.Y: Dict[str, np.ndarray] = {}  # raw (not log) metric arrays

        self.sensitivity: Dict[str, Any] = {}
        self.top_params: List[str] = list(self.problem["names"])

        np.random.seed(self.seed)

    def train(self) -> None:
        """Run CADET on a Saltelli design and fit one GP **per metric**.

        The method

        1. draws a Saltelli sample ``X_full`` with *n_train* base samples,
        2. simulates CADET for each sample and extracts *metrics*,
        3. builds a *single* :class:`sklearn.preprocessing.StandardScaler`
           on the **intersection** of rows valid for *all* metrics, and
        4. fits a :class:`sklearn.gaussian_process.GaussianProcessRegressor`
           on ``log(y)`` for each metric.
        """
        X_full: np.ndarray = sobol_sample.sample(self.problem, self.n_train)
        self.X = X_full.copy()

        raw_results: Dict[str, List[float]] = {m: [] for m in self.metrics}
        cadet = Cadet()
        for sample in X_full:
            proc_copy = copy.deepcopy(self.process)

            for name, val in zip(self.problem["names"], sample):
                set_nested_attr(proc_copy, self.param_config[name], val)
            try:
                simulation_results = Cadet().simulate(proc_copy)
                out = extract(simulation_results)  # user function
                for m in self.metrics:
                    raw_results[m].append(float(out[m]))
            except Exception as exc:  # noqa: BLE001 – broad but logged
                _logger.warning("Simulation failed – setting NaNs: %s", exc)
                for m in self.metrics:
                    raw_results[m].append(np.nan)

        Y_raw = {m: np.asarray(v, dtype=float) for m, v in raw_results.items()}

        valid_all: np.ndarray = np.ones(len(X_full), dtype=bool)
        for y in Y_raw.values():
            valid_all &= ~np.isnan(y)
        if not valid_all.any():
            msg = "No valid simulations – cannot train surrogate."
            raise RuntimeError(msg)

        self._scaler = StandardScaler().fit(X_full[valid_all])
        Xs_full: np.ndarray = self._scaler.transform(X_full)

        for m in self.metrics:
            y_m: np.ndarray = Y_raw[m]
            valid_m: np.ndarray = (~np.isnan(y_m)) & valid_all
            if not valid_m.any():
                raise ValueError(f"Metric '{m}' had no finite data.")

            y_log = np.log(np.clip(y_m[valid_m], EPS, None))
            Xs = Xs_full[valid_m]

            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                normalize_y=True,
                random_state=self.seed,
            )
            gp.fit(Xs, y_log)

            self.models[m] = gp
            self.Y[m] = y_m[valid_m]

    def analyze_sensitivity(
        self,
        *,
        n_samples: int = 1024,
        metric: str | None = None,
        log_space: bool = True,
        print_to_console: bool = False,
    ) -> None:
        """Global Sobol indices *via* the trained GP.

        Parameters
        ----------
        n_samples
            Saltelli base sample size for the surrogate evaluation.
        metric
            If *None* analyse **all** metrics and store under
            :pyattr:`sensitivity`; otherwise analyse only *metric*.
        log_space
            Analyse on the GP training scale (``True``) or exponentiate the
            predictions first (``False``).
        print_to_console
            Forward *SALib*’s textual summary to *stdout*?
        """
        if self._scaler is None or not self.models:
            raise RuntimeError("Surrogate must be .train()‑ed before use.")

        X_s = sobol_sample.sample(self.problem, n_samples)
        Xs = self._scaler.transform(X_s)

        metrics_to_do = self.metrics if metric is None else [metric]
        for m in metrics_to_do:
            y_log = self.models[m].predict(Xs)
            y = y_log if log_space else np.exp(y_log)
            Si = sobol_analyze(
                self.problem,
                y,
                print_to_console=print_to_console,
            )
            self.sensitivity[m] = Si

    def select_important_params(
        self,
        *,
        metric: str | None = None,
        threshold: float | None = 0.05,
        n_top: int | None = None,
    ) -> List[str]:
        """Pick *key* parameters based on total Sobol indices.

        Exactly **one** of *threshold* or *n_top* can be given.  The default
        keeps parameters with ``ST ≥ 0.05`` *for the first metric*.

        Returns
        -------
        list[str]
            The retained parameter names.
        """
        if (threshold is None) == (n_top is None):
            raise ValueError("Specify either 'threshold' or 'n_top', not both.")
        if not self.sensitivity:
            raise RuntimeError("Call .analyze_sensitivity() first.")

        metric_to_use = metric or self.metrics[0]
        ST: np.ndarray = self.sensitivity[metric_to_use]["ST"]  # type: ignore[index]

        if n_top is not None:
            idx = np.argsort(ST)[::-1][: n_top]
            self.top_params = [self.problem["names"][i] for i in idx]
        else:  # threshold mode
            self.top_params = [
                name
                for name, st in zip(self.problem["names"], ST, strict=True)
                if st >= threshold
            ]
            if not self.top_params:  # guarantee ≥1 parameter
                self.top_params = [self.problem["names"][int(np.argmax(ST))]]

        return self.top_params

    def retrain(self) -> None:
        """Retrain **without** new CADET runs, using only *top_params*."""
        if self.X is None:
            raise RuntimeError("Call .train() before .retrain().")
        if not self.top_params:
            _logger.warning("No parameters selected – nothing to retrain.")
            return

        cols = [self.problem["names"].index(p) for p in self.top_params]

        self.X = self.X[:, cols]
        self.problem["names"] = self.top_params
        self.problem["bounds"] = [self.bounds[p] for p in self.top_params]
        self.problem["num_vars"] = len(self.top_params)

        self._scaler = StandardScaler().fit(self.X)
        Xs = self._scaler.transform(self.X)

        for m in self.metrics:
            y_raw = self.Y[m]
            y_log = np.log(np.clip(y_raw, EPS, None))
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                normalize_y=True,
                random_state=self.seed,
            )
            gp.fit(Xs, y_log)
            self.models[m] = gp

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Surrogate not yet trained – no scaler.")
        return self._scaler.transform(X)

    def predict(self, metric: str, X: np.ndarray) -> np.ndarray:
        """Predict **mean** of *metric* on original scale.

        Parameters
        ----------
        metric
            Which KPI to return.
        X
            Array of shape ``(n_samples, n_params)``.
        """
        Xs = self._transform(X)
        y_log, _ = self.models[metric].predict(Xs, return_std=True)
        return np.exp(y_log)

    def predict_var(self, metric: str, X: np.ndarray) -> np.ndarray:
        """Predict **variance** of *metric* (original scale).

        Formula for a log‑normal variable ``Y = exp(Z),  Z~N(μ, σ²)`` ::

            Var[Y] = (e^{σ²} - 1) · e^{2μ + σ²}

        See e.g. *Crow & Shimizu (1988), Lognormal Distributions*.
        """
        Xs = self._transform(X)
        _, std = self.models[metric].predict(Xs, return_std=True)
        var = (np.exp(std**2) - 1.0) * np.exp(2.0 * self.models[metric].predict(Xs) + std**2)
        return var

    def __repr__(self) -> str:
        return (
            f"Surrogate(metrics={self.metrics}, "
            f"params={self.problem['names']}, "
            f"trained={bool(self.models)})"
        )
