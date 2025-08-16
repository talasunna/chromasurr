"""surrogate.py – Gaussian-process replacement of CADET simulations.

This module provides the :class:`Surrogate` class that can

1. sample the CADET parameter space with Saltelli’s method,
2. train one Gaussian-process (GP) emulator **per KPI** in log-space,
3. perform *cheap* Sobol sensitivity analysis *on* the GP,
4. keep only the most important parameters (user-defined criterion), and
5. retrain **without rerunning CADET**.

It also exposes :py:meth:`Surrogate.predict` and
:py:meth:`Surrogate.predict_var` so you can propagate uncertainty later
(e.g. with ``chromasurr.uq.perform_monte_carlo_uq``).
"""

import copy
import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample import saltelli as sobol_sample
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    WhiteKernel,
    RBF,
    ConstantKernel,
)
from sklearn.preprocessing import StandardScaler

from CADETProcess.processModel.process import Process
from CADETProcess.simulator import Cadet

from chromasurr.metrics import extract
from chromasurr.sensitivity import set_nested_attr

_logger = logging.getLogger(__name__)

__all__ = ["Surrogate"]

EPS: float = 1e-12  # lower bound to avoid log(0)
ArrayF = NDArray[np.float64]


class Surrogate:
    """Gaussian-process surrogate for a CADET chromatography workflow.

    Parameters
    ----------
    process : CADETProcess.processModel.process.Process
        Fully configured process model to emulate.
    param_config : dict[str, str]
        Mapping from *parameter names* to *attribute paths* on ``process``,
        e.g. ``{"ads_rate_A": "flow_sheet.column.binding_model.adsorption_rate[0]"}``.
    bounds : dict[str, Sequence[float]]
        Lower/upper bounds for each parameter – same keys/order as ``param_config``.
    metrics : list[str]
        Keys returned by :func:`chromasurr.metrics.extract` to emulate.
    n_train : int, default=128
        Saltelli *base* sample size.  Total sims ≈ ``n_train × (2D + 2)``.
    kernel : Kernel | type | None, optional
        scikit-learn kernel instance or class. Defaults to ``Matern(1.5)+White``.
    seed : int, default=0
        RNG seed (NumPy *and* scikit-learn).
    """

    def __init__(
        self,
        process: Process,
        param_config: dict[str, str],
        bounds: dict[str, list[float] | tuple[float, float]],
        metrics: list[str],
        n_train: int = 128,
        *,
        kernel: Kernel | type | None = None,
        seed: int = 0,
    ) -> None:
        self.process: Process = process
        self.param_config: dict[str, str] = param_config
        self.bounds: dict[str, list[float] | tuple[float, float]] = bounds
        self.metrics: list[str] = metrics
        self.n_train: int = n_train
        self.seed: int = seed

        if kernel is None:
            self.kernel: Kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0, length_scale_bounds=(1e-2, 10.0)
            ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-1))
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            self.kernel = kernel()

        self.problem: dict[str, Any] = {
            "num_vars": len(param_config),
            "names": list(param_config.keys()),
            "bounds": [list(bounds[name]) for name in param_config],
        }

        self.models: dict[str, GaussianProcessRegressor] = {}
        self._scaler: StandardScaler | None = None  # set in .train()
        self.X: ArrayF | None = None  # Saltelli design used for train
        self.Y: dict[str, ArrayF] = {}  # raw (not log) metric arrays

        self.sensitivity: dict[str, Any] = {}
        self.top_params: list[str] = list(self.problem["names"])

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

        Raises
        ------
        RuntimeError
            If no valid simulations are available to train on.
        ValueError
            If a particular metric has no finite data.
        """
        X_full: ArrayF = cast(ArrayF, sobol_sample.sample(self.problem, self.n_train))
        self.X = X_full.copy()

        raw_results: dict[str, list[float]] = {m: [] for m in self.metrics}
        cadet = Cadet()
        for sample in X_full:
            proc_copy = copy.deepcopy(self.process)

            for name, val in zip(self.problem["names"], sample.tolist(), strict=True):
                set_nested_attr(proc_copy, self.param_config[name], float(val))
            try:
                simulation_results = cadet.simulate(proc_copy)
                out = extract(simulation_results)
                for m in self.metrics:
                    raw_results[m].append(float(out[m]))
            except Exception as exc:
                _logger.warning("Simulation failed – setting NaNs: %s", exc)
                for m in self.metrics:
                    raw_results[m].append(np.nan)

        Y_raw: dict[str, ArrayF] = {
            m: np.asarray(v, dtype=float) for m, v in raw_results.items()
        }

        valid_all: NDArray[np.bool_] = np.ones(len(X_full), dtype=bool)
        for y in Y_raw.values():
            valid_all &= ~np.isnan(y)
        if not bool(valid_all.any()):
            msg = "No valid simulations – cannot train surrogate."
            raise RuntimeError(msg)

        self._scaler = StandardScaler().fit(X_full[valid_all])
        Xs_full: ArrayF = np.asarray(self._scaler.transform(X_full), dtype=float)

        for m in self.metrics:
            y_m: ArrayF = Y_raw[m]
            valid_m: NDArray[np.bool_] = (~np.isnan(y_m)) & valid_all
            if not bool(valid_m.any()):
                raise ValueError(f"Metric '{m}' had no finite data.")

            y_log: ArrayF = np.log(np.clip(y_m[valid_m], EPS, None))
            Xs: ArrayF = Xs_full[valid_m]

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
        """Compute global Sobol indices **via** the trained GP.

        Parameters
        ----------
        n_samples : int, default=1024
            Saltelli base sample size for the surrogate evaluation.
        metric : str or None, optional
            If ``None`` analyse **all** metrics and store under
            :pyattr:`sensitivity`; otherwise analyse only the given metric.
        log_space : bool, default=True
            Analyse on the GP training scale (``True``) or exponentiate the
            predictions first (``False``).
        print_to_console : bool, default=False
            Forward SALib’s textual summary to *stdout*?

        Raises
        ------
        RuntimeError
            If the surrogate has not been trained yet.
        """
        if self._scaler is None or not self.models:
            raise RuntimeError("Surrogate must be .train()-ed before use.")

        X_s: ArrayF = cast(ArrayF, sobol_sample.sample(self.problem, n_samples))
        Xs: ArrayF = np.asarray(self._scaler.transform(X_s), dtype=float)

        metrics_to_do = self.metrics if metric is None else [metric]
        for m in metrics_to_do:
            y_log: ArrayF = np.asarray(self.models[m].predict(Xs), dtype=float)
            y: ArrayF = y_log if log_space else np.asarray(np.exp(y_log), dtype=float)
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
    ) -> list[str]:
        """Pick *key* parameters based on total Sobol indices.

        Exactly **one** of *threshold* or *n_top* can be given.  The default
        keeps parameters with ``ST ≥ 0.05`` *for the first metric*.

        Parameters
        ----------
        metric : str or None, optional
            Which metric’s indices to use; defaults to the first metric.
        threshold : float or None, default=0.05
            Keep parameters with ``ST >= threshold``.
        n_top : int or None, optional
            If provided, keep exactly the top-``n_top`` parameters by ``ST``.

        Returns
        -------
        list[str]
            The retained parameter names.

        Raises
        ------
        ValueError
            If both or neither of ``threshold`` and ``n_top`` are provided.
        RuntimeError
            If :meth:`analyze_sensitivity` has not been called yet.
        """
        if (threshold is None) == (n_top is None):
            raise ValueError("Specify either 'threshold' or 'n_top', not both.")
        if not self.sensitivity:
            raise RuntimeError("Call .analyze_sensitivity() first.")

        metric_to_use = metric or self.metrics[0]
        ST: ArrayF = np.asarray(self.sensitivity[metric_to_use]["ST"], dtype=float)

        if n_top is not None:
            idx = np.argsort(ST)[::-1][:n_top]
            self.top_params = [self.problem["names"][int(i)] for i in idx]
        else:
            if threshold is None:
                raise AssertionError("threshold cannot be None in threshold mode")
            thr: float = float(threshold)
            self.top_params = [
                name
                for name, st in zip(self.problem["names"], ST.tolist(), strict=True)
                if float(st) >= thr
            ]
            if not self.top_params:  # guarantee ≥1 parameter
                self.top_params = [self.problem["names"][int(np.argmax(ST))]]

        return self.top_params

    def retrain(self) -> None:
        """Retrain **without** new CADET runs, using only ``top_params``."""
        if self.X is None:
            raise RuntimeError("Call .train() before .retrain().")
        if not self.top_params:
            _logger.warning("No parameters selected – nothing to retrain.")
            return

        cols = [self.problem["names"].index(p) for p in self.top_params]

        self.X = cast(ArrayF, self.X[:, cols])
        self.bounds = {p: self.bounds[p] for p in self.top_params}
        self.problem["names"] = self.top_params
        self.problem["bounds"] = [self.bounds[p] for p in self.top_params]
        self.problem["num_vars"] = len(self.top_params)

        self._scaler = StandardScaler().fit(self.X)
        Xs: ArrayF = np.asarray(self._scaler.transform(self.X), dtype=float)

        for m in self.metrics:
            y_raw: ArrayF = self.Y[m]
            y_log: ArrayF = np.log(np.clip(y_raw, EPS, None))
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                normalize_y=True,
                random_state=self.seed,
            )
            gp.fit(Xs, y_log)
            self.models[m] = gp

    def _transform(self, X: ArrayF) -> ArrayF:
        """Scale input features with the fitted :class:`StandardScaler`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_params)
            Unscaled input design.

        Returns
        -------
        ndarray
            Scaled design with the same shape as ``X``.

        Raises
        ------
        RuntimeError
            If the surrogate has not been trained yet.
        """
        if self._scaler is None:
            raise RuntimeError("Surrogate not yet trained – no scaler.")
        return np.asarray(self._scaler.transform(X), dtype=float)

    def predict(self, metric: str, X: ArrayF) -> ArrayF:
        """Predict **mean** of a KPI on the original scale.

        Parameters
        ----------
        metric : str
            Which KPI (metric name) to return.
        X : ndarray, shape (n_samples, n_params)
            Input design.

        Returns
        -------
        ndarray
            Mean predictions on the original (exp) scale.
        """
        Xs = self._transform(X)
        y_log, _ = self.models[metric].predict(Xs, return_std=True)
        return np.asarray(np.exp(y_log), dtype=float)

    def predict_var(self, metric: str, X: ArrayF) -> ArrayF:
        """Predict the **variance** of a KPI on the original scale.

        The GP is trained on log-responses.  For a log-normal variable
        ``Y = exp(Z)``, with ``Z ~ N(μ, σ²)``, the variance is

        .. math::

            \\operatorname{Var}[Y] = (e^{\\sigma^2} - 1)\\, e^{2\\mu + \\sigma^2}

        Parameters
        ----------
        metric : str
            Which KPI (metric name) to return variance for.
        X : ndarray, shape (n_samples, n_params)
            Input design.

        Returns
        -------
        ndarray
            Variance predictions on the original scale.
        """
        Xs = self._transform(X)
        mu, std = self.models[metric].predict(Xs, return_std=True)
        std = np.clip(np.asarray(std, dtype=float), 1e-6, None)
        mu = np.asarray(mu, dtype=float)
        var = (np.exp(std**2) - 1.0) * np.exp(2.0 * mu + std**2)
        return np.asarray(var, dtype=float)

    def __repr__(self) -> str:
        return (
            f"Surrogate(metrics={self.metrics}, "
            f"params={self.problem['names']}, "
            f"trained={bool(self.models)})"
        )
