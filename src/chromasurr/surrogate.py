import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, Kernel
from sklearn.preprocessing import StandardScaler
from SALib.sample import saltelli as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze
from CADETProcess.simulator import Cadet
from CADETProcess.processModel.process import Process
from chromasurr.sensitivity import set_nested_attr
from chromasurr.metrics import extract


class Surrogate:
    """
    Surrogate model manager for CADET-based chromatography simulations.

    Builds a Gaussian process emulator for user-specified metrics by sampling
    the parameter space with Saltelli's method, training in log-space, and
    offering sensitivity analysis and parameter reduction.

    Parameters
    ----------
    process : Process
        A CADETProcess `Process` object defining the chromatographic workflow.
    param_config : dict[str, str]
        Mapping from parameter names to attribute paths on `process`.
    bounds : dict[str, list[float]]
        Lower and upper bounds for each parameter (keys match `param_config`).
    metrics : list[str]
        Names of the metrics to model (must match keys of `extract(sim)`).
    n_train : int, default=128
        Number of Saltelli base samples for training.
    kernel : Kernel or Kernel class, optional
        A scikit-learn kernel instance or class. If None, defaults to Matern(nu=1.5)+WhiteKernel().
    seed : int, default=0
        Random seed for reproducibility.
    """
    def __init__(
        self,
        process: Process,
        param_config: Dict[str, str],
        bounds: Dict[str, List[float]],
        metrics: List[str],
        n_train: int = 128,
        kernel: Optional[Union[Kernel, type]] = None,
        seed: int = 0
    ) -> None:
        # Kernel default: Matern for flexibility + white noise for numerical stability
        if kernel is None:
            self.kernel = Matern(nu=1.5, length_scale_bounds=(1e-3, 1e3)) \
                        + WhiteKernel(noise_level_bounds=(1e-6, 1e1))
        elif isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            # assume kernel is a class
            self.kernel = kernel()

        self.process: Process = process
        self.param_config: Dict[str, str] = param_config
        self.bounds: Dict[str, List[float]] = bounds
        self.metrics: List[str] = metrics
        self.n_train: int = n_train
        self.seed: int = seed

        # SALib problem definition
        self.problem: Dict[str, Any] = {
            'num_vars': len(param_config),
            'names': list(param_config.keys()),
            'bounds': [bounds[name] for name in param_config]
        }

        # Storage
        self.models: Dict[str, GaussianProcessRegressor] = {}
        self.Y: Dict[str, np.ndarray] = {}
        self.X: Optional[np.ndarray] = None
        self.top_params: List[str] = list(param_config.keys())
        self.sensitivity: Dict[str, Any] = {}
        self._scaler: Optional[StandardScaler] = None

    def train(self) -> None:
        """
        Train Gaussian process surrogates on Saltelli-sampled simulations.

        Samples the parameter space, runs CADET for each sample, extracts metrics,
        log-transforms them, scales inputs, and fits a separate GP for each metric.
        """
        np.random.seed(self.seed)
        X_full = sobol_sample.sample(self.problem, self.n_train)
        results: Dict[str, List[float]] = {m: [] for m in self.metrics}

        for sample in X_full:
            proc_copy = copy.deepcopy(self.process)
            for name, val in zip(self.problem['names'], sample):
                set_nested_attr(proc_copy, self.param_config[name], val)
            try:
                sim = Cadet().simulate(proc_copy)
                out = extract(sim)
                for m in self.metrics:
                    results[m].append(out[m])
            except Exception:
                for m in self.metrics:
                    results[m].append(np.nan)

        self.X = np.array(X_full)
        for m in self.metrics:
            y_raw = np.array(results[m])
            valid = ~np.isnan(y_raw)
            X_valid = self.X[valid]
            y_valid = np.log(y_raw[valid])

            # Scale inputs to zero-mean, unit-variance
            scaler = StandardScaler().fit(X_valid)
            Xs = scaler.transform(X_valid)
            self._scaler = scaler

            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                normalize_y=True,
                random_state=self.seed
            )
            gp.fit(Xs, y_valid)

            self.models[m] = gp
            self.Y[m] = y_raw[valid]

    def analyze_sensitivity(self, n_samples: int = 1024) -> None:
        """
        Perform Sobol sensitivity analysis using the trained surrogate.

        Uses the surrogate in log-space at new Saltelli samples.
        """
        X_s = sobol_sample.sample(self.problem, n_samples)
        results: Dict[str, Any] = {}
        for m in self.metrics:
            # apply scaling
            Xs = self._scaler.transform(X_s)
            y_log = self.models[m].predict(Xs)
            Si = sobol_analyze(self.problem, y_log, print_to_console=True)
            results[m] = Si
        self.sensitivity = results

    def select_important_params(self, threshold: float = 0.05) -> None:
        """
        Retain only parameters whose total Sobol index meets the threshold.
        """
        metric = self.metrics[0]
        ST = self.sensitivity[metric]['ST']
        self.top_params = [
            name for name, st in zip(self.problem['names'], ST) if st >= threshold
        ]
        if not self.top_params:
            self.top_params = list(self.problem['names'])

    def retrain(self) -> None:
        """
        Retrain surrogate using only the selected top parameters.
        """
        reduced_config = {k: self.param_config[k] for k in self.top_params}
        reduced_bounds = {k: self.bounds[k] for k in self.top_params}
        self.__init__(
            self.process,
            reduced_config,
            reduced_bounds,
            self.metrics,
            self.n_train,
            kernel=self.kernel,
            seed=self.seed
        )
        self.train()

    def predict(self, metric: str, X: np.ndarray) -> np.ndarray:
        """
        Predict the mean of the specified metric on the original scale.

        Parameters
        ----------
        metric : str
            Name of the metric to predict.
        X : numpy.ndarray
            New input parameter matrix of shape (n_samples, n_params).

        Returns
        -------
        numpy.ndarray
            Predicted metric values, shape (n_samples,).
        """
        # scale new inputs
        Xs = self._scaler.transform(X)  # type: ignore
        y_log_mean, _ = self.models[metric].predict(Xs, return_std=True)
        return np.exp(y_log_mean)
