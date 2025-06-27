import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from chromasurr.sensitivity import set_nested_attr
from chromasurr.metrics import extract
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from CADETProcess.simulator import Cadet
import copy
from typing import Dict, List, Optional


class Surrogate:
    """
    Class to manage surrogate modeling, sensitivity analysis, and model simplification
    for CADET-based chromatographic process simulations.
    """

    def __init__(
        self,
        process: object,
        param_config: Dict[str, str],
        bounds: Dict[str, List[float]],
        metrics: List[str],
        n_train: int = 128,
        kernel: Optional[object] = None,
        seed: int = 42
    ) -> None:
        """
        Initialize the Surrogate object.

        Parameters
        ----------
        process : object
            CADET process object.
        param_config : dict
            Mapping of parameter names to CADET attribute paths.
        bounds : dict
            Parameter bounds as [min, max] pairs.
        metrics : list
            List of metric names to model.
        n_train : int, optional
            Number of training samples. Default is 128.
        kernel : object, optional
            Custom kernel for Gaussian Process. Default is RBF.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        """
        self.process = process
        self.param_config = param_config
        self.bounds = bounds
        self.metrics = metrics
        self.n_train = n_train
        self.kernel = kernel
        self.seed = seed
        self.problem = {
            'num_vars': len(param_config),
            'names': list(param_config),
            'bounds': [bounds[k] for k in param_config]
        }
        self.models = {}
        self.Y = {}
        self.X = None
        self.top_params = list(param_config)
        self.sensitivity = {}

    def train(self) -> None:
        """Train Gaussian Process surrogate models using Sobol sampling."""
        np.random.seed(self.seed)
        X = sobol_sample.sample(self.problem, self.n_train)
        Y_dict = {m: [] for m in self.metrics}

        for x in X:
            proc = copy.deepcopy(self.process)
            for name, val in zip(self.problem['names'], x):
                set_nested_attr(proc, self.param_config[name], val)
            try:
                sim = Cadet().simulate(proc)
                metrics_out = extract(sim)
                for m in self.metrics:
                    Y_dict[m].append(metrics_out[m])
            except Exception as e:
                print(f"Simulation failed at {x}: {e}")
                for m in self.metrics:
                    Y_dict[m].append(np.nan)

        self.X = np.array(X)
        for m in self.metrics:
            y = np.array(Y_dict[m])
            valid = ~np.isnan(y)
            x_valid = self.X[valid]
            y_valid = y[valid]

            n_features = x_valid.shape[1]
            model_kernel = self.kernel or RBF(
                length_scale=np.ones(n_features), length_scale_bounds=(1e-6, 1e3)
                )
            gp = GaussianProcessRegressor(kernel=model_kernel, normalize_y=True)
            gp.fit(x_valid, y_valid)
            self.models[m] = gp
            self.Y[m] = y_valid

    def analyze_sensitivity(self, n_samples: int = 1024) -> None:
        """
        Perform Sobol sensitivity analysis using trained surrogate models.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for Sobol analysis. Default is 1024.
        """
        X = sobol_sample.sample(self.problem, n_samples)
        results = {}
        for m in self.metrics:
            Y_pred = self.models[m].predict(X)
            Si = sobol_analyze.analyze(self.problem, Y_pred, print_to_console=True)
            results[m] = Si
        self.sensitivity = results

    def select_important_params(self, threshold: float = 0.05) -> None:
        """
        Select parameters whose total Sobol index exceeds a threshold.

        Parameters
        ----------
        threshold : float, optional
            Sensitivity threshold for selection. Default is 0.05.
        """
        metric = self.metrics[0]  # assume single metric for now
        ST = self.sensitivity[metric]['ST']
        self.top_params = [
            n for n, st in zip(self.problem['names'], ST) if st >= threshold
            ]
        if not self.top_params:
            self.top_params = self.problem['names']

    def retrain(self) -> None:
        """Retrain surrogate models using only selected important parameters."""
        reduced_config = {k: self.param_config[k] for k in self.top_params}
        reduced_bounds = {k: self.bounds[k] for k in self.top_params}
        self.__init__(
            self.process, reduced_config, reduced_bounds,
            self.metrics, self.n_train, self.kernel, self.seed
        )
        self.train()

    def predict(self, metric: str, X: np.ndarray) -> np.ndarray:
        """
        Predict the specified metric using the trained surrogate model.

        Parameters
        ----------
        metric : str
            The name of the metric to predict.
        X : np.ndarray
            Array of input parameter sets.

        Returns
        -------
        np.ndarray
            Predicted values for the metric.
        """
        return self.models[metric].predict(X)
