import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from chromasurr.sensitivity import set_nested_attr
from chromasurr.metrics import extract
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from CADETProcess.simulator import Cadet
import copy
from typing import  Optional


class Surrogate:
    """
    Class to manage surrogate modeling, sensitivity analysis, and model simplification
    for CADET-based chromatographic process simulations.
    """

    def __init__(
        self,
        process: object,
        param_config: dict[str, str],
        bounds: dict[str, list[float]],
        metrics: list[str],
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
