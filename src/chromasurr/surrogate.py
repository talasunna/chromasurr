import numpy as np
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from CADETProcess.simulator import Cadet
from chromasurr.metrics import extract
from chromasurr.sensitivity import set_nested_attr
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from typing import TypedDict


class SALibProblem(TypedDict):
    num_vars: int
    names: list[str]
    bounds: list[list[float]]


def train_multi_surrogate_models(
    process: object,
    param_config: dict[str, str],
    bounds: dict[str, list[float]],
    metrics: list[str],
    n_train: int = 128,
    kernel=None,
    seed: int = 42
) -> tuple[dict[str, GaussianProcessRegressor], dict[str, np.ndarray],
           np.ndarray, SALibProblem]:
    """
    Train one surrogate model per metric using shared CADET simulations.

    Parameters
    ----------
    process : object
        CADET process template.
    param_config : dict
        Parameter name to path inside the process.
    bounds : dict
        Parameter name to [min, max] range.
    metrics : list of str
        Metric names to extract from the simulation results.
    n_train : int
        Number of training samples.
    kernel : sklearn kernel or None
        Optional kernel for Gaussian Process. Defaults to RBF.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    models : dict[str, GaussianProcessRegressor]
        One surrogate per metric.
    Y_trains : dict[str, ndarray]
        Training outputs per metric.
    X_train : ndarray
        Shared training input matrix.
    problem : SALibProblem
        Problem dictionary defining the SALib structure.
    """
    np.random.seed(seed)
    problem: SALibProblem = {
        'num_vars': len(param_config),
        'names': list(param_config),
        'bounds': [bounds[name] for name in param_config]
    }

    X_train = sobol.sample(problem, n_train)
    Y_trains = {m: [] for m in metrics}

    for x in X_train:
        proc_copy = copy.deepcopy(process)
        for name, val in zip(problem['names'], x):
            set_nested_attr(proc_copy, param_config[name], val)

        try:
            sim = Cadet().simulate(proc_copy)
            extracted = extract(sim)
            for m in metrics:
                Y_trains[m].append(extracted[m])
        except Exception as e:
            print(f"Simulation failed at {x}: {e}")
            for m in metrics:
                Y_trains[m].append(np.nan)

    X_train = np.array(X_train)
    models = {}
    final_Y_trains = {}

    for m in metrics:
        y = np.array(Y_trains[m])
        valid = ~np.isnan(y)
        x_valid = X_train[valid]
        y_valid = y[valid]

        model_kernel = kernel or RBF(length_scale=np.ones(x_valid.shape[1]))
        gp = GaussianProcessRegressor(kernel=model_kernel, normalize_y=True)
        gp.fit(x_valid, y_valid)

        models[m] = gp
        final_Y_trains[m] = y_valid

    return models, final_Y_trains, X_train, problem


def predict_surrogate(
    model: GaussianProcessRegressor,
    X: np.ndarray
) -> np.ndarray:
    """
    Predict surrogate outputs for new inputs.

    Parameters
    ----------
    model : GaussianProcessRegressor
        Trained surrogate.
    X : ndarray
        New input matrix.

    Returns
    -------
    ndarray
        Predicted output vector.
    """
    return model.predict(X)


def run_multi_surrogate_sensitivity_analysis(
    models: dict[str, GaussianProcessRegressor],
    problem: SALibProblem,
    n_samples: int = 1024
) -> dict[str, dict[str, np.ndarray]]:
    """
    Run SALib Sobol analysis on multiple surrogate models.

    Parameters
    ----------
    models : dict
        Surrogate models keyed by metric name.
    problem : dict
        SALib problem definition (names, bounds).
    n_samples : int
        Number of base samples (Saltelli will expand it internally).

    Returns
    -------
    sobol_results : dict[str, dict]
        Sensitivity indices per metric.
    """
    X = sobol.sample(problem, n_samples)
    results = {}

    for metric, model in models.items():
        Y = predict_surrogate(model, X)
        Si = sobol_analyze.analyze(problem, Y, print_to_console=True)
        results[metric] = {
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf'],
            'S2': Si['S2'],
            'S2_conf': Si['S2_conf'],
        }

    return results
