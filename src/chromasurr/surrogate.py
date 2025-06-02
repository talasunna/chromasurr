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
) -> tuple[dict[str, GaussianProcessRegressor], dict[str, np.ndarray], np.ndarray, SALibProblem]:
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
    return model.predict(X)


def run_multi_surrogate_sensitivity_analysis(
    models: dict[str, GaussianProcessRegressor],
    problem: SALibProblem,
    n_samples: int = 1024
) -> dict[str, dict[str, np.ndarray]]:
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


def select_top_parameters_by_sensitivity(
   sobol_result: dict[str, np.ndarray],
   param_names: list[str],
   threshold: float = 0.05
) -> list[str]:
     """
     Select parameter names whose total Sobol index (ST) exceeds a threshold.

     Parameters
     ----------
     sobol_result : dict
         Sobol result for a given metric, as returned by SALib.
     param_names : list[str]
         List of parameter names in the original problem.
     threshold : float
         Minimum ST value to keep a parameter.

     Returns
     -------
     list[str]
         Selected parameter names.
     """
     ST = sobol_result['ST']
     return [name for name, st in zip(param_names, ST) if st >= threshold]



def retrain_surrogate_on_selected_parameters(
    process: object,
    full_param_config: dict[str, str],
    bounds: dict[str, list[float]],
    selected_params: list[str],
    metrics: list[str],
    n_train: int = 128
) -> tuple[dict[str, GaussianProcessRegressor], dict[str, np.ndarray], np.ndarray, SALibProblem]:
    """
    Retrain surrogate model using only the selected sensitive parameters.

    Parameters
    ----------
    process : object
        CADET Process object.
    full_param_config : dict
        Full parameter config with paths.
    bounds : dict
        Parameter bounds.
    selected_params : list[str]
        Selected sensitive parameter names.
    metrics : list[str]
        Metrics to model.
    n_train : int
        Number of training samples.

    Returns
    -------
    Same as `train_multi_surrogate_models`, but only for reduced parameter set.
    """
    reduced_param_config = {k: full_param_config[k] for k in selected_params}
    reduced_bounds = {k: bounds[k] for k in selected_params}
    return train_multi_surrogate_models(
        process=process,
        param_config=reduced_param_config,
        bounds=reduced_bounds,
        metrics=metrics,
        n_train=n_train
    )
