import copy
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from CADETProcess.simulator import Cadet
from chromasurr.metrics import extract


def set_nested_attr(obj: object, attr_path: str, value: float | int) -> None:
    """
    Set a nested attribute on an object, including indexed lists.

    Parameters
    ----------
    obj : object
        The object whose attribute is to be set.
    attr_path : str
        A dot-separated string representing the path to the attribute.
        List indexing is supported (e.g., 'binding_model.adsorption_rate[0]').
    value : float or int
        The value to assign to the attribute.
    """
    parts = attr_path.split('.')
    for i, part in enumerate(parts[:-1]):
        if '[' in part:
            attr, idx = part[:-1].split('[')
            obj = getattr(obj, attr)[int(idx)]
        else:
            obj = getattr(obj, part)
    final = parts[-1]
    if '[' in final:
        attr, idx = final[:-1].split('[')
        getattr(obj, attr)[int(idx)] = value
    else:
        setattr(obj, final, value)


def run_sensitivity_analysis(
    process: object,
    param_config: dict[str, str],
    bounds: dict[str, list[float]],
    metric_names: list[str] = ["retention_time"],
    n_samples: int = 512
) -> dict[str, dict[str, np.ndarray]]:
    """
    Perform Sobol sensitivity analysis using a CADETProcess simulation.

    Parameters
    ----------
    process : object
        A CADET `Process` object representing the user's base model.
    param_config : dict[str, str]
        Mapping from parameter names to their attribute paths inside the process.
    bounds : dict[str, list[float]]
        Parameter bounds as [min, max] for each variable.
    metric_names : list[str], optional
        Names of the performance metrics to extract (default is ["retention_time"]).
    n_samples : int, optional
        Number of base samples for Saltelli's method (default is 512).
        The total number of simulations will be `n_samples * (2D + 2)`.

    Returns
    -------
    sobol_results : dict[str, dict[str, np.ndarray]]
        A dictionary mapping metric names to their respective Sobol analysis
        result dictionaries.
        Each contains keys like 'S1', 'ST', 'S2', and their confidence intervals.

    Raises
    ------
    KeyError
        If a metric name is not found in the extracted result.
    ValueError
        If simulation fails and metric extraction is not possible.
    """
    problem = {
        'num_vars': len(param_config),
        'names': list(param_config.keys()),
        'bounds': [bounds[name] for name in param_config]
    }

    param_values = saltelli.sample(problem, n_samples)
    metric_results = {m: [] for m in metric_names}

    for param_set in param_values:
        proc_copy = copy.deepcopy(process)

        for name, val in zip(problem['names'], param_set):
            set_nested_attr(proc_copy, param_config[name], val)

        try:
            sim = Cadet().simulate(proc_copy)
            metrics = extract(sim)
        except Exception as e:
            print(f"Simulation failed for {param_set}: {e}")
            metrics = {m: np.nan for m in metric_names}

        for m in metric_names:
            metric_results[m].append(metrics[m])

    for m in metric_names:
        metric_results[m] = np.array(metric_results[m])

    sobol_results = {}
    for m in metric_names:
        print(f"\n=== Sobol Analysis for '{m}' ===")
        Si = sobol.analyze(problem, metric_results[m], print_to_console=True)
        sobol_results[m] = Si

    return sobol_results
