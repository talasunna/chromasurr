from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray
from SALib.analyze import sobol
from SALib.sample import saltelli
from CADETProcess.simulator import Cadet
from chromasurr.metrics import extract

ArrayF = NDArray[np.float64]


class _ProblemSpec(TypedDict):
    """TypedDict for SALib problem specification."""

    num_vars: int
    names: list[str]
    bounds: list[list[float]]


def set_nested_attr(obj: object, attr_path: str, value: float | int) -> None:
    """
    Set a nested attribute on an object, including indexed lists.

    Parameters
    ----------
    obj : object
        The object whose attribute is to be set.
    attr_path : str
        Dot-separated path to the attribute. Indexing supported, e.g.
        ``"binding_model.adsorption_rate[0]"``.
    value : float or int
        The value to assign to the attribute.

    Notes
    -----
    This walks the object graph following the components in ``attr_path``.
    When a component looks like ``name[index]``, it indexes the list/sequence.
    """
    parts = attr_path.split(".")
    for part in parts[:-1]:
        if "[" in part:
            attr, idx = part[:-1].split("[")
            obj = getattr(obj, attr)[int(idx)]
        else:
            obj = getattr(obj, part)
    final = parts[-1]
    if "[" in final:
        attr, idx = final[:-1].split("[")
        getattr(obj, attr)[int(idx)] = value
    else:
        setattr(obj, final, value)


def run_sensitivity_analysis(
    process: Any,
    param_config: Mapping[str, str],
    bounds: Mapping[str, Sequence[float]],
    metric_names: Sequence[str] | None = None,
    n_samples: int = 512,
) -> dict[str, dict[str, ArrayF | float]]:
    """
    Perform Sobol sensitivity analysis using a CADETProcess simulation.

    Parameters
    ----------
    process : Any
        A CADET ``Process`` object representing the base model (type is untyped
        because CADET does not ship type hints).
    param_config : Mapping[str, str]
        Maps *parameter name* → *attribute path* inside the process object.
        Example path: ``"binding_model.adsorption_rate[0]"``.
    bounds : Mapping[str, Sequence[float]]
        Parameter bounds as ``[min, max]`` per parameter name.
    metric_names : Sequence[str] or None, optional
        Which metrics to extract via :py:func:`chromasurr.metrics.extract`.
        Defaults to ``["retention_time"]``.
    n_samples : int, optional
        Base sample size for Saltelli; total sims ~ ``n_samples * (2D + 2)``.

    Returns
    -------
    dict[str, dict[str, ArrayF | float]]
        Mapping ``metric_name`` → Sobol result dict from SALib. Each dict
        typically contains keys like ``"S1"``, ``"ST"``, ``"S2"`` and their
        confidence intervals (some values are arrays, some scalars).

    Raises
    ------
    KeyError
        If a requested metric is not present in extracted results.
    ValueError
        If simulation repeatedly fails and no metrics can be produced.
    """
    # Default metrics without using a mutable default argument
    if metric_names is None:
        metric_names = ["retention_time"]

    problem: _ProblemSpec = {
        "num_vars": len(param_config),
        "names": list(param_config.keys()),
        "bounds": [list(bounds[name]) for name in param_config],
    }

    # SALib returns an ndarray of shape (N, D)
    param_values: ArrayF = saltelli.sample(problem, n_samples)

    # Collect raw metric values as lists, then convert to arrays
    metric_results: dict[str, list[float]] = {m: [] for m in metric_names}

    for param_set in param_values:
        proc_copy = copy.deepcopy(process)

        # zip() needs well-typed iterables → .tolist() gives list[float]
        for name, val in zip(problem["names"], param_set.tolist()):
            set_nested_attr(proc_copy, param_config[name], float(val))

        try:
            sim: Any = Cadet().simulate(proc_copy)
            metrics: dict[str, float] = extract(sim)
        except Exception as e:  # pragma: no cover
            # Fallback to NaN if a simulation fails; keep length consistent
            print(f"Simulation failed for {param_set}: {e}")
            metrics = {m: float("nan") for m in metric_names}

        for m in metric_names:
            metric_results[m].append(float(metrics[m]))

    metric_arrays: dict[str, ArrayF] = {
        m: np.asarray(vals, dtype=float) for m, vals in metric_results.items()
    }

    sobol_results: dict[str, dict[str, ArrayF | float]] = {}
    for m in metric_names:
        print(f"\n=== Sobol Analysis for '{m}' ===")
        Si: dict[str, ArrayF | float] = sobol.analyze(
            problem,
            metric_arrays[m],
            print_to_console=True,
        )
        sobol_results[m] = Si

    return sobol_results
