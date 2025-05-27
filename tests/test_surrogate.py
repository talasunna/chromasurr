"""
Test surrogate-based sensitivity analysis in chromasurr

Created on May 2025
"""

import pytest
import numpy as np
from chromasurr.surrogate import (
    train_multi_surrogate_models,
    run_multi_surrogate_sensitivity_analysis
)
from chromasurr.sensitivity import set_nested_attr
from chromasurr.metrics import extract
from CADETProcess.processModel import (
    ComponentSystem, Langmuir, Inlet,
    LumpedRateModelWithoutPores, Outlet,
    FlowSheet, Process
)


# 🔧 Fixture to create the test CADET process
@pytest.fixture
def test_process():
    cs = ComponentSystem()
    cs.add_component("A")
    cs.add_component("B")

    bm = Langmuir(cs, "langmuir")
    bm.is_kinetic = False
    bm.adsorption_rate = [0.02, 0.03]
    bm.desorption_rate = [1, 1]
    bm.capacity = [100, 100]

    feed = Inlet(cs, "feed")
    feed.c = [10, 10]
    eluent = Inlet(cs, "eluent")
    eluent.c = [0, 0]

    col = LumpedRateModelWithoutPores(cs, "column")
    col.binding_model = bm
    col.length = 0.6
    col.diameter = 0.024
    col.axial_dispersion = 4.7e-7
    col.total_porosity = 0.7
    col.solution_recorder.write_solution_bulk = True

    out = Outlet(cs, "outlet")

    fs = FlowSheet(cs)
    fs.add_unit(feed, feed_inlet=True)
    fs.add_unit(eluent, eluent_inlet=True)
    fs.add_unit(col)
    fs.add_unit(out, product_outlet=True)
    fs.add_connection(feed, col)
    fs.add_connection(eluent, col)
    fs.add_connection(col, out)

    p = Process(fs, "test_proc")
    Q = 60 / (60 * 1e6)
    p.add_event("feed_on", "flow_sheet.feed.flow_rate", Q)
    p.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
    p.add_event("eluent_on", "flow_sheet.eluent.flow_rate", Q)
    p.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0)
    p.add_duration("feed_duration")
    p.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
    p.add_event_dependency("eluent_on", ["feed_off"])
    p.add_event_dependency("eluent_off", ["feed_on"])
    p.cycle_time = 600
    p.feed_duration.time = 60
    return p


@pytest.mark.parametrize("metric", ["retention_time", "peak_width", "num_plates"])
def test_surrogate_sensitivity_analysis(metric, test_process):
    process = test_process

    # Parameter setup
    param_config = {
        "axial_disp": "flow_sheet.column.axial_dispersion",
        "total_porosity": "flow_sheet.column.total_porosity"
    }
    bounds = {
        "axial_disp": [3e-8, 9e-3],
        "total_porosity": [0.1, 0.9]
    }

    # Train surrogate only for the given metric
    models, Y, X, problem = train_multi_surrogate_models(
        process=process,
        param_config=param_config,
        bounds=bounds,
        metrics=[metric],
        n_train=64  # fast test
    )

    # Run surrogate-based Sobol analysis
    Si = run_multi_surrogate_sensitivity_analysis(
        models=models,
        problem=problem,
        n_samples=64
    )

    # Assertions
    assert metric in Si
    assert "S1" in Si[metric]
    assert isinstance(Si[metric]["S1"], np.ndarray)
    assert len(Si[metric]["S1"]) == len(param_config)
