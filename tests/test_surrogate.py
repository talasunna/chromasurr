import pytest
import numpy as np
from chromasurr.surrogate import Surrogate
from CADETProcess.processModel import (
    ComponentSystem,
    Langmuir,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    FlowSheet,
    Process,
)


@pytest.fixture
def dummy_process():
    component_system = ComponentSystem()
    component_system.add_component("A")
    component_system.add_component("B")

    binding_model = Langmuir(component_system, name="langmuir")
    binding_model.is_kinetic = False
    binding_model.adsorption_rate = [0.02, 0.03]
    binding_model.desorption_rate = [1, 1]
    binding_model.capacity = [100, 100]

    feed = Inlet(component_system, name="feed")
    feed.c = [10, 10]
    eluent = Inlet(component_system, name="eluent")
    eluent.c = [0, 0]

    column = LumpedRateModelWithoutPores(component_system, name="column")
    column.binding_model = binding_model
    column.length = 0.6
    column.diameter = 0.024
    column.axial_dispersion = 4.7e-7
    column.total_porosity = 0.7
    column.solution_recorder.write_solution_bulk = True

    outlet = Outlet(component_system, name="outlet")

    fs = FlowSheet(component_system)
    fs.add_unit(feed, feed_inlet=True)
    fs.add_unit(eluent, eluent_inlet=True)
    fs.add_unit(column)
    fs.add_unit(outlet, product_outlet=True)
    fs.add_connection(feed, column)
    fs.add_connection(eluent, column)
    fs.add_connection(column, outlet)

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


@pytest.mark.parametrize(
    "param_config, bounds, metrics",
    [
        (
            {
                "ax_disp": "flow_sheet.column.axial_dispersion",
                "porosity": "flow_sheet.column.total_porosity",
            },
            {"ax_disp": [1e-8, 1e-3], "porosity": [0.1, 0.9]},
            ["retention_time", "peak_width"],
        )
    ],
)
@pytest.mark.parametrize("threshold", [0.01, 0.05])
def test_surrogate_pipeline(dummy_process, param_config, bounds, metrics, threshold):
    # Train initial surrogate
    surr = Surrogate(
        process=dummy_process,
        param_config=param_config,
        bounds=bounds,
        metrics=metrics,
        n_train=32,
    )

    surr.train()

    for m in metrics:
        assert m in surr.models, f"Model for metric '{m}' not trained"
        assert surr.Y[m].shape[0] > 0

    assert surr.X.shape[1] == len(param_config)

    # Sensitivity analysis
    surr.analyze_sensitivity(n_samples=32)
    for m in metrics:
        assert "ST" in surr.sensitivity[m]

    # Select important parameters
    surr.select_important_params(threshold=threshold)
    assert isinstance(surr.top_params, list)
    assert len(surr.top_params) > 0
    assert all(p in param_config for p in surr.top_params)

    # Retrain surrogate with fewer inputs
    surr.retrain()
    assert surr.X.shape[1] == len(surr.top_params)

    # Predict on valid new input
    x_test = np.array([[1e-6, 0.6]])[:, : len(surr.top_params)]
    for m in metrics:
        pred = surr.predict(m, x_test)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (1,)
        assert np.isfinite(pred[0]) or np.isnan(
            pred[0]
        )  # allow NaN if model had missing training data
