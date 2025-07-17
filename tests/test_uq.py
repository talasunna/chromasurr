import pytest
import numpy as np
from chromasurr.surrogate import Surrogate
from chromasurr.calibration import calibrate_surrogate
from chromasurr.uq import perform_monte_carlo_uq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chromasurr.calibration import CalibrationResult

from CADETProcess.processModel import (
    ComponentSystem, Langmuir, Inlet,
    LumpedRateModelWithoutPores, Outlet,
    FlowSheet, Process
)


@pytest.fixture
def dummy_process():
    component_system = ComponentSystem()
    component_system.add_component('A')
    component_system.add_component('B')

    binding_model = Langmuir(component_system, name='langmuir')
    binding_model.is_kinetic = False
    binding_model.adsorption_rate = [0.02, 0.03]
    binding_model.desorption_rate = [1, 1]
    binding_model.capacity = [100, 100]

    feed = Inlet(component_system, name='feed')
    feed.c = [10, 10]
    eluent = Inlet(component_system, name='eluent')
    eluent.c = [0, 0]

    column = LumpedRateModelWithoutPores(component_system, name='column')
    column.binding_model = binding_model
    column.length = 0.6
    column.diameter = 0.024
    column.axial_dispersion = 4.7e-7
    column.total_porosity = 0.7
    column.solution_recorder.write_solution_bulk = True

    outlet = Outlet(component_system, name='outlet')

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


@pytest.fixture
def dummy_surrogate(dummy_process):
    param_config = {
        "ax_disp": "flow_sheet.column.axial_dispersion",
        "porosity": "flow_sheet.column.total_porosity"
    }
    bounds = {
        "ax_disp": [1e-8, 1e-3],
        "porosity": [0.1, 0.9]
    }
    metrics = ["retention_time"]

    surr = Surrogate(
        process=dummy_process,
        param_config=param_config,
        bounds=bounds,
        metrics=metrics,
        n_train=32
    )
    surr.train()
    surr.analyze_sensitivity(n_samples=32)
    surr.select_important_params(threshold=0.05)
    surr.retrain()
    return surr


def test_calibration_point_estimate(dummy_surrogate):
    """Test point-estimate calibration returns CalibrationResult with correct shape and values."""

    surrogate = dummy_surrogate
    metric = "retention_time"

    x_sample = surrogate.X[0]
    y_obs = surrogate.predict(metric, x_sample.reshape(1, -1))[0]

    # Use the new mapping API for calibration
    result = calibrate_surrogate(surrogate, y_obs={metric: y_obs})
    assert isinstance(result, CalibrationResult)

    x_hat = result.x_opt
    assert isinstance(x_hat, np.ndarray)
    assert x_hat.shape == (len(surrogate.top_params),)
    assert np.all(np.isfinite(x_hat))
    # Check that prediction at x_hat is close to y_obs
    y_pred = surrogate.predict(metric, x_hat.reshape(1, -1))[0]
    assert np.abs(y_pred - y_obs) < 1.0  # loosen threshold for model error


def test_uq_monte_carlo(dummy_surrogate):
    """Test uncertainty quantification via Monte Carlo propagation."""
    surrogate = dummy_surrogate
    metric = "retention_time"
    bounds = surrogate.bounds

    def sample_input(n):
        lb, ub = zip(*bounds.values())
        lb = np.array(lb)
        ub = np.array(ub)
        return lb + (ub - lb) * np.random.rand(n, len(bounds))

    result = perform_monte_carlo_uq(
        surrogate=surrogate,
        sample_input=sample_input,
        metric=metric,
        n_samples=256
    )

    assert "mean" in result
    assert "variance" in result
    assert "quantiles" in result
    assert isinstance(result["mean"], float)
    assert isinstance(result["variance"], float)
    assert result["y_means"].shape == (256,)
    assert np.all(np.isfinite(result["y_means"]))
