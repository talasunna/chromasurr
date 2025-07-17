from __future__ import annotations

"""
chromasurr_demo.py

Comprehensive example script for the full Chromasurr workflow:

1. Build CADET Process model
2. Train surrogate
3. Sensitivity analysis & parameter selection
4. Calibration (point-estimate + Bayesian)
5. Uncertainty quantification (UQ)
6. Visualization of Sobol indices and posterior distributions
7. Summarize and plot posterior statistics (including corner plot)
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
from chromasurr.surrogate import Surrogate
from chromasurr.calibration import calibrate_surrogate, bayesian_calibration
from chromasurr.uq import perform_monte_carlo_uq, latin_hypercube_sampler
from chromasurr.visualize import (
    sobol_indices,  # <-- Import the Sobol indices function
    posterior,
    summarize_results,
)
from CADETProcess.processModel import (
    ComponentSystem, Langmuir,
    Inlet, LumpedRateModelWithoutPores, Outlet,
    FlowSheet, Process,
)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── 2. Build Process ─────────────────────────────────────────────────────────
def build_process() -> Process:
    system = ComponentSystem()
    system.add_component("A")
    system.add_component("B")

    langmuir = Langmuir(system, name="Langmuir")
    langmuir.is_kinetic = False
    langmuir.adsorption_rate = [0.02, 0.03]
    langmuir.desorption_rate = [1.0, 1.0]
    langmuir.capacity = [100.0, 100.0]

    feed = Inlet(system, name="feed")
    feed.c = [10.0, 10.0]
    eluent = Inlet(system, name="eluent")
    eluent.c = [0.0, 0.0]

    column = LumpedRateModelWithoutPores(system, name="column")
    column.binding_model = langmuir
    column.length = 0.6
    column.diameter = 0.024
    column.axial_dispersion = 4.7e-7
    column.total_porosity = 0.7
    column.solution_recorder.write_solution_bulk = True

    outlet = Outlet(system, name="outlet")

    fs = FlowSheet(system)
    fs.add_unit(feed, feed_inlet=True)
    fs.add_unit(eluent, eluent_inlet=True)
    fs.add_unit(column)
    fs.add_unit(outlet, product_outlet=True)
    fs.add_connection(feed, column)
    fs.add_connection(eluent, column)
    fs.add_connection(column, outlet)

    proc = Process(fs, name="demo_proc")
    Q = 60 / (60 * 1e6)
    proc.add_event("feed_on", "flow_sheet.feed.flow_rate", Q)
    proc.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0)
    proc.add_event("eluent_on", "flow_sheet.eluent.flow_rate", Q)
    proc.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0)
    proc.add_duration("feed_duration")
    proc.feed_duration.time = 60.0
    proc.add_duration("eluent_duration")
    proc.eluent_duration.time = 60.0
    proc.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
    proc.add_event_dependency("eluent_on", ["feed_off"], None)
    proc.add_event_dependency("eluent_off", ["eluent_on", "eluent_duration"], [1, 1])
    proc.cycle_time = 600.0
    return proc

# ── 3. Main Pipeline ─────────────────────────────────────────────────────────

def main() -> None:
    print("1. Building CADET process...")
    proc = build_process()

    print("2. Training surrogate model...")
    param_config = {
        "ax_disp": "flow_sheet.column.axial_dispersion",
        "porosity": "flow_sheet.column.total_porosity",
    }
    bounds = {
        "ax_disp": [1e-6, 1e-2],
        "porosity": [0.35, 0.95],
    }
    metrics = ["peak_width"]

    surr = Surrogate(proc, param_config, bounds, metrics, n_train=32)
    surr.train()
    print(f"   > Surrogate trained on {surr.X.shape[0]} samples.")
    print("    > Y std:", np.std(surr.Y['peak_width']))

    print("3. Sensitivity analysis...")
    surr.analyze_sensitivity(n_samples=64)

    print("4. Visualize Sensitivity Analysis...")
    sobol_results = surr.sensitivity  # Get Sobol results from the surrogate

    # Call the sobol_indices function to visualize Sobol indices
    sobol_indices(sobol_results, metric=metrics[0])  # Visualizing Sobol indices for the first metric

    print("4. Select important parameters...")
    surr.select_important_params(threshold=0.05)
    surr.retrain()
    print(f"   > Retained: {surr.top_params}")

    print("5a. Point calibration...")
    x_true = surr.X[0]
    y_obs = surr.predict(metrics[0], x_true.reshape(1, -1))[0]
    det = calibrate_surrogate(surr, y_obs=y_obs, metric=metrics[0])
    x_opt = det.x_opt
    print("   > Calibrated x_opt:", x_opt)

    print("5b. Bayesian calibration...")
    bayes = bayesian_calibration(
        surr,
        y_obs={metrics[0]: y_obs},
        n_walkers=24,
        n_steps=3000
    )

    samples = bayes.extra["chain"]
    print("   > MCMC samples:", samples.shape)

    print("Num vars in surrogate:", surr.problem["num_vars"])
    print("Bounds passed to sample_uniform:", surr.bounds)

    # Now use LHS for uncertainty quantification
    print("6. Uncertainty Quantification...")

    lhs_sampler = latin_hypercube_sampler(list(surr.bounds.values()))  # Ensure bounds are in the right format

    uq = perform_monte_carlo_uq(
        surrogate=surr,
        sample_input=lhs_sampler,
        metric=metrics[0],
        n_samples=1000
    )

    print(f"   > UQ mean = {uq['mean']}, 95% CI = ({uq['quantiles']['2.5%']}, {uq['quantiles']['97.5%']})")

    print("7. Posterior visualization...")
    params = list(surr.bounds)
    for p in params:
        posterior(samples, param_names=params, param=p,
                  xlim=(0, 1e-3) if p == 'ax_disp' else (0, 1)
                  )

    print("8. Summary statistics:")
    posterior_df = pd.DataFrame(samples, columns=params)
    summarize_results(
        surrogate=surr,
        metric=metrics[0],
        x_opt=x_opt,
        posterior_df=posterior_df,
        uq_result=uq
    )


if __name__ == "__main__":
    main()
