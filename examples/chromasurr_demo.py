from __future__ import annotations

"""
chromasurr_demo.py

Comprehensive, fully self-contained example script for the Chromasurr workflow.
Includes detailed inline explanations for new users.

Steps:
1. Build a CADET Process model for batch elution
2. Train a Gaussian-process surrogate model
3. Perform global Sobol sensitivity analysis
4. Select important parameters and retrain surrogate
5. Calibrate the model (point estimate + Bayesian MCMC)
6. Estimate prediction uncertainty
7. Visualize results (Sobol plots, posteriors, summary)
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
from chromasurr.surrogate import Surrogate
from chromasurr.uq import perform_monte_carlo_uq, latin_hypercube_sampler
from chromasurr.visualize import sobol_indices, summarize_results, uq_distribution
from chromasurr.process.batch_elution import BatchElution
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning

# Silence common warnings for a cleaner demo
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── 2. Main pipeline ─────────────────────────────────────────────────────────
def main() -> None:
    print("1. Building CADET process model (batch elution)...")
    # Instantiate a reusable, configurable batch elution process
    proc = BatchElution(cycle_time=600.0, feed_duration=60.0, eluent_duration=60.0)

    print("2. Setting up surrogate training...")
    param_config = {
        "ax_disp": "flow_sheet.column.axial_dispersion",
        "porosity": "flow_sheet.column.total_porosity",
    }
    bounds = {
        "ax_disp": [1e-6, 1e-2],
        "porosity": [0.35, 0.95],
    }
    metrics = ["peak_width"]

    print("   > Training GP surrogate on 128 samples...")
    surr = Surrogate(proc, param_config, bounds, metrics, n_train=128)
    surr.train()
    print(f"   > Surrogate trained. Y std = {np.std(surr.Y['peak_width']):.3f}")

    print("3. Running global Sobol sensitivity analysis...")
    surr.analyze_sensitivity(n_samples=1024)
    sobol_results = surr.sensitivity
    sobol_indices(sobol_results, metric=metrics[0])

    print("4. Selecting important parameters...")
    surr.select_important_params(threshold=0.05)
    surr.retrain()
    print(f"   > Retained: {surr.top_params}")

    print("5. Uncertainty quantification via Monte Carlo...")
    lhs_sampler = latin_hypercube_sampler(list(surr.bounds.values()))
    uq = perform_monte_carlo_uq(
        surrogate=surr, sample_input=lhs_sampler, metric=metrics[0], n_samples=1000
    )
    print(f"   > UQ 95% CI = ({uq['quantiles']['2.5%']:.3f}, {uq['quantiles']['97.5%']:.3f})")
    uq_distribution(uq, metric="peak_width")
    summarize_results(
        surrogate=surr,
        metric=metrics[0],
        uq_result=uq,
    )


if __name__ == "__main__":
    main()
