chromasurr – Uncertainty-Quantification Toolkit for Chromatography
===================================================================

*A lightweight Python library that couples the CADET process simulator with modern sensitivity analysis, Gaussian-process surrogates, and diagnostic tools.*

✨ Highlights
------------

- **Vectorized KPI extractor** – Compute retention time, peak width, and number of plates from any simulation in one call (``chromasurr.metrics.extract``)
- **Global Sobol sensitivity** – One-liner ``run_sensitivity_analysis()`` to rank parameters with Saltelli sampling and automatic metric extraction
- **Gaussian-process surrogate manager** – The ``Surrogate`` class trains, prunes, and re-trains emulators in log-space; includes built-in sensitivity on the GP itself
- **Reusable process builder** – The ``BatchElution`` class provides a ready-to-use CADET process object with configurable parameters like cycle time and feed duration
- **End-to-end example** – Full demo in ``examples/chromasurr_demo.py`` covering sensitivity analysis, calibration, and UQ

🚀 Quick start
--------------

Run the demo in *≈ 60 seconds*:

.. code-block:: bash

   # 1. Clone the repository
   git clone https://github.com/talasunna/chromasurr.git && cd chromasurr

   # 2. Create environment (Python ≥ 3.10)
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt  

   # 3. Launch the demo
   python examples/chromasurr_demo.py

This runs a full workflow using a preconfigured ``BatchElution`` process and guides you through GP surrogate modeling, Sobol sensitivity analysis, and Bayesian calibration.

📦 Installation
---------------

.. code-block:: bash

   pip install git+https://github.com/talasunna/chromasurr.git

**Note** – CADET-Core must be installed or compiled on your system. See the `CADET-Core Installation Guide <https://cadet.github.io/master/getting_started/installation_core.html>`_ for details.

CADET-Process is automatically installed via pip when installing ``chromasurr``.

🛠️ Usage at a glance
---------------------

.. code-block:: python

   from chromasurr.process.batch_elution import BatchElution
   from chromasurr.metrics import extract
   from chromasurr.sensitivity import run_sensitivity_analysis
   from chromasurr.surrogate import Surrogate
   from CADETProcess.simulator import Cadet

   # 1. Instantiate a configurable process object
   proc = BatchElution(cycle_time=600.0, feed_duration=50.0)

   # 2. Simulate and extract KPIs
   simulation_results = Cadet().simulate(proc)
   metrics = extract(simulation_results)

   # 3. Run Sobol sensitivity analysis (on CADET model)
   sobol = run_sensitivity_analysis(
       proc,
       param_config,
       bounds,
       metric_names=["retention_time", "peak_width"],
       n_samples=512
   )

   # 4. Train surrogate and predict
   sur = Surrogate(
       proc,
       param_config,
       bounds,
       metrics=["retention_time", "peak_width"],
       n_train=128
   )
   sur.train()
   rt_pred = sur.predict("retention_time", X_new)

Sensitivity Analysis Workflow Options
--------------------------------------

- **Option 1: Run Sensitivity Analysis First, Then Train Surrogate**

  Use ``run_sensitivity_analysis`` on the CADET model to rank parameters, then train a surrogate focusing on the important ones (based on     the number of parameters you'd like to retain, or on a threshold you can decide).

- **Option 2: Train Surrogate First, Then Run Sensitivity Analysis**

  Fit a surrogate with ``Surrogate``, then analyze it with ``analyze_sensitivity()``.

Both paths support uncertainty quantification and parameter calibration workflows.

---

All public functions include **NumPy-style docstrings** and **Python 3.10+ type hints** for autocompletion and static analysis.

📚 Documentation
----------------
*TBD – Documentation will be hosted on ReadTheDocs once stabilized.*

🖇️ Project structure
---------------------

.. code-block:: text

   chromasurr/
   │   __init__.py
   │   metrics.py              ← KPI extractor
   │   sensitivity.py          ← Saltelli driver + helpers
   │   surrogate.py            ← Surrogate manager
   │   calibration.py          ← Point + Bayesian calibration
   │   uq.py                   ← Uncertainty quantification tools
   │   error_analysis.py       ← Error diagnostics
   ├── process/
   │   └── batch_elution.py    ← Configurable CADET Process class
   └── examples/
       └── chromasurr_demo.py  ← End-to-end demo script

   docs/
   tests/

📜 License
-----------

Distributed under the **MIT License** – see *LICENSE* for details.

Made with ☕ by **Tala Al-Sunna**.
