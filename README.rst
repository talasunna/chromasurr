chromasurr – Uncertainty-Quantification Toolkit for Chromatography
===================================================================

*A lightweight Python library that couples the CADET process simulator with modern sensitivity analysis, Gaussian-process surrogates, and diagnostic tools.*

✨ Highlights
------------

- **Vectorized KPI extractor** – Compute retention time, peak width, and number of plates from any simulation in one call (``chromasurr.metrics.extract``)
- **Global Sobol sensitivity** – One-liner ``run_sensitivity_analysis()`` to rank parameters with Saltelli sampling and automatic metric extraction
- **Gaussian-process surrogate manager** – The ``Surrogate`` class trains, prunes, and re-trains emulators in log-space; includes built-in sensitivity on the GP itself
- **Error diagnostics** – 
- **End-to-end demo** – Full workflow from CADET flowsheet → KPIs → Sobol → surrogate in ``examples/chromasurr_demo.py``

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

The script simulates a batch-elution column in CADET, plots outlet profiles, and walks through KPI extraction, global Sobol sensitivity, and GP surrogate creation.

📦 Installation
---------------

.. code-block:: bash

   pip install git+https://github.com/talasunna/chromasurr.git

**Note** – CADET-Process ≥ 0.10 must already be compiled on your machine.  
See the `CADET documentation <https://github.com/fau-cade/cadet>`_ for platform-specific instructions.

🛠️ Usage at a glance
---------------------

.. code-block:: python

   from chromasurr.metrics     import extract
   from chromasurr.sensitivity import run_sensitivity_analysis
   from chromasurr.surrogate   import Surrogate
   from CADETProcess.simulator import Cadet

   # 1. Simulate a CADET process object `proc`
   sim     = Cadet().simulate(proc)
   metrics = extract(sim)  # {'peak_width': …, 'retention_time': …, 'num_plates': …}

   # 2. Global Sobol on either Cadet model (option 1) or surrogate model (option 2). Below is sensitivity analysis on the Cadet model.
   sobol = run_sensitivity_analysis(
       proc,
       param_config,
       bounds,
       metric_names=["retention_time", "peak_width"],
       n_samples=512
   )

   # 3. Fit & use a Gaussian-process emulator
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

  Run sensitivity analysis on the CADET model (or surrogate) using ``run_sensitivity_analysis``, then use results to focus surrogate training on the most important parameters.

- **Option 2: Train Surrogate First, Then Run Sensitivity Analysis**

  Train a surrogate using the ``Surrogate`` class, then perform analysis using the built-in ``analyze_sensitivity()`` method.

Both workflows allow flexibility in uncertainty quantification and model validation for chromatography.

---

All public functions include **NumPy-style docstrings** and **Python 3.10+ type hints** for autocompletion and static analysis.

📚 Documentation
----------------
*TBD*

🖇️ Project structure
---------------------

.. code-block:: text

   chromasurr/
   │   __init__.py
   │   metrics.py            ← KPI extractor
   │   sensitivity.py        ← Saltelli driver + helpers
   │   surrogate.py          ← Surrogate manager
   │   error_analysis.py     ← Diagnostics utilities
   └── examples/
       └── chromasurr_demo.py     ← End-to-end workflow
   docs/
   tests/

📜 License
-----------

Distributed under the **MIT License** – see *LICENSE* for details.

Made with ☕ by **Tala Al-Sunna**.
