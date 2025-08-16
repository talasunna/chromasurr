from __future__ import annotations
from typing import Sequence


"""
batch_elution.py — preconfigured CADET batch-elution process.

This module defines :class:`BatchElution`, a convenience subclass of
:class:`CADETProcess.processModel.Process` that wires up a minimal
batch-elution flow sheet (feed → column → outlet, plus an eluent inlet)
with a Langmuir binding model. It is intended for quick prototyping,
surrogate training, and examples.

All public callables use Python 3.12+ type hints and NumPy-style
docstrings.
"""


from CADETProcess.processModel import (
    ComponentSystem,
    Langmuir,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    FlowSheet,
    Process,
)


class BatchElution(Process):
    """
    Preconfigured CADET batch-elution process with Langmuir adsorption.

    The flow sheet comprises a *feed* inlet, an *eluent* inlet, a single
    lumped-rate column (no pores), and an *outlet*. Two durations control
    when feed and eluent are active; simple events toggle inlet flow rates.

    Parameters
    ----------
    feed_conc : Sequence[float] or None, optional
        Feed concentrations for each component (defaults to ``[10.0, 10.0]``).
    column_length : float, default=0.6
        Column length in meters.
    axial_dispersion : float, default=4.7e-7
        Axial dispersion coefficient in m²/s.
    porosity : float, default=0.7
        Total bed porosity (dimensionless).
    capacity : Sequence[float] or None, optional
        Langmuir capacity per component in mol/m³ (defaults to ``[100.0, 100.0]``).
    adsorption_rate : Sequence[float] or None, optional
        Langmuir adsorption rate constants in 1/s (defaults to ``[0.02, 0.03]``).
    desorption_rate : Sequence[float] or None, optional
        Langmuir desorption rate constants in 1/s (defaults to ``[1.0, 1.0]``).
    feed_duration : float, default=60.0
        Time (s) during which *feed* is active.
    eluent_duration : float, default=60.0
        Time (s) during which *eluent* is active.
    cycle_time : float, default=600.0
        Total simulation time (s).

    Notes
    -----
    - The column diameter is fixed at 24 mm for this template.
    - Events ``feed_on/off`` and ``eluent_on/off`` toggle inlet flow rates.
    - Durations ``feed_duration`` and ``eluent_duration`` can be edited after
      construction to change the sequence timing.
    """

    def __init__(
        self,
        feed_conc: Sequence[float] | None = None,
        column_length: float = 0.6,
        axial_dispersion: float = 4.7e-7,
        porosity: float = 0.7,
        capacity: Sequence[float] | None = None,
        adsorption_rate: Sequence[float] | None = None,
        desorption_rate: Sequence[float] | None = None,
        feed_duration: float = 60.0,
        eluent_duration: float = 60.0,
        cycle_time: float = 600.0,
    ) -> None:
        # ---- defaults for sequence arguments (avoid mutable defaults) ----
        if feed_conc is None:
            feed_conc = [10.0, 10.0]
        if capacity is None:
            capacity = [100.0, 100.0]
        if adsorption_rate is None:
            adsorption_rate = [0.02, 0.03]
        if desorption_rate is None:
            desorption_rate = [1.0, 1.0]

        # Keep user-facing durations accessible
        self.feed_duration_time: float = float(feed_duration)
        self.eluent_duration_time: float = float(eluent_duration)

        # ---- component system and binding model ----
        system = ComponentSystem()
        system.add_component("A")
        system.add_component("B")

        langmuir = Langmuir(system)
        langmuir.is_kinetic = True
        langmuir.capacity = list(capacity)
        langmuir.adsorption_rate = list(adsorption_rate)
        langmuir.desorption_rate = list(desorption_rate)

        # ---- units ----
        feed = Inlet(system, name="feed")
        feed.c = list(feed_conc)

        eluent = Inlet(system, name="eluent")
        eluent.c = [1.0, 2.0]

        column = LumpedRateModelWithoutPores(system, name="column")
        column.binding_model = langmuir
        column.length = float(column_length)
        column.diameter = 0.024  # 24 mm
        column.axial_dispersion = float(axial_dispersion)
        column.total_porosity = float(porosity)
        column.solution_recorder.write_solution_bulk = True
        column.solution_recorder.write_solution_solid = True

        outlet = Outlet(system, name="outlet")

        # ---- flow sheet ----
        fs = FlowSheet(system)
        fs.add_unit(feed, feed_inlet=True)
        fs.add_unit(eluent, eluent_inlet=True)
        fs.add_unit(column)
        fs.add_unit(outlet, product_outlet=True)
        fs.add_connection(feed, column)
        fs.add_connection(eluent, column)
        fs.add_connection(column, outlet)

        super().__init__(fs, name="batch_elution")

        # ---- events & durations ----
        # Base volumetric flow [m^3/s]: 60 µL/min = 60 / (60 * 1e6) m^3/s
        Q: float = 60.0 / (60.0 * 1e6)
        Q_off: float = 8.33e-8  # small "off" flow

        self.add_event("feed_on", "flow_sheet.feed.flow_rate", Q)
        self.add_event("feed_off", "flow_sheet.feed.flow_rate", Q_off)
        self.add_event("eluent_on", "flow_sheet.eluent.flow_rate", Q)
        self.add_event("eluent_off", "flow_sheet.eluent.flow_rate", Q_off)

        self.add_duration("feed_duration")
        self.feed_duration.time = float(feed_duration)

        self.add_duration("eluent_duration")
        self.eluent_duration.time = float(eluent_duration)

        self.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
        self.add_event_dependency("eluent_on", ["feed_off"], None)
        self.add_event_dependency(
            "eluent_off", ["eluent_on", "eluent_duration"], [1, 1]
        )

        self.cycle_time = float(cycle_time)
