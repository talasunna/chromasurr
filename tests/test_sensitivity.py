#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 2025

@author: talasunna
"""

import pytest
import numpy as np
from chromasurr.sensitivity import run_sensitivity_analysis
from CADETProcess.processModel import (
    ComponentSystem,
    Langmuir,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    FlowSheet,
    Process,
)


def create_simple_process():
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


@pytest.mark.parametrize("metric", ["retention_time", "peak_width", "num_plates"])
def test_run_sensitivity_analysis(metric):
    process = create_simple_process()
    param_config = {
        "ads_rate_A": "flow_sheet.column.binding_model.adsorption_rate[0]",
        "length": "flow_sheet.column.length",
    }
    bounds = {"ads_rate_A": [0.01, 0.05], "length": [0.3, 0.9]}

    results = run_sensitivity_analysis(
        process=process,
        param_config=param_config,
        bounds=bounds,
        metric_names=[metric],
        n_samples=64,  # Small sample size for fast testing
    )

    assert metric in results
    Si = results[metric]
    assert isinstance(Si, dict)
    assert "S1" in Si
    assert isinstance(Si["S1"], np.ndarray)
    assert len(Si["S1"]) == len(param_config)
