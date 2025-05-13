#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:24:10 2025

@author: talasunna
"""

from chromasurr.metrics import extract
from CADETProcess.processModel import ComponentSystem
import numpy

component_system = ComponentSystem()
component_system.add_component('A')
component_system.add_component('B')

from CADETProcess.processModel import Langmuir

binding_model = Langmuir(component_system, name='langmuir')
binding_model.is_kinetic = False
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.capacity = [100, 100]

from CADETProcess.processModel import (
    Inlet, LumpedRateModelWithoutPores, Outlet
)
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

from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(feed, feed_inlet=True)
flow_sheet.add_unit(eluent, eluent_inlet=True)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, product_outlet=True)

flow_sheet.add_connection(feed, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, outlet)

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'batch elution')

Q = 60/(60*1e6)
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)

process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)

process.add_duration('feed_duration')

process.add_event_dependency('eluent_on', ['feed_off'])
process.add_event_dependency('eluent_off', ['feed_on'])
process.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1, 1])

process.cycle_time = 600
process.feed_duration.time = 60

if __name__ == '__main__':
    from CADETProcess.simulator import Cadet
    process_simulator = Cadet()

    sim= process_simulator.simulate(process)
    fig, ax = sim.solution.outlet.outlet.plot()


print("Number of chromatograms:", len(sim.chromatograms))

# Inspect each one
for idx, chromatogram in enumerate(sim.chromatograms):
    t = chromatogram.time                   # time grid (numpy array)
    c = chromatogram.total_concentration    # concentration vs time
    print(f"Chromatogram {idx}:  {len(t)} points, peak={c.max():.3e}")


peak, rt = extract(sim, 0)

print(peak)
print(rt)
