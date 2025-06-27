import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from chromasurr.surrogate import Surrogate
from CADETProcess.processModel import (ComponentSystem, Langmuir,
    Inlet, LumpedRateModelWithoutPores, Outlet, FlowSheet, Process)
from CADETProcess.simulator import Cadet


def build_process():
    system = ComponentSystem()
    system.add_component('A')
    system.add_component('B')
    lm = Langmuir(system, name='langmuir')
    lm.is_kinetic = False
    lm.adsorption_rate = [0.02, 0.03]
    lm.desorption_rate = [1, 1]
    lm.capacity = [100, 100]
    feed = Inlet(system, 'feed')
    feed.c = [10, 10]
    eluent = Inlet(system, 'eluent')
    eluent.c = [0, 0]
    column = LumpedRateModelWithoutPores(system, 'column')
    column.binding_model = lm
    column.length = 0.6
    column.diameter = 0.024
    column.axial_dispersion = 4.7e-7
    column.total_porosity = 0.7
    column.solution_recorder.write_solution_bulk = True
    outlet = Outlet(system, 'outlet')
    fs = FlowSheet(system)
    fs.add_unit(feed, feed_inlet=True)
    fs.add_unit(eluent, eluent_inlet=True)
    fs.add_unit(column)
    fs.add_unit(outlet, product_outlet=True)
    fs.add_connection(feed, column)
    fs.add_connection(eluent, column)
    fs.add_connection(column, outlet)
    proc = Process(fs, 'batch elution')
    Q = 60/(60*1e6)
    proc.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q)
    proc.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0)
    proc.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
    proc.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
    proc.add_duration('feed_duration')
    proc.add_event_dependency('eluent_on', ['feed_off'])
    proc.add_event_dependency('eluent_off', ['feed_on'])
    proc.add_event_dependency('feed_off', ['feed_on', 'feed_duration'], [1, 1])
    proc.cycle_time = 600
    proc.feed_duration.time = 60
    return proc


# 1. Train a quick surrogate just for EDA
proc = build_process()
param_config = {
    'ads_A': 'flow_sheet.column.binding_model.adsorption_rate[0]',
    'ads_B': 'flow_sheet.column.binding_model.adsorption_rate[1]',
    'des_A': 'flow_sheet.column.binding_model.desorption_rate[0]',
    'des_B': 'flow_sheet.column.binding_model.desorption_rate[1]',
    'cap_A': 'flow_sheet.column.binding_model.capacity[0]',
    'cap_B': 'flow_sheet.column.binding_model.capacity[1]',
}
bounds = {k: ([0.01, 0.05] if 'ads' in k else
              [0.5, 2.0] if 'des' in k else
              [50, 150]) for k in param_config}
sur = Surrogate(proc, param_config, bounds,
                metrics=['retention_time'], n_train=128, seed=0)
sur.train()

# 2. Pull out the data & drop NaNs
y = sur.Y['retention_time']
X = sur.X
mask = ~np.isnan(y)
y = y[mask]
X = X[mask]

# 3.  Histogram
plt.figure()
plt.hist(y, bins=25, edgecolor='k')
plt.title('Histogram of retention_time')
plt.xlabel('retention_time')
plt.ylabel('Count')
plt.tight_layout()

# 4.  Q–Q plot vs. Normal
plt.figure()
st.probplot(y, dist="norm", plot=plt)
plt.title('Q–Q plot of retention_time')

# 5.  Skewness & kurtosis
print(f"Skewness     : {st.skew(y):.3f}")
print(f"Excess kurt. : {st.kurtosis(y):.3f}")

# 6.  Quick GP fit on raw y to inspect residuals
gp = GaussianProcessRegressor(kernel=RBF(), normalize_y=True, random_state=0)
gp.fit(X, y)
y_pred = gp.predict(X)
resid = y - y_pred

# 7.  Residual histogram
plt.figure()
plt.hist(resid, bins=25, edgecolor='k')
plt.title('Histogram of GP residuals')

# 8.  Residuals vs. predictions
plt.figure()
plt.scatter(y_pred, resid, alpha=0.5)
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('GP mean prediction')
plt.ylabel('Residual')
plt.title('Residuals vs. prediction')

# 9.  Test for heteroscedasticity (Breusch–Pagan)
lm = sm.OLS(resid, sm.add_constant(y_pred)).fit()
bp_p = het_breuschpagan(lm.resid, lm.model.exog)[1]
print(f"Breusch–Pagan p-value: {bp_p:.3f} "
      f"({'heteroscedastic' if bp_p < 0.05 else 'homoscedastic'})")

plt.show()
