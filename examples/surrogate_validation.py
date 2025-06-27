import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from CADETProcess.processModel import (
    ComponentSystem, Langmuir,
    Inlet, LumpedRateModelWithoutPores, Outlet,
    FlowSheet, Process
)

from chromasurr.surrogate import Surrogate


def build_cadet_process():

    system = ComponentSystem()
    system.add_component('A')
    system.add_component('B')

    lm = Langmuir(system, name='langmuir')
    lm.is_kinetic = False
    lm.adsorption_rate = [0.02, 0.03]
    lm.desorption_rate = [1.0,  1.0]
    lm.capacity = [100,  100]

    feed = Inlet(system, name='feed')
    feed.c = [10, 10]
    eluent = Inlet(system, name='eluent')
    eluent.c = [0,  0]
    column = LumpedRateModelWithoutPores(system, name='column')
    column.binding_model = lm
    column.length = 0.6
    column.diameter = 0.024
    column.axial_dispersion = 4.7e-7
    column.total_porosity = 0.7
    column.solution_recorder.write_solution_bulk = True
    outlet = Outlet(system, name='outlet')

    fs = FlowSheet(system)
    fs.add_unit(feed,    feed_inlet=True)
    fs.add_unit(eluent,  eluent_inlet=True)
    fs.add_unit(column)
    fs.add_unit(outlet,  product_outlet=True)
    fs.add_connection(feed,   column)
    fs.add_connection(eluent, column)
    fs.add_connection(column, outlet)

    proc = Process(fs, 'batch elution')
    Q = 60/(60*1e6)

    proc.add_event('feed_on',  'flow_sheet.feed.flow_rate',   Q)
    proc.add_event('feed_off', 'flow_sheet.feed.flow_rate',   0.0)

    proc.add_event('eluent_on',  'flow_sheet.eluent.flow_rate', Q)
    proc.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
    proc.add_duration('feed_duration')
    proc.add_event_dependency('eluent_on',  ['feed_off'])
    proc.add_event_dependency('eluent_off', ['feed_on'])
    proc.add_event_dependency('feed_off', ['feed_on','feed_duration'], [1,1])
    proc.cycle_time = 600
    proc.feed_duration.time = 60

    return proc

param_config = {
    'ads_A': 'flow_sheet.column.binding_model.adsorption_rate[0]',
    'ads_B': 'flow_sheet.column.binding_model.adsorption_rate[1]',
    'des_A': 'flow_sheet.column.binding_model.desorption_rate[0]',
    'des_B': 'flow_sheet.column.binding_model.desorption_rate[1]',
    'cap_A': 'flow_sheet.column.binding_model.capacity[0]',
    'cap_B': 'flow_sheet.column.binding_model.capacity[1]',
}

bounds = {
    'ads_A': [0.01, 0.05],
    'ads_B': [0.01, 0.05],
    'des_A': [0.5,  2.0],
    'des_B': [0.5,  2.0],
    'cap_A': [50,  150],
    'cap_B': [50,  150],
}


def train_surrogate(process):
    sur = Surrogate(
        process,
        param_config=param_config,
        bounds=bounds,
        metrics=['retention_time', 'peak_width'],
        n_train=256,
        seed=42
    )
    print("Training surrogate (this may take a minute)...")
    sur.train()
    return sur


# ---  Validation routine -------------------------------------------------
def validate_surrogate(sur, metric, test_size=0.2, random_state=42):
    """Split sur.X, sur.Y[metric] into train/test, refits a GP, and returns stats."""
    # gather data & drop NaNs
    X_full = sur.X
    y_full = sur.Y[metric]
    mask = ~np.isnan(y_full)
    X, y = X_full[mask], y_full[mask]

    # hold-out split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # fit fresh GP
    gp = GaussianProcessRegressor(kernel=RBF(),
                                  normalize_y=True, random_state=random_state
                                  )
    gp.fit(X_tr, y_tr)

    # predict with uncertainty
    y_pred, y_std = gp.predict(X_te, return_std=True)

    # metrics
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    coverage = np.mean((y_te >= lower) & (y_te <= upper))

    # plot
    fig, ax = plt.subplots()
    ax.errorbar(
        y_te, y_pred,
        yerr=1.96 * y_std,
        fmt='o', alpha=0.6
    )
    mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], '--', label='y = x')
    ax.set_xlabel(f"True {metric}")
    ax.set_ylabel(f"Predicted {metric}")
    ax.set_title(f"Validation of surrogate for '{metric}'")
    ax.legend()
    plt.show()

    return {
        'rmse': rmse,
        'r2': r2,
        'coverage_95%': coverage
    }


if __name__ == '__main__':
    proc = build_cadet_process()
    sur = train_surrogate(proc)

    for m in sur.metrics:
        stats = validate_surrogate(sur, m)
        print(f"\n=== Results for '{m}' ===")
        print(f"  RMSE       : {stats['rmse']:.3f}")
        print(f"  R²         : {stats['r2']:.3f}")
        print(f"  95% coverage: {stats['coverage_95%']:.2%}")
