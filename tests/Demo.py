#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:31:39 2025

@author: talasunna
"""

"""
quick_demo.py  – miniature pipeline with a toy “CADET” function
────────────────────────────────────────────────────────────────
Parameters:
    theta = [D, A]
Toy physics:
    retention time  tR      = A * 50
    peak width      width   = sqrt(D) * 2
    peak height     height  = 1 / width
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc                      # ← SciPy LHS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import emcee, tqdm

# -------------------------------
# 1. Design of Experiments
# -------------------------------
N_TRAIN = 40
BOUNDS  = np.array([[1e-7, 5e-6],      # Dispersion
                    [1.5e-5, 3.0e-5]]) # Area
rng     = np.random.default_rng(0)

# Latin-Hypercube in [0,1]^2
sampler  = qmc.LatinHypercube(d=2, seed=rng)
lhs_unit = sampler.random(N_TRAIN)             # shape (N,2)

# scale to physical bounds
X_train = BOUNDS[:, 0] + lhs_unit * (BOUNDS[:, 1] - BOUNDS[:, 0])

# Toy “CADET” → metrics
def toy_cadet(theta):
    D, A   = theta
    tR     = A * 50
    width  = np.sqrt(D) * 2
    height = 1.0 / width
    return np.array([tR, width, height])

Y_train = np.array([toy_cadet(th) for th in X_train])

# -------------------------------
# 2. Fit GP surrogate
# -------------------------------
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1e-6, 1e-5],
                                   length_scale_bounds=(1e-8, 1e-3))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)
gp.fit(X_train, Y_train)

# -------------------------------
# 3. Synthetic observation + error model
# -------------------------------
theta_true = np.array([2.5e-6, 2.1e-5])
y_true     = toy_cadet(theta_true)

SIGMA = np.array([0.05, 0.0005, 0.02])            # σ_tR, σ_width, σ_height
y_obs = y_true + rng.normal(scale=SIGMA)

print("True metrics:", y_true)
print("Observed    :", y_obs)

inv_var  = 1.0 / (SIGMA ** 2)
log_norm = -0.5 * np.log(2 * np.pi * SIGMA ** 2).sum()

def log_prior(theta):
    return 0.0 if np.all((theta >= BOUNDS[:, 0]) & (theta <= BOUNDS[:, 1])) else -np.inf

def log_like(theta):
    y_pred = gp.predict(theta.reshape(1, -1))[0]
    r      = y_obs - y_pred
    return log_norm - 0.5 * np.sum(r * r * inv_var)

def log_post(theta):
    lp = log_prior(theta)
    return -np.inf if not np.isfinite(lp) else lp + log_like(theta)

# -------------------------------
# 4. MCMC with emcee
# -------------------------------
nwalk, nsteps = 20, 4000
p0 = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(nwalk, 2))
sampler = emcee.EnsembleSampler(nwalk, 2, log_post)

print("Running MCMC…")
sampler.run_mcmc(p0, nsteps, progress=True)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)

# -------------------------------
# 5. Posterior summary & plot
# -------------------------------
mean = samples.mean(axis=0)
hdi  = np.percentile(samples, [2.5, 97.5], axis=0)

print("\nPosterior mean:", mean)
print("95% HDI:\n", np.vstack(hdi).T)
print("True theta   :", theta_true)

plt.figure(figsize=(5, 5))
plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.3)
plt.plot(theta_true[0], theta_true[1], "r*", ms=12, label="true")
plt.xlabel("Dispersion D")
plt.ylabel("Area A")
plt.title("Posterior samples")
plt.legend()
plt.tight_layout()
plt.show()
