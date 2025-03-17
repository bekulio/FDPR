#!/usr/bin/env python3
"""
Author: Benny Boris Kulangiev
Email: BennyBK.research@gmail.com
License: MIT License

Copyright (c) 2025 Benny Boris Kulangiev
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Define the FDPR model
# --------------------------------------------------

def alpha_E(E, alpha0, E_ref, E_c):
    """
    Planck-like FDPR damping coefficient.
    
    Parameters:
      E       : Photon energy in eV
      alpha0  : Damping coefficient at E_ref
      E_ref   : Reference photon energy, e.g. 2 eV (optical)
      E_c     : Critical energy scale
    """
    numerator   = 1 - np.exp(-E / E_c)
    denominator = 1 - np.exp(-E_ref / E_c)
    return alpha0 * (numerator / denominator)

def H0_eff(alpha0, E_c, E_ref=2.0):
    """
    Effective Hubble constant in FDPR for optical photons.
    H_true is the CMB-based Hubble constant = 67 km/s/Mpc.
    c ~ 3e5 km/s.
    """
    c = 3e5
    H_true = 67.0
    # alpha(E_ref)
    alpha_opt = alpha_E(E_ref, alpha0, E_ref, E_c)
    return H_true + c * alpha_opt

# --------------------------------------------------
# 2. Synthetic data: local H0 measurement
# --------------------------------------------------
H0_obs = 73.0
H0_sigma = 1.0

# --------------------------------------------------
# 3. Define log-likelihood and log-priors
# --------------------------------------------------

def log_prior(theta):
    alpha0, E_c = theta
    # Both must be positive; set some upper bounds for safety
    if alpha0 <= 0 or alpha0 > 1e-4:
        return -np.inf
    if E_c <= 0 or E_c > 10:
        return -np.inf
    return 0.0  # flat prior within these ranges

def log_likelihood(theta):
    alpha0, E_c = theta
    model_H0 = H0_eff(alpha0, E_c)
    return -0.5 * ((H0_obs - model_H0) / H0_sigma)**2

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# --------------------------------------------------
# 4. Run MCMC
# --------------------------------------------------
ndim = 2
nwalkers = 32
nsteps = 5000

# Initial guesses near alpha0=2.7e-5, E_c=1.1 eV
initial_guess = [2.7e-5, 1.1]
pos = initial_guess + 1e-6 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, nsteps, progress=True)

# Flatten chain after burn-in
flat_samples = sampler.get_chain(discard=1000, thin=10, flat=True)
print(f"Number of posterior samples: {len(flat_samples)}")

# --------------------------------------------------
# 5. Corner plot
# --------------------------------------------------
labels = [r"$\alpha_0$ (Mpc$^{-1}$)", r"$E_c$ (eV)"]
fig = corner.corner(flat_samples, labels=labels, show_titles=True)
fig.savefig("mcmc_2D_corner_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# 6. Print median results
# --------------------------------------------------
alpha0_median, Ec_median = np.median(flat_samples, axis=0)
print(f"Fitted alpha0 = {alpha0_median:.3e} Mpc^-1")
print(f"Fitted E_c     = {Ec_median:.2f} eV")

H0_final = H0_eff(alpha0_median, Ec_median)
print(f"Inferred H0_eff from best fit: {H0_final:.2f} km/s/Mpc")
