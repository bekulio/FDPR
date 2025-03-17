#!/usr/bin/env python3
"""
Author: Benny Boris Kulangiev
Email: BennyBK.research@gmail.com
License: MIT License

Copyright (c) 2025 Benny Boris Kulangiev
"""

import numpy as np
import matplotlib.pyplot as plt

# BAO observed data: [redshift, observed D_V in Mpc]
bao_data = np.array([
    [0.15, 664],
    [0.38, 1476],
    [0.51, 2005],
    [0.61, 2240],
    [1.48, 3840]
])

z_vals = bao_data[:, 0]
D_V_obs = bao_data[:, 1]

# Constants:
# c/H0 approximated as 4477.61 Mpc (with H0 = 67 km/s/Mpc)
c_div_H0 = 4477.61  # Mpc
alpha = 2e-5        # Mpc^{-1} (baseline damping for optical photons)

def D_V_pred(z, alpha):
    """
    Computes the BAO volume-averaged distance D_V(z) predicted by the FDPR model.
    
    Parameters:
      z     : Redshift
      alpha : Damping coefficient (Mpc^{-1})
      
    Returns:
      D_V(z) in Mpc
    """
    return c_div_H0 * np.log(1 + z + alpha * z**2)

# Compute predicted BAO distances for the observed redshifts
D_V_predicted = D_V_pred(z_vals, alpha)

# Print the numerical comparison table
print("BAO Numerical Comparison:")
print("Redshift z   Observed D_V (Mpc)   Predicted D_V (Mpc)")
for z, d_obs, d_pred in zip(z_vals, D_V_obs, D_V_predicted):
    print(f"{z:0.2f}           {d_obs:8.1f}              {d_pred:8.1f}")

# Plot observed vs predicted BAO distances
plt.figure(figsize=(8,5))
plt.plot(z_vals, D_V_obs, 'o-', label='Observed $D_V(z)$')
plt.plot(z_vals, D_V_predicted, 's-', label='Predicted $D_V(z)$')
plt.xlabel("Redshift, z")
plt.ylabel("$D_V(z)$ [Mpc]")
plt.title("BAO Numerical Comparison in FDPR Model")
plt.legend()
plt.grid(True)
plt.savefig("BAO_comparison.png", dpi=300)
plt.show()
