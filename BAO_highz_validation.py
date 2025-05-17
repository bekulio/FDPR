# Create BAO_highz_validation.png
import numpy as np
import matplotlib.pyplot as plt

# High‑z BAO data (obs) and FDPR predictions (from table)
z   = np.array([0.380, 0.510, 0.610])
dv_obs = np.array([1476, 2005, 2240])
err_obs = np.array([20, 35, 60])
dv_pred = np.array([1469, 1998, 2258])

# Plot
plt.figure(figsize=(6,4.5))
plt.errorbar(z, dv_obs, yerr=err_obs, fmt='o', ms=6, label='Observed BAO', capsize=4)
plt.plot(z, dv_pred, 's', ms=6, label='FDPR prediction')
plt.xlabel('Redshift $z$')
plt.ylabel('$D_V$ [Mpc]')
plt.title('High‑$z$ BAO validation')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('BAO_highz_validation.png')
plt.close()

