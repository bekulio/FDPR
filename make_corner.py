import emcee, numpy as np, corner, matplotlib.pyplot as plt

# Open backend and fetch the full chain
backend = emcee.backends.HDFBackend("fdpr_chain.h5", read_only=True)
chain   = backend.get_chain()              # shape: (steps, walkers, ndim)

# Flatten after burn-in
burnin  = min(800, chain.shape[0] // 2)
flat = chain[burnin:].reshape(-1, 2)

labels = [r"$\alpha_0\;[\mathrm{Mpc}^{-1}]$",
          r"$E_c\;[\mathrm{eV}]$"]

fig = corner.corner(flat, labels=labels,
                    quantiles=[0.16,0.5,0.84],
                    show_titles=True, title_fmt=".2e",
                    title_kwargs={"fontsize":10})
plt.tight_layout()
plt.savefig("corner_alpha_Ec.png", dpi=300)
plt.close()
