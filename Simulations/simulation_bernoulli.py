#!/usr/bin/env python3
"""
simulation_bernoulli.py
======================
Simulates n Bernoulli trials with success probability p, repeats 100 times,
and visualises the suitably normalized behaviour as n grows.

Key normalisation (Central-Limit-Theorem scaling):

    Z_n = (S_n - n·p) / sqrt(n·p·(1-p))

where S_n is the number of successes out of n trials.
As n → ∞, Z_n converges in distribution to N(0,1).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)

P          = 0.3            # Bernoulli success probability
NUM_SIMS   = 15000          # number of repeated simulations
N_MAX      = 500           # maximum number of trials per simulation
N_VALUES   = [20, 50, 200, 500]   # specific n values for histograms
ALPHA_PATH = 0.15           # line transparency for accumulated curves

# ---------------------------------------------------------------------------
# 1.  Generate the Bernoulli trials matrix  (shape: NUM_SIMS × N_MAX)
# ---------------------------------------------------------------------------
print("Generating Bernoulli trials …")
trials = np.random.binomial(n=1, p=P, size=(NUM_SIMS, N_MAX))

# Cumulative sums along each row  S_k(i) for simulation i
cumsums = np.cumsum(trials, axis=1)

# ---------------------------------------------------------------------------
# Normalised process at every step k
# Z_k(i) = (S_k(i) - k·p) / sqrt(k·p·(1-p))
# ---------------------------------------------------------------------------
k = np.arange(1, N_MAX + 1)                       # step indices 1 … N_MAX
expected = k * P                                  # n·p for each step
std_dev  = np.sqrt(k * P * (1 - P))
Z = (cumsums - expected) / std_dev                 # shape (NUM_SIMS, N_MAX)

# ---------------------------------------------------------------------------
# Figure 1 – Raw cumulative successes  S_k
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

for i in range(NUM_SIMS):
    ax1.plot(k, cumsums[i], color="steelblue", alpha=ALPHA_PATH, linewidth=0.8)

ax1.plot(k, expected, color="red", linewidth=2, linestyle="--",
         label=f"Theoretical mean  $E[S_k]=kp$  ($p={P}$)")
ax1.fill_between(k, expected - std_dev, expected + std_dev,
                 color="orange", alpha=0.15,
                  label=r"$\pm 1\sigma$ envelope")

ax1.set_title(f"Figure 1 – Cumulative Successes $S_k$  "
              f"($p={P}$, {NUM_SIMS} simulations)", fontsize=13)
ax1.set_xlabel("Step $k$ (number of trials)", fontsize=12)
ax1.set_ylabel("Number of successes $S_k$", fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig("bernoulli_cumulative.png", dpi=150)
print("  → saved bernoulli_cumulative.png")
plt.close(fig1)



# ---------------------------------------------------------------------------
# Figure 2 – Normalised sample paths  Z_k  (CLT convergence)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))

for i in range(NUM_SIMS):
    ax2.plot(k, Z[i], color="forestgreen", alpha=ALPHA_PATH, linewidth=0.8)

for nv in N_VALUES:
    ax2.axvline(nv, color="grey", linestyle=":", alpha=0.4)

ax2.axhline(0, color="red", linewidth=1.5, linestyle="--", label="mean = 0")
ax2.axhline(1, color="darkred", linewidth=1, linestyle="-.", alpha=0.5, label="+1σ")
ax2.axhline(-1, color="darkred", linewidth=1, linestyle="-.", alpha=0.5, label="-1σ")

ax2.set_title(
    f"Figure 2 – Normalised Paths " \
    rf"$Z_k = (S_k-kp)/\sqrt{{kp(1-p)}}$  ({NUM_SIMS} curves)",
    fontsize=13,
)
ax2.set_xlabel("Step $k$", fontsize=12)
ax2.set_ylabel("$Z_k$", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-4, 4)

plt.tight_layout()
fig2.savefig("bernoulli_normalized_paths.png", dpi=150)
print("  → saved bernoulli_normalized_paths.png")

# ---------------------------------------------------------------------------
# Figure 3 – Histograms of Z_n for selected n vs standard normal
# ---------------------------------------------------------------------------
# Why jitter?  Z_n is discrete (step ΔZ = 1/√(np(1-p))).  With a fixed bin
# count the bars "beat" against the underlying grid, leaving gaps.  A small
# Gaussian jitter (σ = ΔZ / 4) is a standard continuity correction that
# smooths the histogram without distorting the distribution.

x_grid = np.linspace(-4, 4, 300)
phi = stats.norm.pdf(x_grid)  # standard normal PDF

for idx, n in enumerate(N_VALUES):
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = Z[:, n - 1].copy()                       # Z values at step n

    # ── align bin edges to midpoints between consecutive Z values ──
    sorted_unique = np.sort(np.unique(vals))
    bin_edges = (sorted_unique[:-1] + sorted_unique[1:]) / 2
    bin_edges = np.concatenate([[bin_edges[0] - (bin_edges[1] - bin_edges[0])],
                                bin_edges,
                                [bin_edges[-1] + (bin_edges[-1] - bin_edges[-2])]])

    ax.hist(vals, bins=bin_edges, density=True, alpha=0.6,
            color="dodgerblue", edgecolor="grey",
            label=f"$n={n}$")

    ax.plot(x_grid, phi, color="red", linewidth=2, label="$N(0,1)$")
    ax.set_title(
        f"Figure {idx+3} – Distribution of $Z_n$  ($n = {n}$, "
        f"{NUM_SIMS} simulations)",
        fontsize=13,
    )
    ax.set_xlabel("$Z_n$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"bernoulli_histogram_n{n}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  → saved {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 – Convergence of empirical moments of Z_n
# ---------------------------------------------------------------------------
n_range = np.arange(10, N_MAX + 1)
emp_mean = np.array([np.mean(Z[:, n - 1]) for n in n_range])
emp_std  = np.array([np.std(Z[:, n - 1], ddof=1) for n in n_range])
ks_stat = np.array(
    [stats.kstest(Z[:, n - 1], "norm")[0] for n in n_range]
)

fig4, (ax4a, ax4b, ax4c) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 4a – Empirical mean
ax4a.plot(n_range, emp_mean, color="navy", linewidth=1.2)
ax4a.axhline(0, color="red", linestyle="--", linewidth=1.5, label="target = 0")
ax4a.set_ylabel("Empirical mean of $Z_n$", fontsize=11)
ax4a.set_title("Figure 4a – Convergence of $E[Z_n]$  →  0", fontsize=12)
ax4a.legend(fontsize=10)
ax4a.grid(True, alpha=0.3)

# 4b – Empirical standard deviation
ax4b.plot(n_range, emp_std, color="darkgreen", linewidth=1.2)
ax4b.axhline(1, color="red", linestyle="--", linewidth=1.5, label="target = 1")
ax4b.set_ylabel("Empirical σ of $Z_n$", fontsize=11)
ax4b.set_title("Figure 4b – Convergence of σ($Z_n$)  →  1", fontsize=12)
ax4b.legend(fontsize=10)
ax4b.grid(True, alpha=0.3)

# 4c – KS statistic
ax4c.plot(n_range, ks_stat, color="purple", linewidth=1.2)
ax4c.axhline(0, color="red", linestyle="--", linewidth=1.5, label="perfect match")
ax4c.set_ylabel("KS statistic (vs. N(0,1))", fontsize=11)
ax4c.set_xlabel("n  (number of trials)", fontsize=11)
ax4c.set_title(
    "Figure 4c – Kolmogorov-Smirnov distance to the standard normal",
    fontsize=12,
)
ax4c.legend(fontsize=10)
ax4c.grid(True, alpha=0.3)

plt.tight_layout()
fig4.savefig("bernoulli_convergence.png", dpi=150, bbox_inches="tight")
print("  → saved bernoulli_convergence.png")
plt.close(fig4)


# ---------------------------------------------------------------------------
# Summary to console
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Simulation Summary")
print("=" * 60)
for nv in N_VALUES:
    vals = Z[:, nv - 1]
    ks_val = stats.kstest(vals, "norm")[0]
    print(f"  n = {nv:>5}   mean(Z) = {np.mean(vals):.4f}   "
          f"std(Z) = {np.std(vals, ddof=1):.4f}   "
          f"KS(N(0,1)) = {ks_val:.4f}")
print("=" * 60)
print("All figures saved.")



