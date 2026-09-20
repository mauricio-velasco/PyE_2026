#!/usr/bin/env python3
"""
binomial_pic_maker.py
=====================
Computes the binomial probability mass function (PMF) for the number of
successes in n Bernoulli trials with success probability p, and plots
all of its values.

    P(X = k) = C(n, k) · p^k · (1 - p)^(n - k),   k = 0, 1, …, n

with n = 100 and p = 1/3.  The distribution has

    mean    μ = n·p
    stddev  σ = sqrt(n·p·(1 - p))

Two figures are saved:

    binomial_pmf_n100_p13.png           — the binomial PMF (blue)
    binomial_pmf_n100_p13_normal.png    — binomial PMF (blue) with the
                                          normal approximation N(μ, σ²)
                                          overlaid in red
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------
N = 100          # number of Bernoulli trials
P = 1 / 3        # success probability

# ---------------------------------------------------------------------------
# 1.  Compute the binomial PMF for every k = 0, 1, …, N
# ---------------------------------------------------------------------------
print(f"Computing binomial PMF  (n = {N}, p = {P:g}) …")

k = np.arange(0, N + 1)                       # possible numbers of successes
pmf = stats.binom.pmf(k, n=N, p=P)            # P(X = k) for each k

# Basic sanity checks
assert len(k) == N + 1
assert np.allclose(pmf.sum(), 1.0), "PMF must sum to 1"
k_mode = int(k[np.argmax(pmf)])
mu = N * P
sigma = np.sqrt(N * P * (1 - P))

# ---------------------------------------------------------------------------
# 2.  Plot the PMF, zoomed in on k = 0 … K_MAX
# ---------------------------------------------------------------------------
K_MAX = 60  # zoom range; P(X > K_MAX) ≈ 10⁻¹⁰, so nothing meaningful is cut

k_plot = k[k <= K_MAX]
pmf_plot = pmf[k <= K_MAX]
tail_mass = pmf.sum() - pmf_plot.sum()   # probability cut off by the zoom

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(k_plot, pmf_plot, width=0.8, color="dodgerblue", edgecolor="navy",
       alpha=0.75, label=r"$P(X=k) = \binom{n}{k}p^{k}(1-p)^{n-k}$")

# Mark the mean and the ±1σ envelope
ax.axvline(mu, color="red", linewidth=1.5, linestyle="--",
           label=f"Mean  $\\mu = np = {mu:.2f}$")
ax.fill_between(k_plot, 0, mu,
                where=(k_plot >= mu - sigma) & (k_plot <= mu + sigma),
                color="orange", alpha=0.15,
                label=r"$\pm 1\sigma$ envelope")

ax.set_title(f"Binomial PMF  ($n = {N}$ trials,  $p = {P:g}$)  —  "
             f"$P(X = k)$ for $k = 0$ … {K_MAX}  "
             f"(tail beyond: {tail_mass:.1e})", fontsize=13)
ax.set_xlabel("Number of successes $k$", fontsize=12)
ax.set_ylabel("Probability $P(X = k)$", fontsize=12)
ax.set_xlim(-1.5, K_MAX + 1.5)
ax.set_ylim(0, pmf.max() * 1.15)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fname = "binomial_pmf_n100_p13.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"  → saved {fname}")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2b. Second plot: binomial PMF with the normal approximation overlaid
# ---------------------------------------------------------------------------
# The normal approximation to Bin(n, p) is N(μ, σ²) with
#   μ = n·p,   σ² = n·p·(1 - p).
# We draw the binomial PMF in blue (as before) and overlay the smooth
# normal density in red.
print("  plotting binomial PMF + normal approximation (red) …")

fig2, ax2 = plt.subplots(figsize=(11, 6))

ax2.bar(k_plot, pmf_plot, width=0.8, color="dodgerblue", edgecolor="navy",
        alpha=0.5, label="Binomial PMF")
ax2.plot(k_plot, pmf_plot, color="navy", linewidth=1.0, marker="o",
         markersize=2.5, label=r"Binomial $\binom{n}{k}p^k(1-p)^{n-k}$")

# Smooth normal density evaluated on a continuous grid across the same range
x = np.linspace(k_plot.min() - 1.5, k_plot.max() + 1.5, 1000)
norm_pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
ax2.plot(x, norm_pdf, color="red", linewidth=2.5,
         label=r"Normal approx. $N(\mu,\sigma^2)$, "
               r"$\mu=np$, $\sigma^2=np(1-p)$")

# Mark the common mean
ax2.axvline(mu, color="red", linewidth=1.0, linestyle=":", alpha=0.8)

ax2.set_title(f"Binomial PMF vs. normal approximation  ($n = {N}$, $p = {P:g}$)\n"
              f"$\\mu = np = {mu:.2f}$,  "
              f"$\\sigma = \\sqrt{{np(1-p)}} = {sigma:.2f}$", fontsize=13)
ax2.set_xlabel("Number of successes $k$", fontsize=12)
ax2.set_ylabel("Probability / density", fontsize=12)
ax2.set_xlim(-1.5, K_MAX + 1.5)
ax2.set_ylim(0, max(pmf_plot.max(), norm_pdf.max()) * 1.15)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fname2 = "binomial_pmf_n100_p13_normal.png"
fig2.savefig(fname2, dpi=150, bbox_inches="tight")
print(f"  → saved {fname2}")
plt.close(fig2)

# ---------------------------------------------------------------------------
# 3.  Summary to console
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Binomial PMF Summary")
print("=" * 60)
print(f"  n = {N}          p = {P:.6f}          q = 1-p = {1-P:.6f}")
print(f"  mean   μ = n·p            = {mu:.4f}")
print(f"  stddev σ = √(n·p·(1-p))   = {sigma:.4f}")
print(f"  mode   k*                 = {k_mode}")
print(f"  P(X = {k_mode})                       = {pmf[k_mode]:.6f}")
print(f"  sum of all {len(k)} PMF values       = {pmf.sum():.6f}  (≈ 1 ✓)")
print("=" * 60)
print("All figures saved.")