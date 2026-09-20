#!/usr/bin/env python3
"""
gliv_cant.py
============
Compares the theoretical binomial PMF with empirical histograms
from K repeated simulations of a Binomial(n=100, p=1/3) random variable.

Six individual figures are produced (one per K value).  As K grows, the
empirical histogram converges toward the theoretical PMF.

Parameters
----------
n  : number of Bernoulli trials per simulation  (100)
p  : success probability ("tails") per trial     (1/3)
K  : number of independent simulation runs       (50 … 10000)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)

N           = 100          # trials per simulation
P           = 1 / 3        # success probability
K_VALUES    = [50, 100, 150, 500, 1000, 10000]  # simulation counts

# ---------------------------------------------------------------------------
# 1.  Theoretical binomial PMF
# ---------------------------------------------------------------------------
k_vals = np.arange(0, N + 1)                       # possible outcomes 0..100
pmf_vals = stats.binom.pmf(k_vals, N, P)           # exact probabilities

# Integer-aligned bin edges: each bar centres on one integer outcome
bin_edges = np.arange(-0.5, N + 1.5)

# ---------------------------------------------------------------------------
# 2.  Generate individual figures
# ---------------------------------------------------------------------------
for K in K_VALUES:
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- empirical samples ---
    samples = stats.binom.rvs(N, P, size=K)

    # Weights turn the histogram into probability estimates (count / K)
    weights = np.ones(K) / K

    ax.hist(samples, bins=bin_edges, weights=weights,
            alpha=0.6, color="red", edgecolor="grey",
            label=f"Empirical (K={K})")

    # --- theoretical PMF overlay ---
    ax.step(k_vals, pmf_vals, where="mid", color="dodgerblue",
            linewidth=1.5, label="Theoretical PMF")

    ax.set_title(f"Binomial({N}, {P:.4f}) — K = {K} simulations", fontsize=13)
    ax.set_xlabel("Number of tails ($k$)", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xlim(20, 70)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = f"gliv_cant_K{K}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → saved {out}")
    plt.close(fig)
