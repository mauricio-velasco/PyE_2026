"""Generate a picture of the densities of the sample mean of n i.i.d.
standard normal random variables, for n = 1, 5, 10, 50, all on the SAME axes
so that the increasing concentration around the mean (0) as n grows is visible.

For i.i.d. X_j ~ N(0,1) the sample mean
    S_n = (X_1 + ... + X_n) / n  ~  N(0, 1/n),
so its density is
    f_n(x) = (sqrt(n) / sqrt(2 pi)) * exp(-n x^2 / 2),
with standard deviation 1/sqrt(n) and peak height sqrt(n)/sqrt(2 pi):

    n    sd = 1/sqrt(n)    peak = sqrt(n)/sqrt(2 pi)
    1    1.000             0.399
    5    0.447             0.892
    10   0.316             1.261
    50   0.141             2.821

Produces (in the same directory as this script):
  - concentration_average_normals.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 200
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.6, "figure.dpi": DPI})

# (n, color) -- palette consistent with make_distribution_plots.py
cases = [
    (1, "#4C72B0"),
    (5, "#55A868"),
    (10, "#C44E52"),
    (50, "#8172B3"),
]

# wide enough to fully show the n = 1 curve while the larger-n curves
# concentrate at 0; all densities are centred at x = 0 (their mean).
x = np.linspace(-1.5, 1.5, 2000)

fig, ax = plt.subplots(figsize=(8.5, 5))
for n, color in cases:
    sigma = 1.0 / np.sqrt(n)  # standard deviation of S_n
    ax.plot(x, norm.pdf(x, loc=0.0, scale=sigma), color=color,
            linewidth=2.2, label=r"$n=%d$" % n)

# all means coincide at 0 (consistent dashed line with the gaussian plot)
ax.axvline(0, color="0.35", linestyle="--", linewidth=1.3, label=r"media $0$")

ax.set_xlabel("$x$")
ax.set_ylabel(r"$f_{S_n}(x)$")
ax.set_title(r"Densidad de la media $S_n$ de $n$ normales estándar "
             r"($X_j\sim N(0,1)$)")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "concentration_average_normals.png"),
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

path = os.path.join(OUT, "concentration_average_normals.png")
print(f"Saved: {path}  ({os.path.getsize(path)} bytes)")
