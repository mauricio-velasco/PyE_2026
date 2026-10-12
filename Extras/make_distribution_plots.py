"""Generate PNG pictures of common 1D probability densities for Material_P1.tex.

Produces (in the same directory as this script):
  - binomial_pmf.png : Bin(40, 1/4) and Bin(90, 1/3)  [PMF, overlaid on one axes, mean lines]
  - poisson_pmf.png  : Pois(10) and Pois(20)           [PMF, overlaid on one axes, mean lines]
  - gaussian_pdf.png : N(0,1), N(0,2), N(3,1)         [PDF, vertical lines at the means]
  - bivariate_gaussian_same_marginals.png :
      two 2D Gaussians with the SAME marginals (X,Y ~ N(0,1)) but DIFFERENT
      joint distributions (rho = +0.8 vs rho = -0.8), with the shared marginal
      density curves drawn on the top and right edges of each panel.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binom, poisson, norm

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 200
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.6, "figure.dpi": DPI})

MEAN_COLOR = "#C44E52"  # consistent red for the mean line


# ---------------------------------------------------------------------------
# (1) Binomial  -- two cases OVERLAID on a single axes to show the range of the
#     family:  Bin(40, 1/4) [narrow, mean 10]  and  Bin(90, 1/3) [wide, mean 30]
# ---------------------------------------------------------------------------
binom_cases = [
    {"n": 40, "p": 0.25,  "color": "#4C72B0", "mean": 10,
     "name":  r"$\mathrm{Binomial}(40,\,\frac{1}{4})$",
     "m_lab": r"media $\mu=np=10$"},
    {"n": 90, "p": 1 / 3, "color": "#55A868", "mean": 30,
     "name":  r"$\mathrm{Binomial}(90,\,\frac{1}{3})$",
     "m_lab": r"media $\mu=np=30$"},
]

fig, ax = plt.subplots(figsize=(8.8, 4.9))
for c in binom_cases:
    k = np.arange(0, c["n"] + 1)
    pmf = binom.pmf(k, c["n"], c["p"])
    ax.bar(k, pmf, width=0.72, align="center", color=c["color"], alpha=0.55,
           edgecolor="white", linewidth=0.4, label=c["name"])
    ax.axvline(c["mean"], color=c["color"], linestyle="--", linewidth=1.8,
               label=c["m_lab"])
ax.set_xlabel("k")
ax.set_ylabel(r"$\mathbb{P}\{X=k\}$")
ax.set_title("Distribución binomial (dos casos superpuestos)")
ax.set_xlim(-1, 48)
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=10, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "binomial_pmf.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# (2) Poisson  -- two cases OVERLAID on a single axes:  Pois(10)  and  Pois(20)
# ---------------------------------------------------------------------------
pois_cases = [
    {"lam": 10, "color": "#4C72B0", "name": r"$\mathrm{Poisson}(10)$", "m_lab": r"media $\mu=10$"},
    {"lam": 20, "color": "#55A868", "name": r"$\mathrm{Poisson}(20)$", "m_lab": r"media $\mu=20$"},
]

fig, ax = plt.subplots(figsize=(8.8, 4.9))
for c in pois_cases:
    lam = c["lam"]
    kmax = int(np.ceil(lam + 6 * np.sqrt(lam)))
    k = np.arange(0, kmax + 1)
    pmf = poisson.pmf(k, lam)
    ax.bar(k, pmf, width=0.72, align="center", color=c["color"], alpha=0.55,
           edgecolor="white", linewidth=0.4, label=c["name"])
    ax.axvline(lam, color=c["color"], linestyle="--", linewidth=1.8, label=c["m_lab"])
ax.set_xlabel("k")
ax.set_ylabel(r"$\mathbb{P}\{X=k\}$")
ax.set_title("Distribución de Poisson (dos casos superpuestos)")
ax.set_xlim(-1, 49)
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=10, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "poisson_pmf.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# (3) Gaussian  N(0,1), N(0,2), N(3,1)
# ---------------------------------------------------------------------------
cases = [
    ("$\\mu=0,\\;\\sigma=1$", 0, 1, "#4C72B0"),
    ("$\\mu=0,\\;\\sigma=2$", 0, 2, "#55A868"),
    ("$\\mu=3,\\;\\sigma=1$", 3, 1, "#C44E52"),
]
x = np.linspace(-6, 9, 2000)

fig, ax = plt.subplots(figsize=(8.5, 5))
for label, mu, sigma, color in cases:
    ax.plot(x, norm.pdf(x, loc=mu, scale=sigma), color=color, linewidth=2.2, label=label)

# vertical lines at the (distinct) means
for mu in sorted({c[1] for c in cases}):
    ax.axvline(mu, color="0.35", linestyle="--", linewidth=1.3)

# legend with the curves plus a handle for the mean line
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.3))
labels.append("media $\\mu$")
ax.legend(handles, labels, loc="upper right", fontsize=10)

ax.set_xlabel("x")
ax.set_ylabel("$f_X(x)$")
ax.set_title("Distribución gaussiana (normal)")
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gaussian_pdf.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# (4) Bivariate Gaussian -- SAME marginals, DIFFERENT joint distribution.
#     Both panels have X ~ N(0,1) and Y ~ N(0,1) (identical marginals, shown as
#     the curves on the top and right edges), but different correlation:
#     rho = +0.8 (left) vs rho = -0.8 (right) -> different joint distributions.
#     The two panels use the SAME axis range so the only difference is the shape
#     of the joint density (i.e. the correlation).
# ---------------------------------------------------------------------------
from scipy.stats import multivariate_normal

def _draw_biv_panel(fig, x0, y0, w, h, rho, cmap):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov)
    xs = np.linspace(-4, 4, 401)
    X, Y = np.meshgrid(xs, xs)
    Z = np.column_stack([X.ravel(), Y.ravel()])
    P = mvn.pdf(Z).reshape(X.shape)
    marg = norm.pdf(xs, loc=0.0, scale=1.0)  # identical for both panels

    # Main panel: 2D joint density (filled contour + a few contour lines)
    ax = fig.add_axes([x0, y0, w, h])
    ax.contourf(X, Y, P, levels=18, cmap=cmap, alpha=0.92)
    ax.contour(X, Y, P, levels=7, colors="0.15", linewidths=0.6, alpha=0.55)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.grid(False)

    # Top marginal  (density of X) -- identical in both panels
    gap = 0.04
    ax_top = fig.add_axes([x0, y0 + h + gap, w, 0.08], sharex=ax)
    ax_top.plot(xs, marg, color="0.28", lw=1.5)
    ax_top.fill_between(xs, 0, marg, color="0.28", alpha=0.28)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="y", length=0, labelleft=False)
    ax_top.set_ylim(0, 1.0)
    ax_top.grid(False)
    ax_top.set_ylabel(r"$f_X(x)$", fontsize=9)

    # Right marginal (density of Y) -- identical in both panels
    ax_right = fig.add_axes([x0 + w + gap, y0, 0.08, h], sharey=ax)
    ax_right.plot(marg, xs, color="0.28", lw=1.5)
    ax_right.fill_betweenx(xs, 0, marg, color="0.28", alpha=0.28)
    ax_right.set_yticks([])
    ax_right.tick_params(axis="x", length=0, labelbottom=False)
    ax_right.set_xlim(0, 1.0)
    ax_right.grid(False)
    ax_right.set_xlabel(r"$f_Y(y)$", fontsize=9)


fig = plt.figure(figsize=(11.0, 5.4))
_w, _h, _y0 = 0.2749, 0.56, 0.16
_draw_biv_panel(fig, 0.08,  _y0, _w, _h, +0.8, "Blues")
_draw_biv_panel(fig, 0.545, _y0, _w, _h, -0.8, "Oranges")
fig.text(0.08 + _w / 2,  0.895, r"$\rho = +0.8$", ha="center",
         fontsize=13, weight="bold")
fig.text(0.545 + _w / 2, 0.895, r"$\rho = -0.8$", ha="center",
         fontsize=13, weight="bold")
fig.suptitle("Gaussianas con igual marginales ($X, Y \\sim N(0,1)$) "
             "y distinta conjunta", fontsize=12.5, y=0.985)
fig.savefig(os.path.join(OUT, "bivariate_gaussian_same_marginals.png"),
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Saved:")
for name in ("binomial_pmf.png", "poisson_pmf.png", "gaussian_pdf.png",
             "bivariate_gaussian_same_marginals.png"):
    path = os.path.join(OUT, name)
    print(f"  {path}  ({os.path.getsize(path)} bytes)")
