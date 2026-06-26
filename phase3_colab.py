# =====================================================================
# PHASE 3 — THE YIELD CURVE  (Colab-ready)
# Fit the Nelson-Siegel model to Pakistan's yield curve at key crisis dates.
# Uses the Diebold-Li method: fix the decay parameter (lambda) so the three
# coefficients fit stably by ordinary least squares and stay comparable.
#
# DATA NOTE: these are representative term-structure snapshots assembled from
# published SBP T-bill / PIB auction cut-offs and secondary-market reference
# yields around each date. Swap in precise PKRV data for a production tool;
# the fitting engine is identical.
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

mats = np.array([0.25, 0.5, 1.0, 3.0, 5.0, 10.0])   # maturities (years)
LAM = 2.0   # fixed decay parameter

curves = {
    "Jul 2021 (policy 7%, calm)":          [7.2, 7.4, 7.6, 8.8, 9.5, 10.3],
    "Dec 2022 (policy 16%, stress)":       [16.8, 16.9, 16.7, 14.2, 13.6, 13.3],
    "Sep 2023 (policy 22%, inverted)":     [23.5, 23.8, 23.0, 18.5, 16.5, 15.5],
    "Jun 2026 (policy 11%, recovery)":     [11.5, 12.0, 12.0, 12.8, 12.9, 12.3],
}

def ns_loadings(tau, lam):
    t = tau / lam
    f1 = (1 - np.exp(-t)) / t        # slope loading
    f2 = f1 - np.exp(-t)             # curvature loading
    return np.column_stack([np.ones_like(tau), f1, f2])

def ns_curve(tau, b0, b1, b2, lam):
    t = tau / lam
    f1 = (1 - np.exp(-t)) / t
    return b0 + b1*f1 + b2*(f1 - np.exp(-t))

# --- Fit each curve by least squares ---
X = ns_loadings(mats, LAM)
params = {}
print("NELSON-SIEGEL FITS (Diebold-Li, fixed lambda = 2.0)")
for name, ys in curves.items():
    ys = np.array(ys, dtype=float)
    betas, *_ = np.linalg.lstsq(X, ys, rcond=None)
    params[name] = betas
    b0, b1, b2 = betas
    spread = ys[-1] - ys[3]
    print(f"\n{name}")
    print(f"  level beta0     = {b0:6.2f}%")
    print(f"  slope beta1     = {b1:6.2f}   (positive = inverted)")
    print(f"  curvature beta2 = {b2:6.2f}")
    print(f"  10Y-3Y spread   = {spread:+.1f}%  ({'INVERTED' if spread<0 else 'upward'})")

# --- Plot the curves ---
colors = ["#2ec4b6", "#e0aa3e", "#e63946", "#5aa9e6"]
smooth = np.linspace(0.1, 10, 200)
plt.figure(figsize=(11, 6))
for (name, ys), c in zip(curves.items(), colors):
    b0, b1, b2 = params[name]
    plt.scatter(mats, ys, color=c, s=45, zorder=3)
    plt.plot(smooth, ns_curve(smooth, b0, b1, b2, LAM), color=c, lw=2.2, label=name)
plt.title("Pakistan's Yield Curve Through the Crisis", fontweight="bold")
plt.xlabel("Maturity (years)")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot the slope coefficient over time ---
names = list(params.keys())
b1s = [params[n][1] for n in names]
short = [n.split(" (")[0] for n in names]
plt.figure(figsize=(11, 5))
plt.bar(short, b1s, color=["#e63946" if b > 0 else "#2ec4b6" for b in b1s])
plt.axhline(0, color="black", lw=1)
plt.title("The Inversion Gauge: Nelson-Siegel Slope Coefficient", fontweight="bold")
plt.ylabel("beta1 (positive = inverted)")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
