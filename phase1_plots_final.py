"""
PHASE 1 - PLOTS
When Models Break: Quantifying Discontinuous Risk in Pakistan's Financial Markets

Produces four charts in a clean, minimalist academic style:
    1. KSE-100 price journey (2018-2026)
    2. Daily log returns with crisis event labels
    3. Return distribution vs fitted normal curve
    4. Q-Q plot

Run phase1_analysis.py first to generate kse100_clean.csv.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------------------------------------------------
# AESTHETIC: minimalist white background, black lines, serif font
# No decorative colours -- series distinguished by line style and greyscale
# -----------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.6,
    "font.size": 11,
})
INK = "#1a1a1a"
GREY = "#8a8a8a"
MIDGREY = "#bdbdbd"

# -----------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------
df = pd.read_csv("kse100_clean.csv", parse_dates=["Date"])
df = df.dropna(subset=["log_return"]).reset_index(drop=True)
returns = df["log_return"]
mean_r = returns.mean()
std_r = returns.std()

# -----------------------------------------------------------------------
# CHART 1: Price journey
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df["Date"], df["Close"], color=INK, linewidth=0.9)
ax.set_ylabel("Index level (points)")
ax.set_title("KSE-100 Index — The Price Journey (2018–2026)",
             fontsize=12, loc="left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("chart1_price.png", dpi=150)
plt.show()
print("Saved: chart1_price.png")

# -----------------------------------------------------------------------
# CHART 2: Daily returns with crisis labels
# -----------------------------------------------------------------------
CRISES = [
    ("2020-03-24", "COVID-19 crash",        "2019-02-01", -12.5),
    ("2023-07-03", "IMF Stand-By deal",     "2022-02-01",  11.0),
    ("2025-05-08", "India-Pakistan\nconflict", "2023-10-01", -14.0),
    ("2025-05-12", "Ceasefire +\nIMF rally",   "2024-09-01",  12.5),
    ("2026-03-02", "Iran strikes\ncrash",    "2026-01-01", -15.5),
]

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(df["Date"], returns * 100, color=INK, linewidth=0.5)
ax.axhline(5 * std_r * 100,  color=GREY, ls="--", lw=0.8)
ax.axhline(-5 * std_r * 100, color=GREY, ls="--", lw=0.8)
ax.text(df["Date"].iloc[5],  5 * std_r * 100 + 0.4, "+5σ", color=GREY, fontsize=8)
ax.text(df["Date"].iloc[5], -5 * std_r * 100 - 1.2, "−5σ", color=GREY, fontsize=8)

for date, label, tx, ty in CRISES:
    d = pd.to_datetime(date)
    t = pd.to_datetime(tx)
    row = df.loc[df["Date"] == d]
    if row.empty:
        continue
    val = row["log_return"].iloc[0] * 100
    ax.annotate(label, xy=(d, val), xytext=(t, ty),
                color=INK, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                connectionstyle="arc3,rad=0.1"))

ax.set_ylabel("Daily return (%)")
ax.set_ylim(-17, 14)
ax.set_title("Daily log returns — every extreme day has a name",
             fontsize=12, loc="left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("chart2_returns.png", dpi=150)
plt.show()
print("Saved: chart2_returns.png")

# -----------------------------------------------------------------------
# CHART 3: Return distribution vs normal
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(returns * 100, bins=110, density=True,
        color=MIDGREY, edgecolor="none")
x = np.linspace(returns.min(), returns.max(), 400)
ax.plot(x * 100, stats.norm.pdf(x, mean_r, std_r) / 100,
        color=INK, lw=1.5, label="Normal distribution")
ax.set_xlim(-12, 12)
ax.set_xlabel("Daily return (%)")
ax.set_ylabel("Density")
ax.set_title("Return distribution vs normal — taller peak, fatter tails",
             fontsize=12, loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("chart3_distribution.png", dpi=150)
plt.show()
print("Saved: chart3_distribution.png")

# -----------------------------------------------------------------------
# CHART 4: Q-Q plot
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
(osm, osr), (sl, ic, _) = stats.probplot(returns, dist="norm")
ax.scatter(osm, osr * 100, s=7, color=GREY, alpha=0.6)
lx = np.array([osm.min(), osm.max()])
ax.plot(lx, (sl * lx + ic) * 100, color=INK, lw=1.4, label="Perfect normal")
ax.set_xlabel("Theoretical normal quantiles")
ax.set_ylabel("Observed return quantiles (%)")
ax.set_title("Q–Q plot — where reality departs from the model",
             fontsize=12, loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("chart4_qqplot.png", dpi=150)
plt.show()
print("Saved: chart4_qqplot.png")

print()
print("All four charts saved.")
