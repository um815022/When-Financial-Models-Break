"""
PHASE 1 - THE EVIDENCE
When Models Break: Quantifying Discontinuous Risk in Pakistan's Financial Markets

Goal: prove statistically that KSE-100 daily returns do not follow a normal
distribution -- the core assumption of Black-Scholes and standard quant finance.

Steps:
    1. Load and clean KSE-100 data
    2. Compute log returns
    3. Descriptive statistics (skewness, kurtosis)
    4. Jarque-Bera normality test
    5. Identify extreme 'jump' days

Run this before phase1_plots.py -- it produces kse100_clean.csv which
the plots script and the Streamlit app both depend on.
"""

import pandas as pd
import numpy as np
from scipy import stats

# -----------------------------------------------------------------------
# STEP 1 -- LOAD AND CLEAN
# -----------------------------------------------------------------------
# The investing.com CSV file may have slightly different names on different
# devices (e.g. "Karachi 100 Historical Data (2).csv"). Adjust if needed.
import glob, os
candidates = (glob.glob("Karachi 100 Historical Data*.csv") +
              glob.glob("Karachi_100_Historical_Data*.csv") +
              glob.glob("kse100_clean*.csv"))
if not candidates:
    raise FileNotFoundError(
        "Cannot find data file. Make sure 'Karachi 100 Historical Data.csv' "
        "or 'kse100_clean.csv' is in the same folder."
    )
filepath = sorted(candidates)[0]
raw = pd.read_csv(filepath)
raw.columns = raw.columns.str.strip().str.replace('"', '').str.replace('\ufeff', '')

# Handle both the raw investing.com format (Price column) and
# the already-processed format (Close column)
if "Price" in raw.columns:
    raw = raw[["Date", "Price"]].copy()
    raw.columns = ["Date", "Close"]
    raw["Close"] = raw["Close"].astype(str).str.replace(",", "").astype(float)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
elif "Close" in raw.columns:
    raw = raw[["Date", "Close"]].copy()
    raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw["Date"] = pd.to_datetime(raw["Date"])
else:
    raise ValueError("Unrecognised file format — expected a Date and Price/Close column.")

raw = raw.sort_values("Date").reset_index(drop=True)
raw = raw[raw["Date"] >= "2018-01-01"].reset_index(drop=True)

# -----------------------------------------------------------------------
# STEP 2 -- LOG RETURNS
# -----------------------------------------------------------------------
# We use log returns: r_t = ln(P_t / P_{t-1})
# Why logs:
#   (a) They add up across time cleanly
#   (b) They are symmetric -- gains and losses are treated equivalently
#   (c) Black-Scholes assumes log returns are normal, so testing them
#       tests the model's core assumption directly
df = raw.copy()
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)
returns = df["log_return"]

print("=" * 60)
print("STEP 1-2: DATA LOADED")
print("=" * 60)
print(f"Trading days  : {len(df)}")
print(f"Date range    : {df['Date'].min().date()} to {df['Date'].max().date()}")
print()

# -----------------------------------------------------------------------
# STEP 3 -- DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------
mean_r = returns.mean()
std_r = returns.std()
skew_r = stats.skew(returns)
excess_kurt = stats.kurtosis(returns, fisher=True)

print("=" * 60)
print("STEP 3: DESCRIPTIVE STATISTICS")
print("=" * 60)
print(f"Mean (daily)        : {mean_r:.6f}")
print(f"Std dev (daily)     : {std_r:.6f}")
print(f"Std dev (annual)    : {std_r * np.sqrt(252):.4f}")
print(f"Skewness            : {skew_r:.4f}  (normal = 0)")
print(f"Excess kurtosis     : {excess_kurt:.4f}  (normal = 0, positive = fat tails)")
print()

# -----------------------------------------------------------------------
# STEP 4 -- JARQUE-BERA NORMALITY TEST
# -----------------------------------------------------------------------
# Combines skewness and kurtosis into a single formal test.
# Null hypothesis: data IS normally distributed.
# A p-value near zero means we REJECT normality with high confidence.
jb_stat, jb_p = stats.jarque_bera(returns)

print("=" * 60)
print("STEP 4: JARQUE-BERA NORMALITY TEST")
print("=" * 60)
print(f"JB statistic  : {jb_stat:,.1f}")
print(f"p-value       : {jb_p:.3e}")
print(f"Verdict       : {'REJECT normality -- returns are NOT normal' if jb_p < 0.05 else 'Cannot reject normality'}")
print()

# -----------------------------------------------------------------------
# STEP 5 -- EXTREME 'JUMP' DAYS
# -----------------------------------------------------------------------
# Under a normal distribution, a move beyond 5 standard deviations has
# probability ~1 in 1.7 million. We count how many actually occurred.
df["z_score"] = (df["log_return"] - mean_r) / std_r
p_5sigma = 2 * (1 - stats.norm.cdf(5))
expected = p_5sigma * len(returns)
extreme = df[df["z_score"].abs() > 5]

print("=" * 60)
print("STEP 5: EXTREME DAYS (beyond ±5 sigma)")
print("=" * 60)
print(f"Expected under normal : {expected:.4f}  (essentially never)")
print(f"Actually observed     : {len(extreme)}")
print()
print("The largest moves:")
worst = df.reindex(df["z_score"].abs().sort_values(ascending=False).index).head(10)
for _, row in worst.iterrows():
    print(f"  {row['Date'].date()}   {row['log_return']*100:+.2f}%   ({row['z_score']:+.1f} sigma)")

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
df.to_csv("kse100_clean.csv", index=False)
print()
print("Saved: kse100_clean.csv")
print("Run phase1_plots.py next to generate the charts.")
