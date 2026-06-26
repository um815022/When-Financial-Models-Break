# =====================================================================
# PHASE 2 — THE MODEL  (Colab-ready)
# Black-Scholes + Merton Jump-Diffusion, calibrated to KSE-100 data.
# Requires kse100_clean.csv (from Phase 1) in the same Colab session.
# =====================================================================
import pandas as pd
import numpy as np
from scipy.stats import norm
from math import factorial, exp, log

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# 1. CALIBRATE the jump model to real KSE-100 returns
# ---------------------------------------------------------------------
df = pd.read_csv("kse100_clean.csv")
r = df["log_return"].dropna().values
n_days = len(r)

daily_mean = r.mean()
daily_std = r.std()
z = (r - daily_mean) / daily_std

# A "jump day" is any day beyond 3 standard deviations.
is_jump = np.abs(z) > 3
calm = r[~is_jump]
jumps = r[is_jump]

sigma_diff = calm.std() * np.sqrt(TRADING_DAYS)      # calm-market volatility (annual)
lam = (is_jump.sum() / n_days) * TRADING_DAYS         # jumps per year
mu_J = jumps.mean()                                   # average jump size (log)
sigma_J = jumps.std()                                 # jump size volatility
raw_vol = daily_std * np.sqrt(TRADING_DAYS)           # total volatility (annual)

print("CALIBRATED MERTON PARAMETERS (KSE-100)")
print("  Diffusion vol  :", round(sigma_diff, 4))
print("  Jump intensity :", round(lam, 2), "per year")
print("  Mean jump size :", round(mu_J, 5))
print("  Jump vol       :", round(sigma_J, 4))
print("  Total vol check:", round(np.sqrt(sigma_diff**2 + lam*(mu_J**2 + sigma_J**2)), 4),
      "vs raw", round(raw_vol, 4))
print()

# ---------------------------------------------------------------------
# 2. THE TWO PRICERS
# ---------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, opt="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def merton(S, K, T, r, sigma, lam, mu_J, sigma_J, opt="call", N=80):
    # Closed-form Merton: a Poisson-weighted sum of Black-Scholes prices.
    k = exp(mu_J + 0.5*sigma_J**2) - 1
    lam2 = lam * (1 + k)
    price = 0.0
    for n in range(N):
        sigma_n = np.sqrt(sigma**2 + n*sigma_J**2/T)
        r_n = r - lam*k + n*log(1+k)/T
        weight = exp(-lam2*T) * (lam2*T)**n / factorial(n)
        price += weight * black_scholes(S, K, T, r_n, sigma_n, opt)
    return price

# ---------------------------------------------------------------------
# 3. THE HEADLINE: how much Black-Scholes underprices crash protection
# ---------------------------------------------------------------------
S, T, r_rate = 100.0, 0.25, 0.105    # spot 100, 3 months, Pakistan rate 10.5%

print("DEEP OUT-OF-THE-MONEY PUTS (crash protection)")
print(f"{'Strike':>7} {'% OTM':>7} {'Black-Scholes':>14} {'Merton':>9} {'Underpriced':>12}")
for K in [95, 90, 85, 80, 75, 70]:
    bs_p = black_scholes(S, K, T, r_rate, raw_vol, "put")
    m_p = merton(S, K, T, r_rate, sigma_diff, lam, mu_J, sigma_J, "put")
    print(f"{K:>7} {100*(S-K)/S:>6.0f}% {bs_p:>14.4f} {m_p:>9.4f} {100*(m_p-bs_p)/bs_p:>10.1f}%")

# ---------------------------------------------------------------------
# 4. MONTE CARLO — validate the model and compare crash probabilities
# ---------------------------------------------------------------------
np.random.seed(42)
M = 200000
k = exp(mu_J + 0.5*sigma_J**2) - 1

# Black-Scholes world (smooth geometric Brownian motion)
Z = np.random.normal(size=M)
S_gbm = S * np.exp((r_rate - 0.5*raw_vol**2)*T + raw_vol*np.sqrt(T)*Z)

# Merton world (diffusion + Poisson jumps)
Zd = np.random.normal(size=M)
Nj = np.random.poisson(lam*T, size=M)
jump_sum = np.array([np.sum(np.random.normal(mu_J, sigma_J, n)) if n > 0 else 0.0 for n in Nj])
S_mert = S * np.exp((r_rate - 0.5*sigma_diff**2 - lam*k)*T + sigma_diff*np.sqrt(T)*Zd + jump_sum)

K = 80
mc_put = np.exp(-r_rate*T) * np.mean(np.maximum(K - S_mert, 0))
cf_put = merton(S, K, T, r_rate, sigma_diff, lam, mu_J, sigma_J, "put")
print()
print(f"Monte Carlo check (strike-80 put): MC={mc_put:.4f}  closed-form={cf_put:.4f}")
print("These should match closely, validating the model.")
