"""
When Models Break — Quantifying Discontinuous Risk in Pakistan's Financial Markets
An interactive companion to the project. Self-contained: requires only this file,
kse100_clean.csv, and the packages in requirements.txt.
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from math import factorial, exp, log
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# PAGE CONFIG + MINIMAL STYLING
# --------------------------------------------------------------------------
st.set_page_config(page_title="When Models Break", layout="wide",
                   initial_sidebar_state="collapsed")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#333333", "ytick.color": "#333333", "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6, "font.size": 11,
})
INK = "#1a1a1a"; GREY = "#8a8a8a"; MIDGREY = "#bdbdbd"

st.markdown("""
<style>
  .stApp { background-color: #ffffff; }
  /* Force ALL headings dark regardless of Streamlit theme */
  h1, h2, h3, h4, h5, h6,
  [data-testid="stHeading"],
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
      font-family: Georgia, 'Times New Roman', serif !important;
      color: #1a1a1a !important;
  }
  /* Override Streamlit's default salmon/red subheader colour */
  [data-testid="stHeadingWithActionElements"] * { color: #1a1a1a !important; }
  p, li { font-family: Georgia, 'Times New Roman', serif; color: #2a2a2a; font-size: 1.02rem; }
  .stMetric { border: 1px solid #ececec; border-radius: 6px; padding: 10px; }
  /* Responsive title: smaller on mobile */
  .big-title {
      font-size: clamp(1.5rem, 5vw, 2.5rem);
      font-weight: 700; line-height: 1.15; margin-bottom: 0.2rem;
      color: #1a1a1a; word-break: break-word;
  }
  .subtitle {
      font-size: clamp(0.95rem, 3vw, 1.2rem);
      color: #555; font-style: italic; margin-bottom: 0.6rem;
  }
  .rule { border-top: 2px solid #1a1a1a; margin: 0.4rem 0 1.2rem 0; }
  .note { font-size: 0.9rem; color: #777; font-style: italic; }
  /* Prevent horizontal overflow on mobile */
  .block-container { overflow-x: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# DATA + CALIBRATION (cached)
# --------------------------------------------------------------------------
TRADING_DAYS = 252

@st.cache_data
def load_data():
    import glob, os
    # Accept any filename matching kse100_clean*.csv
    candidates = (glob.glob("kse100_clean*.csv") +
                  glob.glob("kse100_clean *.csv") +
                  glob.glob("kse100*.csv"))
    candidates = [f for f in candidates if os.path.isfile(f)]
    if not candidates:
        st.error("Data file not found. Make sure kse100_clean.csv is in the repository.")
        st.stop()
    filepath = sorted(candidates)[0]
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df

@st.cache_data
def calibrate(returns):
    r = returns.values
    n = len(r)
    mu, sd = r.mean(), r.std()
    z = (r - mu) / sd
    is_jump = np.abs(z) > 3
    calm, jumps = r[~is_jump], r[is_jump]
    sigma_diff = calm.std() * np.sqrt(TRADING_DAYS)
    lam = (is_jump.sum() / n) * TRADING_DAYS
    mu_J = jumps.mean(); sigma_J = jumps.std()
    raw_vol = sd * np.sqrt(TRADING_DAYS)
    return dict(sigma_diff=sigma_diff, lam=lam, mu_J=mu_J, sigma_J=sigma_J,
                raw_vol=raw_vol, mu=mu, sd=sd,
                skew=stats.skew(r), kurt=stats.kurtosis(r),
                jb_p=stats.jarque_bera(r)[1], n_jump=int(is_jump.sum()), n=n)

def bs(S, K, T, r, sigma, opt="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def merton(S, K, T, r, sigma, lam, mu_J, sigma_J, opt="call", N=70):
    k = exp(mu_J + 0.5*sigma_J**2) - 1
    lam2 = lam*(1+k); price = 0.0
    for n in range(N):
        sn = np.sqrt(sigma**2 + n*sigma_J**2/T)
        rn = r - lam*k + n*log(1+k)/T
        price += exp(-lam2*T)*(lam2*T)**n/factorial(n) * bs(S, K, T, rn, sn, opt)
    return price

df = load_data()
P = calibrate(df["log_return"])

# Crisis events (date, label) present in the series
CRISES = [
    ("2020-03-24", "COVID-19 crash"),
    ("2023-07-03", "IMF Stand-By deal"),
    ("2025-05-08", "India-Pakistan conflict"),
    ("2025-05-12", "Ceasefire + IMF rally"),
    ("2026-03-02", "Iran strikes crash"),
]

# Yield curve snapshots (representative, from SBP/market reference yields)
MATS = np.array([0.25, 0.5, 1.0, 3.0, 5.0, 10.0]); LAM = 2.0
CURVES = {
    "Jul 2021 (policy 7%, calm)":        [7.2, 7.4, 7.6, 8.8, 9.5, 10.3],
    "Dec 2022 (policy 16%, stress)":     [16.8, 16.9, 16.7, 14.2, 13.6, 13.3],
    "Sep 2023 (policy 22%, inverted)":   [23.5, 23.8, 23.0, 18.5, 16.5, 15.5],
    "Jun 2026 (policy 11%, recovery)":   [11.5, 12.0, 12.0, 12.8, 12.9, 12.3],
}
def ns_fit(ys):
    t = MATS/LAM; f1 = (1-np.exp(-t))/t
    X = np.column_stack([np.ones_like(MATS), f1, f1-np.exp(-t)])
    return np.linalg.lstsq(X, np.array(ys, float), rcond=None)[0]
def ns_curve(tau, b0, b1, b2):
    t = tau/LAM; f1 = (1-np.exp(-t))/t
    return b0 + b1*f1 + b2*(f1-np.exp(-t))

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<div class="big-title">When Models Break</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Quantifying Discontinuous Risk in Pakistan\'s Financial Markets</div>',
            unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    "The mathematics that prices risk in modern finance assumes markets move **smoothly and continuously**. "
    "Pakistan's markets do not: political upheaval, IMF negotiations, emergency rate decisions, regional conflict "
    "and currency shocks arrive as sudden **jumps**. This tool shows, with real data, how and where the standard "
    "models break down — across the stock market, the options market, and the government bond market."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["  1 · The Evidence  ", "  2 · The Model  ", "  3 · The Yield Curve  ", "  Conclusion  "])

# ==========================================================================
# TAB 1 — THE EVIDENCE
# ==========================================================================
with tab1:
    st.subheader("The returns of Pakistan's market are not normal")
    st.write(
        f"Across **{P['n']:,} trading days** of the KSE-100 index "
        f"({df['Date'].min().date()} to {df['Date'].max().date()}), daily returns depart sharply from the "
        "normal distribution that standard models assume. Four measures make the case.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skewness", f"{P['skew']:.2f}", "normal = 0", delta_color="off")
    c2.metric("Excess kurtosis", f"{P['kurt']:.2f}", "normal = 0", delta_color="off")
    c3.metric("Jarque–Bera p", f"{P['jb_p']:.3f}", "normal rejected", delta_color="off")
    c4.metric("Days beyond 5σ", f"{(np.abs((df['log_return']-P['mu'])/P['sd'])>5).sum()}",
              "≈0 expected", delta_color="off")

    st.markdown("###### Daily log returns — every extreme day has a name")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(df["Date"], df["log_return"]*100, color=INK, lw=0.5)
    ax.axhline(5*P['sd']*100, color=GREY, ls="--", lw=0.8)
    ax.axhline(-5*P['sd']*100, color=GREY, ls="--", lw=0.8)
    ax.text(df["Date"].iloc[5], 5*P['sd']*100+0.4, "+5σ", color=GREY, fontsize=8)
    ax.text(df["Date"].iloc[5], -5*P['sd']*100-1.1, "−5σ", color=GREY, fontsize=8)
    offs = {"2020-03-24": (-0.18, -11.5), "2023-07-03": (-0.10, 11),
            "2025-05-08": (-0.05, -13.5), "2025-05-12": (0.02, 12.5),
            "2026-03-02": (0.0, -15.5)}
    for date, label in CRISES:
        d = pd.to_datetime(date)
        row = df.loc[df["Date"] == d]
        if row.empty: continue
        val = row["log_return"].iloc[0]*100
        dx, ty = offs[date]
        ax.annotate(label, xy=(d, val),
                    xytext=(d + pd.Timedelta(days=int(dx*365)), ty),
                    color=INK, fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
    ax.set_ylabel("Daily return (%)"); ax.set_ylim(-17, 14)
    st.pyplot(fig, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("###### Return distribution vs normal")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.hist(df["log_return"]*100, bins=110, density=True, color=MIDGREY, edgecolor="none")
        x = np.linspace(df["log_return"].min(), df["log_return"].max(), 400)
        ax.plot(x*100, norm.pdf(x, P['mu'], P['sd'])/100, color=INK, lw=1.4,
                label="Normal")
        ax.set_xlim(-12, 12); ax.set_xlabel("Daily return (%)"); ax.set_ylabel("Density")
        ax.legend(frameon=False)
        st.pyplot(fig, use_container_width=True)
    with cc2:
        st.markdown("###### Q–Q plot: where reality leaves the model")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        (osm, osr), (sl, ic, _) = stats.probplot(df["log_return"], dist="norm")
        ax.scatter(osm, osr*100, s=6, color=GREY, alpha=0.6)
        lx = np.array([osm.min(), osm.max()])
        ax.plot(lx, (sl*lx+ic)*100, color=INK, lw=1.3, label="Perfect normal")
        ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Observed (%)")
        ax.legend(frameon=False)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<p class="note">Ordinary days sit on the line; the crashes curl away. '
                'That departure in the tails is exactly what the next section prices.</p>',
                unsafe_allow_html=True)

# ==========================================================================
# TAB 2 — THE MODEL
# ==========================================================================
with tab2:
    st.subheader("Pricing the risk the standard model cannot see")
    st.write(
        "The Black–Scholes model assumes smooth, normal returns. The **Merton Jump-Diffusion** model adds a "
        "Poisson jump process to capture the crises documented in Section 1. Calibrated to the KSE-100 data, "
        "the difference between the two reveals how badly the standard model misprices crash protection.")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Calm vol (annual)", f"{P['sigma_diff']*100:.1f}%")
    cc2.metric("Jumps per year (λ)", f"{P['lam']:.1f}")
    cc3.metric("Typical jump size", f"±{P['sigma_J']*100:.1f}%")
    cc4.metric("Total volatility", f"{P['raw_vol']*100:.1f}%")

    st.markdown("###### Try the pricer")
    pc1, pc2 = st.columns([1, 2])
    with pc1:
        spot = 100.0
        strike_pct = st.slider("Strike (% of spot)", 60, 130, 85, 1,
                               help="Below 100 = out-of-the-money put (crash protection).")
        months = st.slider("Months to expiry", 1, 12, 3, 1)
        rate = st.slider("Risk-free rate (%)", 5.0, 25.0, 10.5, 0.5,
                         help="Pakistan's policy rate: 22% at the 2023 peak, ~11% now.")
        K = spot * strike_pct/100; T = months/12; r = rate/100
        opt = "put" if strike_pct < 100 else "call"
        bs_p = bs(spot, K, T, r, P['raw_vol'], opt)
        m_p = merton(spot, K, T, r, P['sigma_diff'], P['lam'], P['mu_J'], P['sigma_J'], opt)
        gap = (m_p - bs_p)/bs_p*100 if bs_p > 1e-6 else float('nan')
        st.metric(f"Black–Scholes {opt}", f"{bs_p:.3f}")
        st.metric(f"Merton {opt}", f"{m_p:.3f}")
        st.metric("Black–Scholes underprices by", f"{gap:.0f}%" if np.isfinite(gap) else "—")
    with pc2:
        strikes = np.arange(70, 116, 2)
        gaps = []
        for k in strikes:
            o = "put" if k < 100 else "call"
            b = bs(spot, k, T, r, P['raw_vol'], o)
            m = merton(spot, k, T, r, P['sigma_diff'], P['lam'], P['mu_J'], P['sigma_J'], o)
            gaps.append((m-b)/b*100 if b > 1e-6 else np.nan)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        gaps_plot = [min(g, 300) if np.isfinite(g) else 0 for g in gaps]
        ax.bar(strikes, gaps_plot, width=1.5, color=MIDGREY, edgecolor=INK, linewidth=0.4)
        ax.axvline(strike_pct, color=INK, ls="--", lw=1, label="Your strike")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylim(bottom=min(0, min(gaps_plot)-5))
        ax.set_xlabel("Strike (spot = 100)"); ax.set_ylabel("Underpricing by Black–Scholes (%)")
        ax.legend(frameon=False)
        ax.set_title("Mispricing is concentrated in out-of-the-money puts", fontsize=11, loc="left")
        if any(g > 300 for g in gaps if np.isfinite(g)):
            ax.text(0.02, 0.97, "Note: bars capped at 300% for readability",
                    transform=ax.transAxes, fontsize=8, color=GREY, va="top")
        st.pyplot(fig, use_container_width=True)

    st.markdown('<p class="note">The deeper out-of-the-money the put (further left), the more severely '
                'Black–Scholes underprices the crash protection — by several hundred per cent in the tail.</p>',
                unsafe_allow_html=True)

# ==========================================================================
# TAB 3 — THE YIELD CURVE
# ==========================================================================
with tab3:
    st.subheader("The same crises, written into the bond market")
    st.write(
        "A **yield curve** plots the interest rate the government pays against how long it borrows for. Normally it "
        "slopes upward; when it **inverts** (short rates above long), it signals acute stress. Pakistan's curve "
        "produced one of the most extreme inversions of any country during its 2022–23 crisis. The **Nelson–Siegel** "
        "model summarises each curve in three numbers: level, slope, and curvature.")

    params = {n: ns_fit(ys) for n, ys in CURVES.items()}
    yc1, yc2 = st.columns(2)
    with yc1:
        st.markdown("###### The curve through the crisis")
        styles = ["-", "--", "-.", ":"]
        smooth = np.linspace(0.1, 10, 200)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for (name, ys), ls in zip(CURVES.items(), styles):
            b = params[name]
            ax.plot(smooth, ns_curve(smooth, *b), color=INK, lw=1.3, ls=ls, label=name.split(" (")[0])
            ax.scatter(MATS, ys, color=INK, s=14, zorder=3)
        ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig, use_container_width=True)
    with yc2:
        st.markdown("###### The inversion gauge (slope coefficient)")
        names = list(params.keys()); b1s = [params[n][1] for n in names]
        labels = [n.split(" (")[0] for n in names]
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        bars = ax.bar(labels, b1s, color=MIDGREY, edgecolor=INK, linewidth=0.5, width=0.6)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_ylabel("β₁  (positive = inverted)")
        ax.text(0, b1s[0]-1.4, "normal", ha="center", color=GREY, fontsize=9)
        ax.text(2, b1s[2]+0.5, "deep inversion", ha="center", color=INK, fontsize=9)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<p class="note">The slope coefficient moves from clearly normal in 2021, to an extreme '
                'inversion at the 2023 crisis peak, and back toward flat by 2026 — the crisis and recovery '
                'in a single number.</p>', unsafe_allow_html=True)

# ==========================================================================
# TAB 4 — CONCLUSION
# ==========================================================================
with tab4:
    st.subheader("What the three sections together show")
    st.write(
        "**The Evidence** proved that KSE-100 returns are violently non-normal — fat-tailed and crash-prone in a "
        "way standard models forbid. **The Model** turned that into a price: ignoring jumps underprices crash "
        "protection by up to several hundred per cent. **The Yield Curve** found the same crises in the government "
        "bond market, where the curve inverted to one of the most extreme degrees seen anywhere.")
    st.write(
        "The unifying finding: Pakistan's financial markets are shaped by sudden, discontinuous, crisis-driven "
        "jumps that the standard toolkit of quantitative finance — built for the smooth markets of the developed "
        "world — systematically fails to capture. Measuring that failure, across both the equity and bond markets, "
        "is the contribution of this project.")
    st.markdown("###### Methods built from scratch")
    st.write(
        "Distribution statistics and the Jarque–Bera test · log returns and volatility · the Black–Scholes formula "
        "and its Greeks · Brownian motion and the Poisson process · the Merton Jump-Diffusion model and its "
        "calibration · Monte Carlo simulation · Nelson–Siegel curve fitting by least-squares optimisation.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="note">Data: KSE-100 daily index (2018–2026). Yield curves are representative term-structure '
        'snapshots from published State Bank of Pakistan auction and secondary-market reference yields. '
        'Built as an independent research project.</p>', unsafe_allow_html=True)
