# When Models Break — Quantifying Discontinuous Risk in Pakistan's Financial Markets

An interactive research tool examining how standard financial models fail in
Pakistan's crisis-prone markets, across three dimensions:

1. **The Evidence** — statistical proof that KSE-100 returns are non-normal (fat tails, crash days).
2. **The Model** — Black-Scholes vs Merton Jump-Diffusion option pricing, calibrated to KSE data.
3. **The Yield Curve** — Nelson-Siegel fits showing Pakistan's bond-market inversion through the crisis.

Built with Python and Streamlit as an independent project.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
`kse100_clean.csv` — KSE-100 daily index, 2018–2026.
