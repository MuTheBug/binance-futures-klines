"""How far can leverage push v4? Growth vs ruin, assuming the backtest edge is real and assuming
it is only HALF real (live Sharpe is typically well below backtest). 1yr block bootstrap with
close-to-close liquidation (intraday wicks not modelled -> ruin numbers are a LOWER bound)."""
import numpy as np, pandas as pd
import engine, strategies as st, lab, target as tgt
from explore_v4 import load_all, build_v3, I
from robust_v4 import sleeve

c, v, r = load_all(); e = st.eligibility(c, v, 60, 5e6, 30)
v3 = engine.simulate(engine.vol_target(build_v3(c, v, e), r, I, target_vol=0.40), r, I)["net"]
net = (0.7 * v3 + 0.3 * sleeve(c, v, r, e, 0.25, 5)[0]["net"]).dropna()
net = net[net.index >= "2020-08-01"]
mu, sd = net.mean(), net.std()
half = net - mu / 2                                   # same volatility, half the edge
print(f"v4 backtest: Sharpe {mu/sd*np.sqrt(365):.2f}, vol {sd*np.sqrt(365):.0%}; "
      f"half-edge case Sharpe {half.mean()/sd*np.sqrt(365):.2f}")
print(f"Kelly leverage (on top of v4's ~40% vol): backtest {mu/sd**2:.1f}x, half-edge {half.mean()/sd**2:.1f}x\n")
print(f"{'lev':>4} | {'backtest edge: $1k ->1yr median':>32} {'P(-50%)':>8} {'ruin':>5} | {'half edge: median':>18} {'P(-50%)':>8} {'ruin':>5}")
for k in (1, 2, 3, 4, 6, 8, 10):
    a = tgt.block_bootstrap(net.to_numpy(), k, n_paths=3000)
    b = tgt.block_bootstrap(half.to_numpy(), k, n_paths=3000)
    print(f"{k:>3}x | {'$'+format(1000*a['median_year_x'],',.0f'):>32} {a['p_lose_half']:>8.0%} {a['p_ruin']:>5.0%} | "
          f"{'$'+format(1000*b['median_year_x'],',.0f'):>18} {b['p_lose_half']:>8.0%} {b['p_ruin']:>5.0%}")
