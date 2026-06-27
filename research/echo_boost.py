"""
Disciplined attempt to PUSH PnL further. Rule: only accept a change that improves the
IN-SAMPLE Sharpe (tuned on data <= 2024-12-31); OOS is a confirmation, never the target.
We try: (a) risk-parity vs fixed blend, (b) a candidate 3rd orthogonal sleeve,
(c) rebalance cadence, (d) drawdown-aware leverage.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lab, engine, echo, strategies as st

close, vol, ret, elig = lab.load_data("1d")
IS_END = lab.IS_END
bpy = 365.0


def vt(w):
    return engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0)


def net_of(w):
    return engine.simulate(vt(w), ret, "1d")["net"]


def shp(net):
    i, o = lab.split(net)
    return engine.metrics(i, "1d")["Sharpe"], engine.metrics(o, "1d")["Sharpe"]


def isshp(net):                      # in-sample Sharpe of a raw (gross~1) net series
    i, _ = lab.split(net)
    return engine.metrics(i, "1d")["Sharpe"]


# ---- candidate sleeves (each gross~1, market-neutral-ish, daily) ----
sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0)
A_trend = st.concentrate(st.ema_ensemble(st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
B_resid = echo.residual_continuation(close, ret, elig, beta_window=60, formations=(3, 5, 8), cap=0.10)
C_xsmom = st.ema_ensemble(st.xs_momentum(close, vol, lookback=60, skip=2, vol_lookback=30, elig=elig, quantile=0.3), (5, 10, 15))
D_donch = st.concentrate(st.ema_ensemble(st.donchian(close, vol, lookback=40, vol_lookback=30, elig=elig), (5, 10, 15)), 10)

sleeves = {"A trend": A_trend, "B residual": B_resid, "C xs-mom": C_xsmom, "D donchian": D_donch}
nets = {k: net_of(v) for k, v in sleeves.items()}

print("=" * 84)
print("CANDIDATE SLEEVES — standalone Sharpe (IS | OOS)")
for k, nn in nets.items():
    i, o = shp(nn); print(f"  {k:12} IS {i:5.2f} | OOS {o:5.2f}")
print("\ncorrelation of sleeve net returns (IS):")
isnets = pd.DataFrame({k: lab.split(v)[0] for k, v in nets.items()}).dropna()
print(isnets.corr().round(2).to_string())

# ---- (a) risk-parity (inverse-vol) blend of A+B vs fixed 50/50 ----
print("\n" + "=" * 84)
print("(a) BLEND CONSTRUCTION for A trend + B residual")
def blend_fixed(a):
    return a * A_trend + (1 - a) * B_resid
def blend_riskparity():
    # weight each sleeve by inverse of its trailing realized vol (lagged) -> equal risk
    va = lab.split(nets["A trend"])[0].std(); vb = lab.split(nets["B residual"])[0].std()
    wa = (1/va) / (1/va + 1/vb)
    return wa * A_trend + (1 - wa) * B_resid, wa
for a in (0.5, 0.6):
    i, o = shp(net_of(blend_fixed(a))); print(f"  fixed {int(a*100)}/{int((1-a)*100)}        IS {i:5.2f} | OOS {o:5.2f}")
wrp, wa = blend_riskparity()
i, o = shp(net_of(wrp)); print(f"  risk-parity (A={wa:.2f})  IS {i:5.2f} | OOS {o:5.2f}")

# ---- (b) add a 3rd sleeve only if IS improves ----
print("\n" + "=" * 84)
print("(b) ADD A 3rd SLEEVE (equal-risk 3-way) — accept only if IS Sharpe rises")
base = net_of(blend_fixed(0.5)); base_is = isshp(base)
print(f"  baseline A+B 50/50: IS {base_is:.2f}")
for name, W in [("+ C xs-mom", C_xsmom), ("+ D donchian", D_donch)]:
    three = (A_trend + B_resid + W) / 3.0
    i, o = shp(net_of(three))
    verdict = "ACCEPT" if i > base_is + 0.02 else "reject (no IS gain)"
    print(f"  {name:14} (1/3 each)  IS {i:5.2f} | OOS {o:5.2f}   -> {verdict}")

# ---- (c) rebalance cadence on the chosen blend ----
print("\n" + "=" * 84)
print("(c) REBALANCE CADENCE on A+B 50/50 (turnover/cost effect)")
wv = vt(blend_fixed(0.5))
for k in (1, 2, 3):
    wk = wv.copy()
    if k > 1:
        keep = np.zeros(len(wk), bool); keep[::k] = True
        wk = wk.where(pd.Series(keep, index=wk.index), np.nan).ffill()
    nn = engine.simulate(wk, ret, "1d")["net"]
    i, o = shp(nn); tno = engine.simulate(wk, ret, "1d")["turnover"].mean()
    print(f"  every {k}d: IS {i:5.2f} | OOS {o:5.2f} | turnover/day {tno:.3f}")

# ---- (d) drawdown-aware leverage (maximize median terminal wealth s.t. tail-DD cap) ----
print("\n" + "=" * 84)
print("(d) LEVERAGE: median 1yr wealth vs worst-path drawdown (block bootstrap)")
net = blend_fixed(0.5); net = net_of(net)
r = net.dropna().to_numpy(); n = len(r); rng = np.random.default_rng(9)
H, block, P = 365, 20, 40000
nb = int(np.ceil(H/block)); s0 = rng.integers(0, n-block, size=(P, nb))
idx = (s0[:, :, None] + np.arange(block)[None, None, :]).reshape(P, -1)[:, :H]; paths = r[idx]
for K in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
    lev = paths*K; thr = -(1-0.005)/K; ruined = (lev <= thr).any(axis=1)
    gk = np.where(lev <= thr, -1.0, lev); eq = np.cumprod(1+gk, axis=1)
    fin = eq[:, -1].copy(); fin[ruined] = 0.0
    dd = (eq/np.maximum.accumulate(eq, axis=1)-1).min(axis=1)
    print(f"  {K:>4g}x: median {np.median(fin):4.2f}x | p5 {np.quantile(fin,.05):4.2f}x | "
          f"median maxDD {np.median(dd)*100:4.0f}% | p1 worst maxDD {np.quantile(dd,.01)*100:4.0f}% | "
          f"ruin {ruined.mean()*100:.1f}%")
print("\nDONE.")
