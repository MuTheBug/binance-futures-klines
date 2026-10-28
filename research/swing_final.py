"""LOCKED swing book (design frozen on DEV coins <= 2024 before this file was run).

Four setups, each a discrete-trade system from swing.py, combined at equal risk:
  A  BREAKOUT    long 20d-high breakouts when coin & BTC trend up; 2ATR stop, 4ATR chandelier, 30d max
  B  IGNITION    coin/BTC jumps >25% in a day -> long coin / short BTC, hold 10d, 2ATR stop
  C  REL-STRENGTH coin/BTC 20d breakout (long) or breakdown (short), BTC-hedged, 3ATR trail, 30d max
  D  PULLBACK    coin in uptrend (strength>0.3) drops >1.5ATR in 3 days -> buy, 3ATR stop, 5d hold
Each sleeve is normalised by its own lagged expanding vol, averaged, then scaled to 40%/yr vol.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import engine, lab, target as tgt
from swing import load, split_coins, backtest, sh, P, I
import swing

SETUPS = {
    "A breakout": dict(don=20, trail_atr=4.0, sides=(1,), regime=True),
    "B ignition": dict(setup="ignite", hedge=True, ignite_thr=0.25, max_days=10, trail_atr=99.0, sides=(1,)),
    "C rel-strength": dict(hedge=True, don=20, trail_atr=3.0, max_days=30),
    "D pullback": dict(setup="pullback", trend_min=0.3, pull_atr=1.5, max_days=5, trail_atr=99.0,
                       stop_atr=3.0, sides=(1,)),
}
TV = 0.40


def norm(x):
    sd = x.expanding(min_periods=60).std().shift(1)
    return (x / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)


def book(o, h, l, c, v, cols, extra=None):
    sleeves = {k: backtest(o, h, l, c, v, {**P, **g, **(extra or {})}, cols) for k, g in SETUPS.items()}
    m = pd.DataFrame({k: norm(x) for k, x in sleeves.items()}).mean(axis=1)
    # warm-up guard: need 120 days of book history before sizing; cap the scale (bug fix, not a design change)
    scale = (TV / np.sqrt(365) / m.expanding(min_periods=120).std()).shift(1).clip(upper=0.1).fillna(0.0)
    return m * scale, sleeves


def line(tag, x):
    m = engine.metrics(x, I)
    print(f"  {tag:<40} Sharpe {m['Sharpe']:5.2f}  CAGR {m['CAGR']:7.1%}  maxDD {m['maxDD']:6.1%}  Calmar {m['Calmar']:5.2f}")


def main():
    o, h, l, c, v = load()
    dev, hold = split_coins(c.columns)
    allc = list(c.columns)
    isp, oosp = (lambda x: lab.split(x)[0]), (lambda x: lab.split(x)[1])

    bd, sd = book(o, h, l, c, v, dev)
    bh, shd = book(o, h, l, c, v, hold)
    ba, sa = book(o, h, l, c, v, allc)
    print("=== swing book, locked config ===")
    line("design   : DEV coins, <=2024", isp(bd))
    line("TEST A   : HOLDOUT coins, <=2024 (clean)", isp(bh))
    line("TEST B   : ALL coins, 2025-01..2026-06", oosp(ba))
    line("full     : ALL coins, 2020-06..2026-06", ba)
    print("\n  per-setup Sharpe  (HOLDOUT<=2024 | ALL 2025-26 | ALL full)")
    for k in SETUPS:
        print(f"    {k:<16} {sh(isp(shd[k])):5.2f} | {sh(oosp(sa[k])):5.2f} | {sh(sa[k]):5.2f}")
    print("\n  sleeve correlation (ALL coins, full):")
    print(pd.DataFrame({k: norm(x) for k, x in sa.items()}).corr().round(2).to_string())
    print("\n  yearly (ALL coins):", "  ".join(f"{y}:{g:+.0%}" for y, g in ((1 + ba).groupby(ba.index.year).prod() - 1).items()))
    print("  yearly Sharpe:     ", "  ".join(f"{y}:{sh(g):.1f}" for y, g in ba.groupby(ba.index.year)))
    ms = engine.monthly_summary(ba)
    print(f"  monthly: median {ms['median_monthly']:.1%}  positive {ms['pct_positive']:.0%}  worst {ms['worst']:.0%}")

    print("\n=== robustness (ALL coins: full / 2025-26) ===")
    for tag, ex in [("costs x2 (14bps/side + 20bps stop slip)", "cost2"), ("enter 1 day late", dict(delay=1)),
                    ("risk-per-trade 1% (vs 0.5%)", dict(risk=0.01))]:
        if ex == "cost2":
            swing.COST, swing.STOP_SLIP = 0.0014, 0.0020; x, _ = book(o, h, l, c, v, allc)
            swing.COST, swing.STOP_SLIP = 0.0007, 0.0010
        else:
            x, _ = book(o, h, l, c, v, allc, ex)
        print(f"  {tag:<42} {sh(x):5.2f} / {sh(oosp(x)):5.2f}")
    print("  drop top-20 days:                          ", end="")
    y = ba.copy(); y[y.nlargest(20).index] = 0; print(f"{sh(y):5.2f} / {sh(oosp(y)):5.2f}")

    a = ba[ba.index >= ba.ne(0).idxmax()].dropna().to_numpy()
    print("\n=== leverage on top of 40% vol (1yr block bootstrap, full sample) ===")
    for k in (0.5, 1.0, 1.5):
        rb = tgt.block_bootstrap(a, k, n_paths=3000)
        print(f"  {k:.1f}x  median {rb['median_monthly']:+.1%}/mo  P(lose half) {rb['p_lose_half']:.0%}  ruin {rb['p_ruin']:.0%}")
    (1 + ba).cumprod().rename("equity").to_csv("results/equity_swing.csv")


if __name__ == "__main__":
    main()
