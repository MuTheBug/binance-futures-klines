"""What would a $100 account have experienced running one grid on one coin?

For the chosen design, every eligible coin is run as its own $100 account (compounding, exposure
1x / 2x / 3x, a grid that loses its whole account is liquidated), started on 2023-01-01 (IS window)
and on 2025-01-01 (OOS window). Prints the distribution of ending balances and drawdowns.
Minimum order size: at <= 10 levels a side a $100 single-coin grid stays above Binance's ~5 USDT
minimum at exposure >= 0.6x, so returns scale directly (the bot shrinks levels if not).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from grid_study import load
from grid_final import design_inputs, params

WINDOWS = [("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-06-07")]


def account_runs(f, elig, design, exposure, start, end):
    from gridsim import simulate
    sname, sv, n, ename, bname = design
    g_frame, allow, bframe = design_inputs(f, sname, sv, bname)
    sl = lambda d: None if d is None else d.loc[start:end]
    cols = [c for c in f["close"].columns if elig[c].loc[start:end].iloc[:24 * 7].all()]   # tradable at start
    p = params(*design, exposure=exposure)
    p.compound = True
    allow_w = elig.loc[start:end, cols] & (allow.loc[start:end, cols].fillna(False) if allow is not None else True)
    res = simulate(*(f[k].loc[start:end, cols] for k in ("open", "high", "low", "close")), p,
                   g_frame=sl(g_frame[cols]) if g_frame is not None else None, allow=allow_w,
                   bias_frame=sl(bframe[cols]) if bframe is not None else None)
    eq = res["equity"] * 100.0
    dd = (eq / eq.cummax() - 1).min()
    return eq.iloc[-1], dd, res["liquidations"]


if __name__ == "__main__":
    design = eval(sys.argv[1])
    f, elig = load(start="2022-06-01")
    bh_rows = {}
    for start, end in WINDOWS:
        c = f["close"].loc[start:end]
        print(f"\n=== {design}  window {start}..{end} ===")
        for x in (1.0, 2.0, 3.0):
            final, dd, liq = account_runs(f, elig, design, x, start, end)
            q = final.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            print(f"  exposure {x:.0f}x: {len(final)} coins | $100 -> median ${q[0.5]:.0f}  (p10 ${q[0.1]:.0f}, "
                  f"p25 ${q[0.25]:.0f}, p75 ${q[0.75]:.0f}, p90 ${q[0.9]:.0f}) | ended above $100: {(final > 100).mean():.0%}"
                  f" | median maxDD {dd.median():.0%}, worst {dd.min():.0%} | accounts liquidated: {(liq > 0).mean():.0%}")
        bh = (c.ffill().iloc[-1] / c.bfill().iloc[0] * 100).reindex(final.index)
        print(f"  buy & hold same coins: median ${bh.median():.0f}, ended above $100: {(bh > 100).mean():.0%}")
