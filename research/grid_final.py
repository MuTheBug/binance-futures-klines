"""Robustness report for chosen grid designs (run after grid_search.py).

For each design: DEV/HOLDOUT/OOS stats, per-year Sharpe, correlation with equal-weight buy & hold
(is it just market beta?), and stress tests: reversed intrabar path, limit fills needing 0.05%
trade-through (queue position), doubled fees, and funding at 0% / 30% per year.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from gridsim import GridParams
from grid_study import (IS_END, load, split_coins, run, evaluate, hourly_vol, daily_trend_on_hours,
                        buy_and_hold, stats)


def design_inputs(f, sname, sv, bname):
    sig = hourly_vol(f)
    trend = daily_trend_on_hours(f)
    g_frame = sig * sv if sname.startswith("k") else None
    allow = (trend.abs() < 0.3) if bname == "neutral|weak-trend" else None
    bframe = None
    if bname == "trend-bias":
        bframe = pd.DataFrame(np.select([trend > 0.3, trend < -0.3], [1.0, -1.0], 0.0),
                              index=trend.index, columns=trend.columns).where(trend.notna())
    return g_frame, allow, bframe


def params(sname, sv, n, ename, bname, **over):
    ekw = {"hold+monthly": dict(on_exit="hold", period=720), "reset": dict(on_exit="reset", exit_buffer=1),
           "stop24h": dict(on_exit="stop", exit_buffer=1, cooldown=24)}[ename]
    if bname == "trend-bias" and ekw.get("period", 0) == 0:
        ekw = {**ekw, "period": 168}
    g_fixed = sv if not sname.startswith("k") else 0.01
    return GridParams(g=g_fixed, n=n, bias="long" if bname == "long" else "neutral", compound=False, **ekw, **over)


def full_report(f, elig, dev, hold, bh, design):
    sname, sv, n, ename, bname = design
    g_frame, allow, bframe = design_inputs(f, sname, sv, bname)
    print(f"\n######## {sname} n={n} {ename} {bname}")
    base_res, base_pf = run(f, elig, params(*design), g_frame=g_frame, allow=allow, bias_frame=bframe)
    out = evaluate("base", base_res, dev, hold)
    d = (1 + base_pf).resample("D").prod() - 1
    b = (1 + bh).resample("D").prod() - 1
    print(f"  corr with equal-weight buy&hold (daily): IS {d[:IS_END].corr(b[:IS_END]):+.2f}  "
          f"OOS {d[IS_END:].corr(b[IS_END:]):+.2f}")
    yr = {y: round(stats(g)["sharpe"], 2) for y, g in base_pf.groupby(base_pf.index.year) if len(g) > 24 * 60}
    print(f"  Sharpe by year: {yr}")
    stresses = [("reversed intrabar path", dict(path="reverse")), ("0.05% trade-through", dict(through=0.0005)),
                ("2x fees", dict(maker_fee=0.0004, taker_fee=0.001)), ("funding 0%", dict(funding_annual=0.0)),
                ("funding 30%", dict(funding_annual=0.30))]
    for name, over in stresses:
        res, _ = run(f, elig, params(*design, **over), g_frame=g_frame, allow=allow, bias_frame=bframe)
        evaluate(f"  {name}", res, dev, hold)
    return out


if __name__ == "__main__":
    designs = [eval(a) for a in sys.argv[1:]]      # e.g. "('k3', 3.0, 5, 'hold+monthly', 'neutral|weak-trend')"
    f, elig = load()
    dev, hold = split_coins(f["close"].columns)
    bh = buy_and_hold(f, elig)
    for dsg in designs:
        full_report(f, elig, dev, hold, bh, dsg)
