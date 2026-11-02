"""Phase B: pre-registered improvement search for grids.

Hypotheses (from theory + data, fixed before running):
  H1 exit rule   : cutting inventory when price leaves the range (reset / stop) beats holding it,
                   because breakouts continue at multi-day horizons (time-series momentum).
  H2 spacing     : spacing scaled to volatility, g = k * sigma_1h(7d), keeps every coin's grid at
                   the horizon where the variance ratios show reversal (k=1.5 ~ 2h, 3 ~ 9h, 6 ~ 36h).
  H3 regime      : neutral grids only when the daily trend is weak (|v3 strength| < 0.3).
  H4 trend bias  : long grid when the daily trend is up (> 0.3), short grid when down (< -0.3),
                   neutral otherwise; re-centred weekly so the bias follows the trend.
Selection uses DEV coins, data <= 2024 ONLY. HOLDOUT-coin and OOS (2025-26) results are printed
just for the top 3 configurations by DEV-IS Sharpe.
"""
from __future__ import annotations
import itertools
import sys
import numpy as np
import pandas as pd
from gridsim import GridParams
from grid_study import (load, split_coins, run, evaluate, hourly_vol, daily_trend_on_hours)

f, elig = load()
dev, hold = split_coins(f["close"].columns)
sig = hourly_vol(f)
trend = daily_trend_on_hours(f)
weak = trend.abs() < 0.3
bias_trend = pd.DataFrame(np.select([trend > 0.3, trend < -0.3], [1.0, -1.0], 0.0),
                          index=trend.index, columns=trend.columns).where(trend.notna())

spacings = [("k1.5", 1.5), ("k3", 3.0), ("k6", 6.0), ("g2%", 0.02), ("g4%", 0.04)]
ns = [5, 10]
exits = [("hold+monthly", dict(on_exit="hold", period=720)),
         ("reset", dict(on_exit="reset", exit_buffer=1)),
         ("stop24h", dict(on_exit="stop", exit_buffer=1, cooldown=24))]
biases = [("neutral", None, None), ("neutral|weak-trend", None, weak), ("long", None, None),
          ("trend-bias", bias_trend, None)]

SHARD, NSHARD = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
rows = []
for i, ((sname, sv), n, (ename, ekw), (bname, bframe, allow)) in enumerate(itertools.product(spacings, ns, exits, biases)):
    if i % NSHARD != SHARD:
        continue
    g_frame = sig * sv if sname.startswith("k") else None
    g_fixed = sv if g_frame is None else 0.01
    base_bias = "long" if bname == "long" else "neutral"
    kw = dict(ekw)
    if bname == "trend-bias" and kw.get("period", 0) == 0:
        kw["period"] = 168                                   # weekly re-centre to follow the trend
    gp = GridParams(g=g_fixed, n=n, bias=base_bias, compound=False, **kw)
    res, _ = run(f, elig, gp, g_frame=g_frame, allow=allow, bias_frame=bframe)
    out = evaluate(f"{sname:<5} n={n:<2} {ename:<12} {bname}", res, dev, hold, quiet=True)
    out["res_key"] = (sname, n, ename, bname)
    rows.append(out)
    d = out["dev"]
    print(f"{out['name']:<50} DEV-IS Sharpe {d['sharpe']:5.2f}  CAGR {d['cagr']:7.1%}  maxDD {d['mdd']:6.1%}  "
          f"fills/d {out['fills_per_day']:.1f}  liq/yr {out['liq_per_grid_year']:.2f}", flush=True)

pd.to_pickle(rows, f"results/grid_search_{SHARD}of{NSHARD}.pkl")
if NSHARD > 1:
    sys.exit(0)
top = sorted(rows, key=lambda r: -np.nan_to_num(r["dev"]["sharpe"], nan=-9))[:3]
print("\n=== TOP 3 by DEV-IS Sharpe: now (and only now) the held-out tests ===")
for r in top:
    d, h, o = r["dev"], r["hold"], r["oos"]
    print(f"{r['name']:<50} DEV-IS {d['sharpe']:5.2f} {d['cagr']:7.1%} dd {d['mdd']:6.1%} | HOLDOUT-IS {h['sharpe']:5.2f} "
          f"{h['cagr']:7.1%} dd {h['mdd']:6.1%} | OOS {o['sharpe']:5.2f} {o['cagr']:7.1%} dd {o['mdd']:6.1%}")
