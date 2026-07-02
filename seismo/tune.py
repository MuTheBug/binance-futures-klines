"""Coarse in-sample grid search. Nothing after 2024-07-01 is touched here.

Design constants that are NOT tuned: 4h trigger horizon, 30d z/beta
lookbacks, $1M liquidity floor, breadth 0.85, 10bps/side costs, 3x max
leverage cap. Tuned (coarsely): z threshold, aftershock gap/window,
n picks, hold hours, risk budget.
"""
import itertools

import pandas as pd

from seismo.engine import Params, compute_signals, metrics, run
from seismo.panel import build_panel

IS_END = "2024-07-01"

panel = build_panel()
sig = compute_signals(panel)

grid = {
    "z_thr": [2.0, 2.5, 3.0],
    "aftershock_gap": [24.0, 48.0],
    "aftershock_win": [72.0, 96.0, 120.0],
    "n_picks": [3, 5, 8],
    "hold_hours": [12, 24, 36],
    "target_risk": [0.06, 0.10],
}

rows = []
keys = list(grid)
for vals in itertools.product(*grid.values()):
    kw = dict(zip(keys, vals))
    p = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0, **kw)
    port, tr, ev = run(sig, p, end=IS_END)
    m = metrics(port.loc[:pd.Timestamp(IS_END, tz="UTC")], tr, ev, "")
    m.update(kw)
    rows.append(m)

df = pd.DataFrame(rows).drop(columns=["label"])
df.to_csv("seismo/out/grid_is.csv", index=False)

ok = df[df["events"] >= 40].copy()
ok["score"] = ok["sharpe"]
cols = keys + ["events", "total_return", "cagr", "max_dd", "sharpe",
               "event_winrate", "event_pf", "avg_event", "worst_event"]
print("top 15 by IS Sharpe (>=40 events):")
print(ok.sort_values("score", ascending=False)[cols].head(15).round(3).to_string(index=False))
print("\nbottom 5:")
print(ok.sort_values("score")[cols].head(5).round(3).to_string(index=False))
print(f"\ncombos: {len(df)}, with>=40ev: {len(ok)}, "
      f"share with positive IS return: {(ok['total_return'] > 0).mean():.0%}, "
      f"share with PF>1: {(ok['event_pf'] > 1).mean():.0%}")
