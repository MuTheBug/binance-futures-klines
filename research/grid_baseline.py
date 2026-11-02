"""Phase A: classic grids as users deploy them (re-centred monthly, static inside the month)."""
import numpy as np
import pandas as pd
from gridsim import GridParams
from grid_study import load, split_coins, run, evaluate, buy_and_hold, report

f, elig = load()
dev, hold = split_coins(f["close"].columns)
report("benchmark: equal-weight buy & hold", buy_and_hold(f, elig))
rows = []
for bias in ("neutral", "long"):
    for R in (0.10, 0.25):
        for g in (0.005, 0.01, 0.02, 0.04):
            n = max(2, int(round(R / g)))
            gp = GridParams(g=g, n=n, bias=bias, on_exit="hold", period=24 * 30, compound=False)
            res, _ = run(f, elig, gp)
            rows.append(evaluate(f"{bias:<7} range±{R:.0%} g={g:.1%} n={n}", res, dev, hold))
pd.to_pickle(rows, "results/grid_baseline.pkl")
