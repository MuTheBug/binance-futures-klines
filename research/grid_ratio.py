"""H5 (added before seeing the main search results): grid on the coin/BTC ratio, BTC-hedged.

Idiosyncratic hourly returns mean-revert (VR(24h) 0.88-0.96, grid_meanrev.py), and hedging out BTC
removes the market-wide trends that wreck neutral grids. Execution needs two market orders per
level crossing (coin + BTC), so each fill is charged 0.2% (2 legs x (taker 0.05% + slip 0.05%)).
Close-only ratio path (o=h=l=c): conservative, no intrabar crossings. Funding nets to ~0 (long one
perp, short another) and is set to 0. Same protocol: select on DEV-IS, then HOLDOUT / OOS once.
"""
import itertools
import numpy as np
import pandas as pd
from gridsim import GridParams
from grid_study import load, split_coins, evaluate, IS_END
from gridsim import simulate

f, elig = load()
ratio = f["close"].div(f["close"]["BTC"], axis=0).drop(columns=["BTC"])
el = elig.drop(columns=["BTC"]) & elig["BTC"].to_numpy()[:, None]
dev, hold = split_coins(ratio.columns)
sig = np.log(ratio).diff().rolling(168, min_periods=84).std()
rows = []
for k, n, (ename, ekw) in itertools.product((1.5, 3.0, 6.0), (5, 10),
                                            [("hold+monthly", dict(on_exit="hold", period=720)),
                                             ("reset", dict(on_exit="reset", exit_buffer=1)),
                                             ("stop24h", dict(on_exit="stop", exit_buffer=1, cooldown=24))]):
    gp = GridParams(g=0.01, n=n, bias="neutral", maker_fee=0.002, taker_fee=0.001, slippage=0.001,
                    funding_annual=0.0, compound=False, **ekw)
    res = simulate(ratio, ratio, ratio, ratio, gp, g_frame=sig * k, allow=el)
    out = evaluate(f"ratio k={k} n={n} {ename}", res, dev, hold, quiet=True)
    rows.append(out)
    d = out["dev"]
    print(f"{out['name']:<36} DEV-IS Sharpe {d['sharpe']:5.2f}  CAGR {d['cagr']:7.1%}  maxDD {d['mdd']:6.1%}  "
          f"fills/d {out['fills_per_day']:.1f}", flush=True)
rows.sort(key=lambda r: -np.nan_to_num(r["dev"]["sharpe"], nan=-9))
print("\nTOP 2 by DEV-IS Sharpe, held-out tests:")
for r in rows[:2]:
    d, h, o = r["dev"], r["hold"], r["oos"]
    print(f"{r['name']:<36} DEV-IS {d['sharpe']:5.2f} {d['cagr']:7.1%} | HOLDOUT-IS {h['sharpe']:5.2f} {h['cagr']:7.1%} "
          f"| OOS {o['sharpe']:5.2f} {o['cagr']:7.1%} dd {o['mdd']:6.1%}")
pd.to_pickle(rows, "results/grid_ratio.pkl")
