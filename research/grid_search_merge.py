"""Merge sharded grid_search results; print the DEV-IS ranking and the held-out tests of the top 3
(the only configurations whose HOLDOUT / OOS numbers are looked at)."""
import glob
import numpy as np
import pandas as pd

rows = [r for f in sorted(glob.glob("results/grid_search_*of*.pkl")) for r in pd.read_pickle(f)]
rows.sort(key=lambda r: -np.nan_to_num(r["dev"]["sharpe"], nan=-9))
print(f"{len(rows)} configurations, ranked by DEV-IS Sharpe (selection uses this column only)")
for r in rows:
    d = r["dev"]
    print(f"{r['name']:<50} DEV-IS Sharpe {d['sharpe']:5.2f}  CAGR {d['cagr']:7.1%}  maxDD {d['mdd']:6.1%}  "
          f"fills/d {r['fills_per_day']:.1f}")
print("\n=== TOP 3 by DEV-IS Sharpe: held-out tests ===")
for r in rows[:3]:
    d, h, o = r["dev"], r["hold"], r["oos"]
    print(f"{r['name']:<50} DEV-IS {d['sharpe']:5.2f} {d['cagr']:7.1%} dd {d['mdd']:6.1%} | HOLDOUT-IS {h['sharpe']:5.2f} "
          f"{h['cagr']:7.1%} dd {h['mdd']:6.1%} | OOS {o['sharpe']:5.2f} {o['cagr']:7.1%} dd {o['mdd']:6.1%}")
