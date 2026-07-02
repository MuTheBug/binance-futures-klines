"""Head-to-head: RIFT (this branch) vs ECHO v2 (branch ...-bim2hm).

ECHO v2 daily net returns were produced by running THAT branch's own code
unchanged (research/echo_engine.py pipeline) and exporting the vol-targeted
(40%/yr, ~1.3x gross) blend net series at 7 and 10 bps/side; copies live in
seismo/out/echo_v2_net_*.csv.

Fairness rules:
- cost parity: both compared at 10 bps/side (ECHO's native 7 bps also shown)
- windows: full common period; COMMON OOS = after 2024-12-31 (ECHO's IS
  fence; also OOS for RIFT whose fence is 2024-07-01). RIFT's own OOS
  window (2024-07 ->) is shown too, but Jul-Dec 2024 was ECHO's IS.
- Sharpe/PF/win-rate are leverage-invariant; for CAGR/maxDD we also show
  RIFT scaled to ECHO's realized vol on the common window ("vol-matched").
"""
import numpy as np
import pandas as pd

from seismo.xsmom import build_daily, run_xs, signals

COMMON_OOS = "2025-01-01"
RIFT_OOS = "2024-07-01"
KW = dict(lookback=30, skip=0, n_side=10, rebal=7, use_resid=False)


def m(r, label):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.25
    dd = (eq / eq.cummax() - 1).min()
    g, l = r[r > 0].sum(), -r[r <= 0].sum()
    act = r[r != 0]
    return {"strategy": label, "ann_vol": r.std() * np.sqrt(365),
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1, "sharpe":
            r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan,
            "max_dd": dd, "daily_pf": g / l,
            "daily_wr": (act > 0).mean(), "total_x": eq.iloc[-1]}


def load_echo(tag):
    s = pd.read_csv(f"seismo/out/echo_v2_net_{tag}.csv", index_col=0)
    s.index = pd.to_datetime(s.index, utc=True)
    return s.iloc[:, 0]


def main():
    sig = signals(build_daily())
    rift, _ = run_xs(sig, w_cap=0.2, cost_side=0.0010, **KW)
    rift_g, _ = run_xs(sig, w_cap=0.2, cost_side=0.0010, gate_window=20, **KW)
    echo10, echo7 = load_echo("10bps"), load_echo("7bps")

    lo = max(rift.index[0], echo10.index[0])
    hi = min(rift.index[-1], echo10.index[-1])
    windows = [("full common", lo, hi),
               ("COMMON OOS (>=2025-01)", pd.Timestamp(COMMON_OOS, tz="UTC"), hi),
               ("RIFT OOS (>=2024-07)*", pd.Timestamp(RIFT_OOS, tz="UTC"), hi)]

    all_rows = []
    for wname, a, b in windows:
        rows = []
        e = echo10.loc[a:b]
        r = rift.loc[a:b]
        rg = rift_g.loc[a:b]
        scale = e.std() / r.std()
        rows.append(m(e, "ECHO v2 @10bps (native 1x vol-tgt)"))
        rows.append(m(echo7.loc[a:b], "ECHO v2 @7bps (their spec)"))
        rows.append(m(r, "RIFT @10bps (2x gross spec)"))
        rows.append(m(r * scale, "RIFT vol-matched to ECHO"))
        rows.append(m(rg, "RIFT gated @10bps"))
        both = 0.5 * (e / e.std() + r / r.std()) * e.std()
        rows.append(m(both, "50/50 blend (vol-weighted)"))
        t = pd.DataFrame(rows)
        t.insert(0, "window", wname)
        all_rows.append(t)
        corr = e.corr(r)
        print(f"\n===== {wname}  ({a.date()} -> {b.date()})  corr(ECHO,RIFT)={corr:+.2f} =====")
        print(t.drop(columns='window').round(3).to_string(index=False))

    pd.concat(all_rows).to_csv("seismo/out/compare_echo_rift.csv", index=False)
    print("\n* Jul-Dec 2024 was inside ECHO's tuning window (their fence: "
          "2024-12-31), so that row slightly flatters ECHO.")


if __name__ == "__main__":
    main()
