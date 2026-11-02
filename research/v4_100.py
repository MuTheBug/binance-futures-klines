"""v4 for a $100 account: concentrate the book so every order clears Binance's minimum.

Problem: v4 holds ~10 trend positions plus pump longs and a BTC hedge. At $100 x 1.5 leverage many
of those orders are below the ~5 USDT minimum (BTC: assumed 100 USDT), so the bot silently skips
them and the account stops tracking the strategy.

Pre-registered protocol (written before running):
  * Every variant is run through the REAL bot pipeline (bot/replay.py: rolling-window strategy ->
    order planning with min-notional / rebalance band / caps -> paper fills with fees, slippage,
    generic funding), starting with 100 USDT at 1.5x leverage.
  * Variants: k_hold in {3, 4, 5} x pump sleeve in {btc (as v4; the hedge leg is skipped whenever
    it is below BTC's minimum), none (unhedged), off (v3 trend book only)}.
  * Selection: highest Sharpe on 2020-10-01..2024-12-31 among variants with max drawdown >= -40%.
    Only the selected variant is then run on 2025-01-01..2026-05-28 (OOS; note this window was
    already used to evaluate v3/v4, so it is not a clean holdout).
  * References (not selectable): v4 as-is (k_hold 10) at $100 and at $10,000.
"""
from __future__ import annotations
import logging
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bot.config import Config          # noqa: E402
from bot.data import CsvData           # noqa: E402
from bot.replay import replay          # noqa: E402

IS = ("2020-10-01", "2024-12-31")
OOS = ("2025-01-01", "2026-05-28")
VARIANTS = [(k, p) for k in (3, 4, 5) for p in ("btc", "none", "off")]
REFS = [("ref v4 k10 $100", 10, "btc", 100.0), ("ref v4 k10 $10k", 10, "btc", 10_000.0)]


def make_cfg(k_hold, pump, equity, leverage=1.5):
    cfg = Config()
    cfg.paper.initial_equity = equity
    cfg.risk.leverage = leverage
    cfg.strategy.k_hold = k_hold
    if pump == "off":
        cfg.strategy.w_trend, cfg.strategy.w_pump = 1.0, 0.0
    else:
        cfg.strategy.pump_hedge = pump
    return cfg


def line(name, res):
    b, i = res["bot"], res["ideal"]
    return (f"{name:<26} Sharpe {b['sharpe']:5.2f}  CAGR {b['cagr']:7.1%}  maxDD {b['max_dd']:6.1%}  "
            f"total {b['total']:+8.1%} | ideal Sharpe {i['sharpe']:5.2f} | corr {res['corr']:.2f} | "
            f"avg positions {res['avg_positions']:.1f} | held/target gross {res['held_vs_target_gross']:.0%}")


def run_is(shard, nshard):
    data = CsvData(ROOT, Config().strategy.exclude)
    jobs = [(f"k{k} pump={p}", k, p, 100.0) for k, p in VARIANTS] + REFS
    rows = {}
    for j, (name, k, p, eq) in enumerate(jobs):
        if j % nshard != shard:
            continue
        res = replay(make_cfg(k, p, eq), *IS, data=data, progress=False)
        rows[name] = res
        print(line(name, res), flush=True)
    pd.to_pickle({k: {kk: v for kk, v in r.items() if kk != "final_positions"} for k, r in rows.items()},
                 os.path.join(ROOT, "research", "results", f"v4_100_is_{shard}of{nshard}.pkl"))


def select_and_oos():
    import glob
    rows = {}
    for f in glob.glob(os.path.join(ROOT, "research", "results", "v4_100_is_*of*.pkl")):
        rows.update(pd.read_pickle(f))
    print(f"IS {IS[0]}..{IS[1]}, $100 at 1.5x unless noted:")
    for name in sorted(rows, key=lambda n: -rows[n]["bot"]["sharpe"]):
        print("  " + line(name, rows[name]))
    ok = {n: r for n, r in rows.items() if not n.startswith("ref") and r["bot"]["max_dd"] >= -0.40}
    best = max(ok, key=lambda n: ok[n]["bot"]["sharpe"])
    print(f"\nSELECTED (pre-registered rule): {best}")
    k = int(best.split()[0][1:])
    p = best.split("pump=")[1]
    data = CsvData(ROOT, Config().strategy.exclude)
    print(f"\nOOS {OOS[0]}..{OOS[1]}:")
    for lev in (1.0, 1.5, 2.0):
        res = replay(make_cfg(k, p, 100.0, lev), *OOS, data=data, progress=False)
        print("  " + line(f"{best} @{lev}x", res))
    res = replay(make_cfg(10, "btc", 100.0), *OOS, data=data, progress=False)
    print("  " + line("ref v4 k10 $100 @1.5x", res))
    for lev in (1.0, 2.0):
        res = replay(make_cfg(k, p, 100.0, lev), *IS, data=data, progress=False)
        print("  IS " + line(f"{best} @{lev}x", res))


# ------------------------------------------------------------------------------------------------
# Added after the first IS results (before selection): the IS replay compounds, so the account is
# large after a year and minimum order sizes stop binding. The $100 reality check restarts a FRESH
# $100 account every 6 months and runs it for 6 months, so the small-account constraints bind the
# whole time. Rule fixed now: if the selected variant trails another by > 0.3 Sharpe here, flag it.
WINDOW_STARTS = ["2020-10-01", "2021-04-01", "2021-10-01", "2022-04-01", "2022-10-01", "2023-04-01",
                 "2023-10-01", "2024-04-01", "2025-01-01", "2025-07-01", "2026-01-01"]


def fresh_windows(name, k, p, lev=1.5):
    import numpy as np
    data = CsvData(ROOT, Config().strategy.exclude)
    out = []
    for st in WINDOW_STARTS:
        en = min(pd.Timestamp(st) + pd.DateOffset(months=6) - pd.Timedelta(days=1), pd.Timestamp(OOS[1]))
        res = replay(make_cfg(k, p, 100.0, lev), st, str(en.date()), data=data, progress=False)
        eq = res["equity"]
        out.append({"start": st, "end": str(en.date()), "final": float(eq.iloc[-1]),
                    "mdd": float((eq / eq.cummax() - 1).min()), "sharpe": res["bot"]["sharpe"],
                    "positions": res["avg_positions"], "held": res["held_vs_target_gross"]})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(ROOT, "research", "results", f"v4_100_fresh_{name.replace(' ', '_').replace('=', '')}.csv"),
              index=False)
    is_ = df[df.start < "2025-01-01"]
    oos = df[df.start >= "2025-01-01"]
    print(f"{name:<22} fresh $100 x 6 months | IS windows: median ${is_.final.median():.0f} "
          f"(range ${is_.final.min():.0f}..${is_.final.max():.0f}), mean Sharpe {is_.sharpe.mean():.2f}, "
          f"worst DD {is_.mdd.min():.0%} | OOS windows: " +
          ", ".join(f"{r.start[:7]} ${r.final:.0f}" for r in oos.itertuples()) +
          f" | avg positions {df.positions.mean():.1f}, held/target {df.held.mean():.0%}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    if sys.argv[1] == "is":
        run_is(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "fresh":
        jobs = [(f"k{k} pump={p}", k, p) for k, p in VARIANTS] + [("ref v4 k10", 10, "btc")]
        shard, nshard = int(sys.argv[2]), int(sys.argv[3])
        for j, (name, k, p) in enumerate(jobs):
            if j % nshard == shard:
                fresh_windows(name, k, p)
    else:
        select_and_oos()
