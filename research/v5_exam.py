"""The FRESH exam of V5_PROTOCOL.md, and its harness check on old data.

    python3 v5_exam.py val      # harness check: C0 through the real bot pipeline on VAL (design data)
    python3 v5_exam.py freeze   # write results/v5_frozen.json (what goes to the exam) - before unblinding
    python3 v5_exam.py fresh    # THE EXAM. Runs once: needs v5_frozen.json, refuses if a result exists.

C0 = v4 with the shipped rules (bot/config.100usd.toml: $100, 1.5x) through bot/replay.py, on the
full universe with the protocol's hygiene rules, real funding, Binance minimum order sizes, and
settlement of positions in contracts that stop trading.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import v5_lib as L                      # noqa: E402
import vision_panel as vp               # noqa: E402
from bot.config import load_config      # noqa: E402
from bot.data import CsvData            # noqa: E402
from bot.replay import replay           # noqa: E402

FRESH_START = "2026-06-08"
FROZEN = os.path.join(ROOT, "research", "results", "v5_frozen.json")
RESULT = os.path.join(ROOT, "research", "results", "v5_fresh_result.json")


def vision_source(fresh: bool):
    ren = lambda d: d.rename(columns=lambda x: x + "USDT")
    o, h, l, c, v = (vp.load(f, fresh) for f in ("open", "high", "low", "close", "volume"))
    data = CsvData.from_frames(ren(o), ren(h), ren(l), ren(c), ren(v))
    crypto = vp.crypto_mask(c, L.WEEKEND_THR)
    funding = ren(vp.load("funding", fresh).reindex_like(c))

    def universe_fn(day):
        cols = crypto.columns[crypto.loc[day].to_numpy()]
        return [x + "USDT" for x in cols]
    return data, universe_fn, funding, c


def run_bot(start, end, fresh, equity=100.0, leverage=1.5, cfg_edit=None):
    cfg = load_config(os.path.join(ROOT, "bot", "config.100usd.toml"), env={})
    cfg.paper.initial_equity, cfg.risk.leverage = equity, leverage
    if cfg_edit:
        cfg_edit(cfg)
    data, ufn, funding, _ = vision_source(fresh)
    return replay(cfg, start, end, data=data, funding=funding, universe_fn=ufn, progress=False)


def describe(tag, res):
    b, eq = res["bot"], res["equity"]
    m = eq.resample("ME").last().pct_change()
    m.iloc[0] = eq.resample("ME").last().iloc[0] / eq.iloc[0] - 1
    print(f"{tag}: ${eq.iloc[0]:.0f} -> ${eq.iloc[-1]:.2f}  total {b['total']:+.1%}  Sharpe {b['sharpe']:.2f}  "
          f"maxDD {b['max_dd']:.1%}  | orders {res['orders']}, fees ${res['fees']:.2f}, funding paid ${res['funding_paid']:+.2f}, "
          f"settled {res['settled']}, avg positions {res['avg_positions']:.1f}, held/target {res['held_vs_target_gross']:.0%}")
    print("   by month: " + "  ".join(f"{t:%Y-%m} {x:+.1%}" for t, x in m.items()))
    return {"start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "final": float(eq.iloc[-1]), **b,
            "monthly": {f"{t:%Y-%m}": float(x) for t, x in m.items()}}


def benchmarks(start, end, fresh):
    p = L.load_panel(fresh)
    r, e = p["r"], p["e"]
    btc = r["BTC"].loc[start:end]
    basket = r.where(e.shift(1, fill_value=False)).mean(axis=1).loc[start:end]
    blend, n3, npump = L.v4(p)
    out = {}
    for name, x in (("BTC buy & hold", btc), ("equal-weight eligible coins", basket),
                    ("v4 idealized, 1x (vectorized)", blend.loc[start:end]),
                    ("  trend part alone", n3.loc[start:end]), ("  pump part alone", npump.loc[start:end])):
        s = L.stats(x)
        tot = float((1 + x.fillna(0)).prod() - 1)
        print(f"   {name:<32} total {tot:+7.1%}  Sharpe {s['sharpe']:5.2f}  maxDD {s['mdd']:6.1%}")
        out[name.strip()] = {"total": tot, **s}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "val":
        print("harness check on VAL (design data): C0 at $100, 1.5x, 2025-01-01..2026-06-07")
        describe("C0 real pipeline", run_bot("2025-01-01", "2026-06-07", fresh=False))
        benchmarks("2025-01-01", "2026-06-07", fresh=False)
    elif cmd == "freeze":
        frozen = json.loads(sys.argv[2])
        frozen["frozen_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        json.dump(frozen, open(FROZEN, "w"), indent=2)
        print(json.dumps(frozen, indent=2))
    elif cmd == "fresh":
        if not os.path.exists(FROZEN):
            sys.exit("refusing: candidates are not frozen (results/v5_frozen.json missing)")
        if os.path.exists(RESULT):
            sys.exit("refusing: the FRESH exam has already been run (results/v5_fresh_result.json exists)")
        frozen = json.load(open(FROZEN))
        end = str(vp.load("close", fresh=True)["BTC"].dropna().index[-1].date())
        print(f"FRESH exam {FRESH_START}..{end}; frozen candidates: {frozen['candidates']}")
        out = {"window": [FRESH_START, end], "frozen": frozen}
        out["C0 $100 1.5x"] = describe("C0 v4 as shipped, $100, 1.5x", run_bot(FRESH_START, end, fresh=True))
        out["C0 $100 1.0x"] = describe("C0 at 1.0x", run_bot(FRESH_START, end, fresh=True, leverage=1.0))
        out["C0 $1000 1.5x"] = describe("C0 with $1,000 (small-account effect)", run_bot(FRESH_START, end, fresh=True, equity=1000.0))
        f = vp.load("funding", fresh=True).loc[FRESH_START:end]
        last_f = f.index[f.notna().sum(axis=1) > 50][-1]
        print(f"real funding data available through {last_f.date()}; later days assume zero funding")
        out["funding_through"] = str(last_f.date())
        print("benchmarks:")
        out["benchmarks"] = benchmarks(FRESH_START, end, fresh=True)
        json.dump(out, open(RESULT, "w"), indent=2)
