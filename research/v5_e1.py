"""Addendum E1 of V5_PROTOCOL.md: extreme-funding harvest (pre-registered 2026-11-11).

    python3 v5_e1.py design    # IS / VAL verdict on design data, plus diagnostics
    python3 v5_e1.py fresh     # only if 'design' passed; runs once on 2026-06-08..2026-10-31

Each day, hold the side that RECEIVES funding in every coin whose funding over the day just ended
was extreme (|rate| >= 0.30%/day), inverse-vol sized, 25% name cap, BTC-hedged, 40% vol target.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

import v5_lib as L
import vision_panel as vp

THR, CAP = 0.003, 0.25
DESIGN_JSON = "results/v5_e1_design.json"
FRESH_JSON = "results/v5_e1_fresh.json"
FRESH = (pd.Timestamp("2026-06-08", tz="UTC"), pd.Timestamp("2026-10-31", tz="UTC"))


def panel(fresh=False):
    p = L.load_panel(fresh)
    p["e"] = vp.eligibility(p["c"], p["qv"], p["crypto"], 30, 1e6, 30)        # E1 universe
    liq = p["qv"].rolling(30, min_periods=15).median()
    p["cost"] = pd.DataFrame(np.where(liq < 5e6, 0.0020, L.COST), index=liq.index, columns=liq.columns)
    return p


def weights(p, thr=THR, cap=CAP, side=0, hedge=True, delay=0):
    """side: 0 both, +1 only longs (negative funding), -1 only shorts (positive funding)."""
    f = p["F"].where(p["e"])
    direction = -np.sign(f)
    trig = f.abs() >= thr
    if side:
        trig &= direction == side
    vol = p["r"].rolling(20, min_periods=10).std()
    raw = (direction / vol).where(trig, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).clip(-cap, cap)
    if hedge:
        w["BTC"] = w["BTC"] - w.sum(axis=1)
    return w.shift(delay).fillna(0.0) if delay else w, trig


def run(p, cost=None, **kw):
    w, trig = weights(p, **kw)
    net, Wt = L.run(w, p["r"], p["cost"] if cost is None else cost, p["F"])
    return net, Wt, trig


def line(name, net):
    a, b = (L.stats(x) for x in L.split(net))
    return (f"{name:<36} IS Sharpe {a['sharpe']:5.2f} CAGR {a['cagr']:+7.1%} DD {a['mdd']:6.1%} | "
            f"VAL Sharpe {b['sharpe']:5.2f} CAGR {b['cagr']:+7.1%} DD {b['mdd']:6.1%}"), a, b


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "design":
        p = panel()
        net, Wt, trig = run(p)
        txt, a, b = line("E1 as registered", net)
        passed = bool(a["sharpe"] >= 1.0 and b["sharpe"] >= 0.7)
        print(txt + f" | {'PASS' if passed else 'fail'}")
        json.dump({"is_sharpe": a["sharpe"], "val_sharpe": b["sharpe"], "passed": passed}, open(DESIGN_JSON, "w"))
        # what the trade looked like
        t = trig.loc[L.START:L.DESIGN_END]
        f = p["F"].where(trig)
        print(f"triggered coin-days: {int(t.sum().sum())} ({int((f < 0).loc[L.START:].sum().sum())} long / "
              f"{int((f > 0).loc[L.START:].sum().sum())} short); days with a position {float((t.sum(1) > 0).mean()):.0%}; "
              f"avg names when active {t.sum(1)[t.sum(1) > 0].mean():.1f}; distinct coins {int(t.any().sum())}")
        print("   by year:", t.sum(1).groupby(t.index.year).sum().astype(int).to_dict())
        Wp, R = Wt.shift(1).fillna(0.0), p["r"].fillna(0.0)
        for tag, sl in (("IS", slice(L.START, L.IS_END)), ("VAL", slice(L.IS_END + pd.Timedelta(days=1), L.DESIGN_END))):
            yrs = len(R.loc[sl]) / 365
            price = (Wp * R).loc[sl].sum().sum() / yrs
            fund = -(Wp * p["F"].reindex_like(R).fillna(0.0)).loc[sl].sum().sum() / yrs
            turn = (Wt - Wt.shift(1)).abs()
            cost = (turn * p["cost"]).loc[sl].sum().sum() / yrs
            print(f"   {tag}: per year, price P&L {price:+.2f}, funding received {fund:+.2f}, costs {-cost:+.2f}; "
                  f"avg gross exposure {Wp.loc[sl].abs().sum(1).mean():.2f}")
        v4net = L.v4(L.load_panel())[0]
        print(f"   correlation with v4 (IS): {float(net.loc[L.START:L.IS_END].corr(v4net.loc[L.START:L.IS_END])):+.2f}")
        print("diagnostics (cannot change the verdict):")
        for name, kw in (("threshold 0.15%/day", dict(thr=0.0015)), ("threshold 0.60%/day", dict(thr=0.006)),
                         ("longs only (negative funding)", dict(side=1)), ("shorts only (positive funding)", dict(side=-1)),
                         ("no BTC hedge", dict(hedge=False)), ("enter 1 day late", dict(delay=1))):
            print("   " + line(name, run(p, **kw)[0])[0])
        c50 = pd.DataFrame(0.0050, index=p["cost"].index, columns=p["cost"].columns)
        print("   " + line("costs 50 bps/side everywhere", run(p, cost=c50)[0])[0])
        print("   " + line("zero costs", run(p, cost=0.0)[0])[0])
    elif cmd == "fresh":
        d = json.load(open(DESIGN_JSON))
        if not d["passed"]:
            sys.exit("refusing: E1 did not pass on design data, so it is rejected (no fresh run)")
        if os.path.exists(FRESH_JSON):
            sys.exit("refusing: E1 fresh run already done")
        p = panel(fresh=True)
        net, Wt, trig = run(p)
        x = net.loc[FRESH[0]:FRESH[1]]
        s = L.stats(x)
        tot = float((1 + x).prod() - 1)
        print(f"E1 FRESH {FRESH[0].date()}..{FRESH[1].date()} (1x): total {tot:+.1%}, Sharpe {s['sharpe']:.2f}, maxDD {s['mdd']:.1%}; "
              f"triggered coin-days {int(trig.loc[FRESH[0]:FRESH[1]].sum().sum())}")
        m = (1 + x).resample("ME").prod() - 1
        print("   by month: " + "  ".join(f"{t:%Y-%m} {v:+.1%}" for t, v in m.items()))
        json.dump({"window": [str(FRESH[0].date()), str(FRESH[1].date())], "total": tot, **s,
                   "monthly": {f"{t:%Y-%m}": float(v) for t, v in m.items()}}, open(FRESH_JSON, "w"), indent=2)
