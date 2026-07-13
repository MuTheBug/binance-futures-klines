"""Deepest honest research for a $100 account: the binding constraint is
Binance's ~$5 minimum order, so every variant is evaluated in a DOLLAR-level
simulator (order quantization, skipped dust orders) instead of the smooth
weight-space engine.

Search (IS-gated, fence 2024-12-31, plateau rule):
  N       top-N position count {5,6,8,10,12}
  renorm  recycle truncated gross into kept names {off,on}
  mix     sleeve blend under concentration {40/40/20, A-only, 50/50 AB, 60/20/20}
  cadence {3,5,7} days
  lev     {1.5, 2.0, 2.5}

Selection runs use fixed-$100 sizing (isolates the granularity drag);
the final projection compounds (the handicap fades as the account grows).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import engine
import lab
from apex import hold_cadence
from pf_push import conviction_floor, deadband, sleeves

close, vol, ret, elig = lab.load_data("1d")
R = ret.fillna(0.0).values
IDX = close.index
MIN_ORDER = 5.0
COST = 0.0010


def build_wv(mix=(0.40, 0.40, 0.20), cadence=3):
    A, B, C = sleeves()
    blend = mix[0] * A + mix[1] * B + mix[2] * C
    W = conviction_floor(blend, 0.3)
    W = deadband(W, 0.25)
    g = W.abs().sum(axis=1)
    W = W.mul((g > 0.5 * g[g > 0].median()).astype(float), axis=0)
    W = hold_cadence(W, cadence)
    return engine.vol_target(W, ret, "1d", target_vol=0.40, max_leverage=3.0)


def truncate(row, n, renorm):
    if (np.abs(row) > 1e-9).sum() <= n:
        return row
    keep_idx = np.argsort(-np.abs(row))[:n]
    out = np.zeros_like(row)
    out[keep_idx] = row[keep_idx]
    if renorm:
        g0, g1 = np.abs(row).sum(), np.abs(out).sum()
        if g1 > 0:
            out *= g0 / g1
    return out


def dollar_sim(wv, lev=2.0, n_top=8, renorm=False, equity0=100.0,
               fixed_equity=True, cost_side=COST, min_order=MIN_ORDER):
    """Returns daily return series of the quantized account."""
    Wv = wv.fillna(0.0).values
    n_bars, n_sym = Wv.shape
    pos = np.zeros(n_sym)          # dollar positions
    eq = equity0
    out = np.zeros(n_bars)
    for t in range(n_bars):
        pnl = float(np.dot(pos, R[t]))
        base = equity0 if fixed_equity else eq
        out[t] = pnl / max(base, 1e-9)
        eq += pnl
        pos = pos * (1.0 + R[t])
        if eq <= 1.0:              # busted (only possible when compounding)
            out[t + 1:] = 0.0
            break
        size_base = equity0 if fixed_equity else eq
        tgt = truncate(Wv[t], n_top, renorm) * lev * size_base
        trade = tgt - pos
        doit = np.abs(trade) >= min_order
        executed = np.where(doit, trade, 0.0)
        cost = np.abs(executed).sum() * cost_side
        eq -= cost
        out[t] -= cost / max(size_base, 1e-9)
        pos = pos + executed
    return pd.Series(out, index=IDX)


def is_stats(r, label):
    i, _ = lab.split(r)
    m = engine.metrics(i, "1d")
    return {"variant": label, "IS_sharpe": m["Sharpe"], "IS_cagr": m["CAGR"],
            "IS_dd": m["maxDD"], "IS_pf": m["profit_factor"]}


def main():
    rows = []
    wv_base = build_wv()

    # ---- stage 1: N x renorm ------------------------------------------------
    print("stage 1: position count x renormalization ($100 fixed, 2x, 10bps)")
    best1, best1_s = None, -np.inf
    for renorm in (False, True):
        for n in (5, 6, 8, 10, 12):
            r = dollar_sim(wv_base, n_top=n, renorm=renorm)
            m = is_stats(r, f"N={n} renorm={'on' if renorm else 'off'}")
            rows.append(m)
            print(f"  {m['variant']:22} IS Sharpe {m['IS_sharpe']:.2f}  "
                  f"CAGR {m['IS_cagr'] * 100:.0f}%  DD {m['IS_dd'] * 100:.0f}%")
            if m["IS_sharpe"] > best1_s:
                best1_s, best1 = m["IS_sharpe"], (n, renorm)
    n_best, renorm_best = best1
    print(f"  -> stage-1 winner: N={n_best}, renorm={'on' if renorm_best else 'off'}")

    # ---- stage 2: sleeve mix under concentration -----------------------------
    print("\nstage 2: sleeve mix at the winning N")
    mixes = {"40/40/20 (base)": (0.40, 0.40, 0.20), "A only": (1.0, 0.0, 0.0),
             "50/50 A+B": (0.50, 0.50, 0.0), "60/20/20 A-heavy": (0.60, 0.20, 0.20)}
    best2, best2_s, wv_best = None, -np.inf, wv_base
    for name, mix in mixes.items():
        wv = wv_base if name == "40/40/20 (base)" else build_wv(mix=mix)
        r = dollar_sim(wv, n_top=n_best, renorm=renorm_best)
        m = is_stats(r, f"mix {name}")
        rows.append(m)
        print(f"  {m['variant']:26} IS Sharpe {m['IS_sharpe']:.2f}  "
              f"CAGR {m['IS_cagr'] * 100:.0f}%")
        if m["IS_sharpe"] > best2_s:
            best2_s, best2, wv_best = m["IS_sharpe"], name, wv

    # ---- stage 3: cadence ----------------------------------------------------
    print("\nstage 3: cadence")
    best3, best3_s = 3, -np.inf
    for cad in (3, 5, 7):
        wv = wv_best if cad == 3 else build_wv(mix=mixes[best2], cadence=cad)
        r = dollar_sim(wv, n_top=n_best, renorm=renorm_best)
        m = is_stats(r, f"cadence {cad}d")
        rows.append(m)
        print(f"  {m['variant']:22} IS Sharpe {m['IS_sharpe']:.2f}")
        if m["IS_sharpe"] > best3_s:
            best3_s, best3 = m["IS_sharpe"], cad
    wv_final = wv_best if best3 == 3 else build_wv(mix=mixes[best2], cadence=best3)

    # ---- stage 4: leverage ---------------------------------------------------
    print("\nstage 4: leverage on the winning config")
    for lev in (1.5, 2.0, 2.5):
        r = dollar_sim(wv_final, lev=lev, n_top=n_best, renorm=renorm_best)
        m = is_stats(r, f"lev {lev}x")
        rows.append(m)
        print(f"  {m['variant']:22} IS Sharpe {m['IS_sharpe']:.2f}  "
              f"CAGR {m['IS_cagr'] * 100:.0f}%  DD {m['IS_dd'] * 100:.0f}%")

    pd.DataFrame(rows).to_csv("../seismo/out/micro100_results.csv", index=False)

    # ---- final: single OOS read + full, at 2x --------------------------------
    r_fin = dollar_sim(wv_final, lev=2.0, n_top=n_best, renorm=renorm_best)
    i, o = lab.split(r_fin)
    print("\nFINAL $100 config:", f"N={n_best} renorm={renorm_best} "
          f"mix={best2} cadence={best3} lev=2x")
    for tag, rr in [("IS", i), ("OOS", o), ("FULL", r_fin)]:
        m = engine.metrics(rr, "1d")
        print(f"  {tag:4} Sharpe {m['Sharpe']:.2f}  CAGR {m['CAGR'] * 100:.0f}%  "
              f"DD {m['maxDD'] * 100:.0f}%  PF {m['profit_factor']:.2f}")
    r_fin.to_csv("results/micro100_final_net.csv")

    # baseline comparison (old proxy config: N=8, no renorm, 40/40/20)
    r_old = dollar_sim(wv_base, lev=2.0, n_top=8, renorm=False)
    m = engine.metrics(r_old, "1d")
    print(f"  old $100 spec: FULL Sharpe {m['Sharpe']:.2f}  "
          f"CAGR {m['CAGR'] * 100:.0f}%")


if __name__ == "__main__":
    main()
