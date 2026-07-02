"""Backtest of the EXACT combined book the bot trades (bot/signals.py):

  ECHO v2 blend (my independent replica) and RIFT (research/rift.py, weekly
  anchor), each vol-targeted to 40%/yr on its own trailing net, combined
  50/50, optional bot-style top-N position truncation and leverage
  multiplier with a 3x gross cap. Costs 10 bps/side, fills at next open.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research")
import rift  # noqa: E402

from seismo.echo_replica import build_weights, simulate, stats  # noqa: E402

TARGET_VOL = 0.40
COST = 0.0010


def raw_net(panel, W, cost_side=COST):
    Of = panel["O"].ffill()
    RO = Of.pct_change().fillna(0.0)
    Wl = W.fillna(0.0)
    gross = (Wl.shift(2) * RO).sum(axis=1)
    cost = (Wl.diff().abs().sum(axis=1) * cost_side).shift(1).fillna(0.0)
    return gross - cost


def vt_weights(panel, W, cost_side=COST):
    base = raw_net(panel, W, cost_side)
    rv = base.rolling(30, min_periods=15).std() * np.sqrt(365)
    scale = (TARGET_VOL / rv).shift(1).clip(upper=3.0).fillna(0.0)
    return W.fillna(0.0).mul(scale, axis=0)


def truncate_top_n(W, n):
    keep = W.abs().rank(axis=1, ascending=False, method="first") <= n
    return W.where(keep, 0.0)


def cap_gross(W, cap):
    g = W.abs().sum(axis=1)
    over = (g / cap).clip(lower=1.0)
    return W.div(over, axis=0)


def window(r, s=None, e=None):
    if s:
        r = r.loc[pd.Timestamp(s, tz="UTC"):]
    if e:
        r = r.loc[:pd.Timestamp(e, tz="UTC")]
    return r


def main():
    panel, W_echo = build_weights()
    C, U = panel["C"], panel["U"]
    W_rift = rift.rift_weights(C, C.pct_change(), U)

    ve = vt_weights(panel, W_echo)
    vr = vt_weights(panel, W_rift)
    combined = 0.5 * ve + 0.5 * vr

    books = {
        "ECHO alone (vol-tgt 1x)": ve,
        "RIFT alone (vol-tgt 1x)": vr,
        "combined 50/50 (1x)": combined,
        "combined, bot top-14 truncation": truncate_top_n(combined, 14),
        "combined, top-30 truncation": truncate_top_n(combined, 30),
        "combined @2x lev, 3x gross cap (bot default)":
            cap_gross(2.0 * truncate_top_n(combined, 30), 3.0),
    }
    rows = []
    for label, Wb in books.items():
        net = raw_net(panel, Wb)
        for wname, s, e in [("full", None, None),
                            ("common OOS >=2025-01", "2025-01-01", None)]:
            rows.append(stats(window(net, s, e), f"{label} | {wname}"))
        if label == "combined @2x lev, 3x gross cap (bot default)":
            net.to_csv("seismo/out/combined_bot_net.csv")
    t = pd.DataFrame(rows)
    t.to_csv("seismo/out/combined_results.csv", index=False)
    print(t.round(3).to_string(index=False))

    e_net, r_net = raw_net(panel, ve), raw_net(panel, vr)
    print(f"\ncorr(ECHO, RIFT) daily: {e_net.corr(r_net):+.2f}")

    gross = books["combined @2x lev, 3x gross cap (bot default)"].abs().sum(axis=1)
    npos = (books["combined, top-30 truncation"].abs() > 1e-6).sum(axis=1)
    print(f"bot-book gross exposure mean/p95: {gross.mean():.2f}x / "
          f"{gross.quantile(0.95):.2f}x   positions mean: {npos.mean():.0f}")


if __name__ == "__main__":
    main()
