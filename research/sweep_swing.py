"""
IS-only parameter sweeps for the swing strategy (tune ONLY <= 2024-12-31).

Driven by the event-study findings (see SWING_STRATEGY.md):
  * long breakout continuation in daily uptrends carries the alpha
    (fat right tail, edge grows past 7d);
  * compression pre-filter destroys the long edge -> off;
  * shorts have negative mean events; tested only as a small optional sleeve.

We sweep ONE dimension at a time around a base config and look for plateaus,
not the argmax.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import swing_signals
from swing_engine import Config, run, metrics, fmt, trade_stats
from run_swing import windows

BASE_SIG = dict(N=42, ts_thr=0.10, require_compression=False)
BASE_CFG = dict(risk_frac=0.0075, stop_atr=3.0, trail_atr=4.0, tmax_bars=90,
                max_positions=10, heat_cap=0.075, gross_cap=2.0,
                allow_short=False,
                entry_mode="retest", stop_mode="close")


def one(base, name, sig_kw=None, cfg_kw=None, sig_cache={}):
    skw = {**BASE_SIG, **(sig_kw or {})}
    key = tuple(sorted(skw.items()))
    if key not in sig_cache:
        sig_cache[key] = base.build(**skw)
    sig = sig_cache[key]
    cfg = Config(**{**BASE_CFG, **(cfg_kw or {})})
    i0, i_split, _ = windows(base)
    out = run(sig, cfg, start=i0, end=i_split)
    m = metrics(out["net"], name)
    ts = trade_stats(out["trades"])
    extra = (f" trades={ts.get('n_trades',0)} win={ts.get('win_rate',0)*100:.0f}% "
             f"avgR={ts.get('avg_R',0):+.2f} gross={out['gross'].mean():.2f}x") if ts else ""
    print(fmt(m) + extra)
    return m


if __name__ == "__main__":
    print("loading ...")
    base = swing_signals.Base()

    print("\n-- baseline (long-only breakout, no compression) --")
    one(base, "base")

    print("\n-- channel lookback N --")
    for n in (12, 20, 28, 42):
        one(base, f"N={n}", sig_kw=dict(N=n))

    print("\n-- regime threshold ts_thr --")
    for th in (0.0, 0.05, 0.10, 0.20, 0.30):
        one(base, f"thr={th}", sig_kw=dict(ts_thr=th))

    print("\n-- initial stop (ATR) --")
    for s in (2.0, 2.5, 3.0, 3.5, 4.0):
        one(base, f"stop={s}", cfg_kw=dict(stop_atr=s))

    print("\n-- trail (ATR) --")
    for tr in (3.0, 4.0, 5.0, 6.0):
        one(base, f"trail={tr}", cfg_kw=dict(trail_atr=tr))

    print("\n-- time stop (bars) --")
    for tm in (42, 60, 90, 126, 100000):
        one(base, f"tmax={tm}", cfg_kw=dict(tmax_bars=tm))

    print("\n-- max positions / heat (risk scaled so full book ~ same total risk) --")
    for mp in (6, 8, 10, 14, 20):
        one(base, f"maxpos={mp}", cfg_kw=dict(max_positions=mp,
                                              heat_cap=0.0075 * mp))
    for mp, rf in ((14, 0.0054), (20, 0.00375)):
        one(base, f"maxpos={mp} risk={rf}", cfg_kw=dict(max_positions=mp,
                                                        risk_frac=rf,
                                                        heat_cap=rf * mp))

    print("\n-- execution details --")
    for rb in (6, 12, 18, 30):
        one(base, f"retest_bars={rb}", cfg_kw=dict(retest_bars=rb))
    for da in (1.0, 2.0, 3.0):
        one(base, f"disaster_atr={da}", cfg_kw=dict(disaster_atr=da))

    print("\n-- compression / volume confirmation (should not help) --")
    one(base, "with compression q=.35", sig_kw=dict(require_compression=True, comp_q=0.35))
    one(base, "vol_conf>=1.5", sig_kw=dict(vol_conf=1.5))

    print("\n-- short sleeve (breakdown continuation, half risk) --")
    one(base, "long+short", cfg_kw=dict(allow_short=True))
    one(base, "short only", cfg_kw=dict(allow_short=True, allow_long=False))
