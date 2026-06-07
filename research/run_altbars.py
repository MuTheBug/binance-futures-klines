"""Test Renko / dollar bars / volume bars / volume profile vs the daily-trend flagship.
Same pipeline for all: signal -> inv-vol weights -> top-8 -> vol-target 40% -> engine."""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
import data, engine, strategies as st, altbars as ab
import lab

INTERVAL = "1d"
TV = 0.40
K = 8


def daily_trend_signal(close, lookbacks=(15, 30, 60, 90)):
    return sum(np.sign(close.pct_change(L)) for L in lookbacks) / len(lookbacks)


def main():
    c, v, r, e = lab.load_data(INTERVAL)
    didx = c.index

    def evalsig(name, signal, vol_lookback=15):
        signal = signal.reindex_like(c).fillna(0.0)
        w = st.signal_to_weights(signal, r, e, vol_lookback=vol_lookback)
        w = st.concentrate(w, K)
        w = engine.vol_target(w, r, INTERVAL, target_vol=TV)
        net = engine.simulate(w, r, INTERVAL)["net"]
        is_, oos_ = lab.split(net)
        mi, mo = engine.metrics(is_, INTERVAL), engine.metrics(oos_, INTERVAL)
        print(f"  {name:<26} IS: CAGR={mi['CAGR']*100:>6.1f}% Sh={mi['Sharpe']:>5.2f} "
              f"DD={mi['maxDD']*100:>5.0f}% | OOS: CAGR={mo['CAGR']*100:>6.1f}% "
              f"Sh={mo['Sharpe']:>5.2f} DD={mo['maxDD']*100:>5.0f}%")
        return net, (mi, mo)

    print("="*104)
    print("BASELINE")
    print("="*104)
    base_sig = daily_trend_signal(c)
    base_net, _ = evalsig("daily-trend (flagship)", base_sig)

    print("\n" + "="*104)
    print("ALTERNATIVE BARS (trend computed on the alt-bar clock, mapped to daily)")
    print("="*104)
    t0 = time.time()
    dsig = ab.bar_trend_signal("dollar", didx, lookbacks_bars=(4, 8, 16, 32), K=24.0)
    dnet, _ = evalsig("dollar-bar trend (K24)", dsig)
    vsig = ab.bar_trend_signal("volume", didx, lookbacks_bars=(4, 8, 16, 32), K=24.0)
    vnet, _ = evalsig("volume-bar trend (K24)", vsig)
    print(f"  [dollar/volume bars built in {time.time()-t0:.0f}s]")

    print("\n" + "="*104)
    print("RENKO (brick trend, mapped to daily)")
    print("="*104)
    t0 = time.time()
    for bp in [0.02, 0.03, 0.05]:
        rsig = ab.renko_trend_signal(didx, brick_pct=bp, lookbacks_bricks=(3, 6, 12, 24))
        evalsig(f"renko trend brick={bp:.0%}", rsig)
    print(f"  [renko built in {time.time()-t0:.0f}s]")

    print("\n" + "="*104)
    print("VOLUME PROFILE (POC reversion vs value-area breakout)")
    print("="*104)
    t0 = time.time()
    vp = ab.volume_profile(didx, window_h=720, bin_pct=0.01)
    print(f"  [volume profile built in {time.time()-t0:.0f}s]")
    poc = vp["poc_dist"]; va = vp["va_pos"]
    # POC reversion: fade distance from POC (clip to keep it a bounded signal)
    rev_sig = (-poc.clip(-0.3, 0.3) / 0.3)
    evalsig("VP POC-reversion", rev_sig)
    # Value-area breakout: long above VA, short below (momentum)
    evalsig("VP value-area breakout", va)
    # VP breakout aligned WITH trend (only take breakouts in the trend direction)
    va_a = va.reindex_like(c).fillna(0.0)
    aligned = va_a.where(np.sign(va_a) == np.sign(base_sig), 0.0)
    evalsig("VP breakout x trend", aligned)

    print("\n" + "="*104)
    print("ENSEMBLES (average of signals, then same pipeline)  -- does anything beat flagship OOS?")
    print("="*104)
    def avg(*sigs):
        s = sum(x.reindex_like(c).fillna(0.0) for x in sigs) / len(sigs)
        return s
    evalsig("trend + renko3%", avg(base_sig, ab.renko_trend_signal(didx, 0.03)))
    evalsig("trend + dollar-bar", avg(base_sig, dsig))
    evalsig("trend + VA-breakout", avg(base_sig, va))
    evalsig("trend + renko + dollar", avg(base_sig, ab.renko_trend_signal(didx, 0.03), dsig))


if __name__ == "__main__":
    main()
