"""Baseline run: default params, IS vs OOS, with benchmarks. No tuning yet."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import data, engine, strategies as st

INTERVAL = "1h"
IS_END = "2024-12-31"           # parameters may ONLY be chosen using data <= IS_END
START = "2022-05-25"           # majors populated from here in 1h

# universe / eligibility params (point-in-time, kept simple)
MIN_HISTORY = 504               # 21 days of hourly bars before a name is tradeable
MIN_DVOL = 3e5                  # trailing median hourly $ volume >= $300k
LIQ_LB = 720                    # 30-day liquidity window


def load():
    close = data.load("close", INTERVAL)
    vol = data.load("volume", INTERVAL)
    close = close[close.index >= START]
    vol = vol[vol.index >= START]
    # drop symbols that never have data in window
    keep = close.columns[close.notna().sum() > MIN_HISTORY]
    return close[keep], vol[keep]


def split(s: pd.Series):
    return s[s.index <= IS_END], s[s.index > IS_END]


def report(name, net):
    is_net, oos_net = split(net)
    mi = engine.metrics(is_net, INTERVAL, name + " [IS]")
    mo = engine.metrics(oos_net, INTERVAL, name + " [OOS]")
    print(engine.fmt_metrics(mi))
    print(engine.fmt_metrics(mo))
    msi, mso = engine.monthly_summary(is_net), engine.monthly_summary(oos_net)
    if msi and mso:
        print(f"{'':>24}  monthly mean IS={msi['mean_monthly']*100:>5.1f}% "
              f"(med {msi['median_monthly']*100:>5.1f}%, +{msi['pct_positive']*100:.0f}%) | "
              f"OOS={mso['mean_monthly']*100:>5.1f}% "
              f"(med {mso['median_monthly']*100:>5.1f}%, +{mso['pct_positive']*100:.0f}%)")
    print()
    return mi, mo


def main():
    close, vol = load()
    ret = engine.to_returns(close)
    elig = strategies_elig = st.eligibility(close, vol, MIN_HISTORY, MIN_DVOL, LIQ_LB)
    print(f"universe: {close.shape[1]} symbols, {close.shape[0]} bars "
          f"({close.index[0].date()}..{close.index[-1].date()})")
    print(f"eligible names: median={int(elig.sum(axis=1).median())} "
          f"min={int(elig.sum(axis=1).iloc[MIN_HISTORY:].min())} "
          f"max={int(elig.sum(axis=1).max())}\n")

    def run(w):
        return engine.simulate(w, ret, INTERVAL)["net"]

    # ---- benchmarks ----
    btc = ret["BTC"].copy()
    report("BTC buy&hold", btc)

    ew_w = elig.div(elig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    report("EqualWeight long", run(ew_w))

    # ---- strategies (default params) ----
    report("TS-mom L/S (14d)",
           run(st.ts_momentum(close, vol, lookback=336, vol_lookback=168, elig=elig)))
    report("TS-mom long-only",
           run(st.ts_momentum(close, vol, lookback=336, vol_lookback=168, elig=elig, long_only=True)))
    report("XS-mom L/S (14d)",
           run(st.xs_momentum(close, vol, lookback=336, skip=24, vol_lookback=168, elig=elig, quantile=0.3)))
    report("XS-mom long-only",
           run(st.xs_momentum(close, vol, lookback=336, skip=24, vol_lookback=168, elig=elig, quantile=0.3, long_only=True)))
    report("XS-reversal (1d)",
           run(st.xs_reversal(close, vol, lookback=24, vol_lookback=168, elig=elig, quantile=0.3)))
    report("Donchian L/S (14d)",
           run(st.donchian(close, vol, lookback=336, vol_lookback=168, elig=elig)))
    report("Donchian long-only",
           run(st.donchian(close, vol, lookback=336, vol_lookback=168, elig=elig, long_only=True)))


if __name__ == "__main__":
    main()
