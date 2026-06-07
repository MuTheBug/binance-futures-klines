"""Reusable research harness: load, eligibility, IS/OOS split, evaluate."""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st

IS_END = pd.Timestamp("2024-12-31", tz="UTC")   # tune ONLY on data <= IS_END

DEFAULTS = {
    "1h": dict(start="2022-05-25", min_history=504, min_dvol=3e5, liq_lb=720),
    "1d": dict(start="2020-05-31", min_history=60,  min_dvol=5e6, liq_lb=30),
}


def load_data(interval):
    d = DEFAULTS[interval]
    close = data.load("close", interval); vol = data.load("volume", interval)
    close = close[close.index >= d["start"]]; vol = vol[vol.index >= d["start"]]
    keep = close.columns[close.notna().sum() > d["min_history"]]
    close, vol = close[keep], vol[keep]
    elig = st.eligibility(close, vol, d["min_history"], d["min_dvol"], d["liq_lb"])
    ret = engine.to_returns(close)
    return close, vol, ret, elig


def split(s):
    return s[s.index <= IS_END], s[s.index > IS_END]


def evaluate(name, weights, ret, interval, target_vol=0.40, verbose=True,
             max_leverage=3.0):
    """Vol-target the weights then simulate; print IS/OOS; return (net, sim)."""
    if target_vol is not None:
        weights = engine.vol_target(weights, ret, interval, target_vol=target_vol,
                                    max_leverage=max_leverage)
    sim = engine.simulate(weights, ret, interval)
    net = sim["net"]
    is_net, oos_net = split(net)
    mi = engine.metrics(is_net, interval, name + " [IS]")
    mo = engine.metrics(oos_net, interval, name + " [OOS]")
    if verbose:
        print(engine.fmt_metrics(mi)); print(engine.fmt_metrics(mo))
        msi, mso = engine.monthly_summary(is_net), engine.monthly_summary(oos_net)
        ge = sim["gross_exposure"]
        if msi and mso:
            print(f"{'':>24}  monthly IS mean={msi['mean_monthly']*100:>5.1f}% "
                  f"med={msi['median_monthly']*100:>5.1f}% +{msi['pct_positive']*100:.0f}% "
                  f"worst={msi['worst']*100:.0f}% | OOS mean={mso['mean_monthly']*100:>5.1f}% "
                  f"med={mso['median_monthly']*100:>5.1f}% +{mso['pct_positive']*100:.0f}% "
                  f"worst={mso['worst']*100:.0f}%")
            print(f"{'':>24}  avg gross exposure={ge.mean():.2f}x  turnover/bar={sim['turnover'].mean():.3f}")
        print()
    return net, sim, (mi, mo)


if __name__ == "__main__":
    for interval, lbs in [("1h", (168, 336, 720, 1440)), ("1d", (15, 30, 60, 90))]:
        close, vol, ret, elig = load_data(interval)
        print(f"\n########## {interval}  universe={close.shape[1]} bars={close.shape[0]} "
              f"({close.index[0].date()}..{close.index[-1].date()}) "
              f"elig median={int(elig.sum(1).median())} ##########\n")
        vlb = lbs[0]
        evaluate(f"TSmom-multi L/S (vt40)",
                 st.ts_momentum_multi(close, vol, lookbacks=lbs, vol_lookback=vlb, elig=elig),
                 ret, interval, target_vol=0.40)
        evaluate(f"TSmom-multi long-only (vt40)",
                 st.ts_momentum_multi(close, vol, lookbacks=lbs, vol_lookback=vlb, elig=elig, long_only=True),
                 ret, interval, target_vol=0.40)
        evaluate(f"TSmom-single L/S (vt40)",
                 st.ts_momentum(close, vol, lookback=lbs[1], vol_lookback=vlb, elig=elig),
                 ret, interval, target_vol=0.40)
        # no-vol-target comparison
        evaluate(f"TSmom-multi L/S (no vt)",
                 st.ts_momentum_multi(close, vol, lookbacks=lbs, vol_lookback=vlb, elig=elig),
                 ret, interval, target_vol=None)
