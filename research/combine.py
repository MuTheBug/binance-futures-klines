"""
Test ONE robust improvement: timeframe diversification (daily trend + hourly trend).
Higher Sharpe -> higher growth-optimal leverage -> higher achievable return.
Only keep it if it clearly helps on IS (avoid over-engineering = a form of overfitting).
Each sleeve's P&L is reduced to a DAILY return series and combined risk-weighted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st
import lab


def daily_series_from(net, interval):
    """Compound a net-return series to daily (UTC) returns."""
    if interval == "1d":
        return net
    eq = (1 + net.fillna(0)).cumprod()
    d = eq.resample("1D").last().pct_change().dropna()
    return d


def sharpe(s):
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(365)) if s.std() > 0 else 0.0


def stats(name, s):
    is_s, oos_s = lab.split(s)
    mi, mo = engine.metrics(is_s, "1d", name + " IS"), engine.metrics(oos_s, "1d", name + " OOS")
    print(f"  {name:>22}  IS Sh={mi['Sharpe']:>5.2f} CAGR={mi['CAGR']*100:>6.1f}% DD={mi['maxDD']*100:>5.0f}% "
          f"| OOS Sh={mo['Sharpe']:>5.2f} CAGR={mo['CAGR']*100:>6.1f}% DD={mo['maxDD']*100:>5.0f}%")
    return mi, mo


def main():
    # ---- daily sleeve (flagship, top-8) ----
    c1, v1, r1, e1 = lab.load_data("1d")
    wd = st.ts_momentum_multi(c1, v1, lookbacks=(15, 30, 60, 90), vol_lookback=15, elig=e1)
    wd = st.concentrate(wd, 8)
    wd = engine.vol_target(wd, r1, "1d", target_vol=0.40)
    daily_net = engine.simulate(wd, r1, "1d")["net"]

    # ---- hourly sleeve (single-lookback L/S trend that survived OOS) ----
    c2, v2, r2, e2 = lab.load_data("1h")
    wh = st.ts_momentum(c2, v2, lookback=336, vol_lookback=168, elig=e2)
    wh = st.concentrate(wh, 8)
    wh = engine.vol_target(wh, r2, "1h", target_vol=0.40)
    hourly_net = engine.simulate(wh, r2, "1h")["net"]
    hourly_daily = daily_series_from(hourly_net, "1h")

    # align
    df = pd.DataFrame({"daily": daily_net, "hourly": hourly_daily}).dropna()
    corr = df["daily"].corr(df["hourly"])
    print(f"daily vs hourly trend: overlap days={len(df)}, correlation={corr:.2f}\n")

    print("Sleeves alone:")
    stats("daily-trend top8", df["daily"])
    stats("hourly-trend top8", df["hourly"])

    print("\nCombinations (re-vol-targeted to 40% on the combined daily series):")
    for wname, wd_, wh_ in [("50/50", 0.5, 0.5), ("60/40 d/h", 0.6, 0.4), ("70/30 d/h", 0.7, 0.3)]:
        combo = wd_ * df["daily"] + wh_ * df["hourly"]
        # re-scale to ~40% annual vol using trailing 30d vol (lagged)
        rv = combo.rolling(30, min_periods=15).std() * np.sqrt(365)
        scale = (0.40 / rv).shift(1).clip(upper=3.0).fillna(0.0)
        combo_vt = combo * scale
        stats(f"combo {wname}", combo_vt)

    # also: simple inverse-vol blend weight
    print("\nReference: daily-only re-vol-targeted (same overlap window):")
    rv = df["daily"].rolling(30, min_periods=15).std() * np.sqrt(365)
    scale = (0.40 / rv).shift(1).clip(upper=3.0).fillna(0.0)
    stats("daily-only (vt)", df["daily"] * scale)


if __name__ == "__main__":
    main()
