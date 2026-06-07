"""
Alternative bar types & volume profile, built from 1h OHLCV and mapped to the DAILY
grid as [-1,1] signal matrices (so the existing daily engine/costs apply unchanged).

NO look-ahead / NO repaint:
  * dollar/volume bars: a bar is emitted only when cumulative dollar/volume crosses a
    threshold; its signal is stamped at the COMPLETION timestamp and forward-filled.
  * Renko: a brick is confirmed only once price has moved one brick beyond the prior
    brick close — known in real time; stamped at the bar where it completes.
  * Volume profile: fixed-width LOG price bins (anchored, not data-range) so future
    extremes never change a past bin; POC/value-area use only the trailing window.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data


def _to_daily(times, values, daily_index) -> pd.Series:
    """Stamp event values at their timestamps and as-of (ffill) onto the daily grid."""
    if len(times) == 0:
        return pd.Series(0.0, index=daily_index)
    s = pd.Series(values, index=pd.DatetimeIndex(times)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(daily_index, method="ffill")


# --------------------------- dollar / volume bars ---------------------------
def _event_bar_edges(weight_per_hour: np.ndarray, K: float):
    """Indices where cumulative weight (dollar or volume) crosses k*threshold.
    threshold = median positive hourly weight * K."""
    w = np.nan_to_num(weight_per_hour, nan=0.0)
    pos = w[w > 0]
    if len(pos) < 50:
        return None
    thr = np.median(pos) * K
    if thr <= 0:
        return None
    cum = np.cumsum(w)
    if cum[-1] < thr:
        return None
    edges = np.searchsorted(cum, np.arange(thr, cum[-1], thr))
    return np.unique(np.clip(edges, 0, len(w) - 1))


def bar_trend_signal(kind: str, daily_index, lookbacks_bars=(4, 8, 16, 32), K=24.0):
    """kind: 'dollar' or 'volume'. Multi-lookback sign-trend computed on event bars,
    then mapped to the daily grid. Returns a DataFrame (daily_index x symbols) in [-1,1]."""
    close = data.load("close", "1h")
    vol = data.load("volume", "1h")
    out = {}
    for sym in close.columns:
        c = close[sym].to_numpy(dtype=float)
        v = vol[sym].to_numpy(dtype=float)
        idx = close.index
        m = ~np.isnan(c)
        if m.sum() < 200:
            continue
        c_, v_, t_ = c[m], v[m], idx[m]
        weight = c_ * v_ if kind == "dollar" else v_
        edges = _event_bar_edges(weight, K)
        if edges is None or len(edges) < max(lookbacks_bars) + 5:
            continue
        bc = c_[edges]                       # bar closes
        bt = t_[edges]                       # bar completion timestamps
        bc_s = pd.Series(bc)
        sig = np.zeros(len(bc))
        for L in lookbacks_bars:
            sig += np.sign(bc_s.pct_change(L).to_numpy())
        sig = np.nan_to_num(sig / len(lookbacks_bars))
        out[sym] = _to_daily(bt, sig, daily_index)
    return pd.DataFrame(out).reindex(daily_index)


# --------------------------- Renko ---------------------------
def _renko_dirs(logc: np.ndarray, times, brick: float):
    """Return (timestamps, brick_close_levels, directions) for confirmed bricks."""
    last = logc[0]
    bt, blvl, bdir = [], [], []
    for i in range(1, len(logc)):
        x = logc[i]
        while x - last >= brick:
            last += brick; bt.append(times[i]); blvl.append(last); bdir.append(1)
        while last - x >= brick:
            last -= brick; bt.append(times[i]); blvl.append(last); bdir.append(-1)
    return bt, np.array(blvl), np.array(bdir)


def renko_trend_signal(daily_index, brick_pct=0.03, lookbacks_bricks=(3, 6, 12, 24),
                       interval="1h"):
    """Renko brick trend (sign-averaged over several brick-lookbacks), mapped to daily."""
    close = data.load("close", interval)
    brick = np.log(1.0 + brick_pct)
    out = {}
    for sym in close.columns:
        c = close[sym].to_numpy(dtype=float)
        idx = close.index
        m = ~np.isnan(c) & (c > 0)
        if m.sum() < 300:
            continue
        logc, t_ = np.log(c[m]), idx[m]
        bt, blvl, bdir = _renko_dirs(logc, t_, brick)
        if len(blvl) < max(lookbacks_bricks) + 3:
            continue
        lvl = pd.Series(blvl)
        sig = np.zeros(len(blvl))
        for L in lookbacks_bricks:
            sig += np.sign(lvl.diff(L).to_numpy())     # net up/down over L bricks
        sig = np.nan_to_num(sig / len(lookbacks_bricks))
        out[sym] = _to_daily(bt, sig, daily_index)
    return pd.DataFrame(out).reindex(daily_index)


# --------------------------- volume profile ---------------------------
def volume_profile(daily_index, window_h=720, bin_pct=0.01, value_area=0.70):
    """Rolling volume profile from 1h bars. Returns dict of daily-grid DataFrames:
    'poc_dist'  = (close-POC)/close  (negative => price below POC)
    'va_pos'    = +1 above value-area-high, -1 below value-area-low, 0 inside.
    Fixed-width log bins (no look-ahead). Sampled on the daily grid."""
    close = data.load("close", "1h"); high = data.load("high", "1h")
    low = data.load("low", "1h"); vol = data.load("volume", "1h")
    binstep = np.log(1.0 + bin_pct)
    poc_dist, va_pos = {}, {}
    for sym in close.columns:
        c = close[sym].to_numpy(float); h = high[sym].to_numpy(float)
        lo = low[sym].to_numpy(float); v = vol[sym].to_numpy(float)
        idx = close.index
        m = ~np.isnan(c) & (c > 0)
        if m.sum() < window_h + 50:
            continue
        c, h, lo, v, t_ = c[m], h[m], lo[m], v[m], idx[m]
        tp = np.log((h + lo + c) / 3.0)                 # typical price (log)
        ref = tp[0]
        b = np.floor((tp - ref) / binstep).astype(int)  # bin index per hour (fixed anchor)
        b -= b.min()
        nb = b.max() + 1
        T = len(c)
        # scatter volume into (T x nb) then rolling-window sum via cumsum
        M = np.zeros((T, nb))
        M[np.arange(T), b] = v
        cum = np.vstack([np.zeros(nb), np.cumsum(M, axis=0)])
        # sample on daily timestamps that fall within this symbol's range
        day_ts = daily_index[(daily_index >= t_[window_h]) & (daily_index <= t_[-1])]
        pos_in_t = np.searchsorted(t_.asi8, day_ts.asi8, side="right") - 1  # last hour <= day close
        pd_vals, va_vals = [], []
        bin_price = np.exp(ref + (np.arange(nb) + 0.5) * binstep)
        for p in pos_in_t:
            lo_i = max(0, p - window_h)
            wprof = cum[p + 1] - cum[lo_i]               # volume per bin over window
            tot = wprof.sum()
            if tot <= 0:
                pd_vals.append(0.0); va_vals.append(0.0); continue
            poc = int(np.argmax(wprof))
            # expand around POC until >= value_area of volume -> VAH/VAL
            lo_b = hi_b = poc; acc = wprof[poc]
            target = value_area * tot
            while acc < target and (lo_b > 0 or hi_b < nb - 1):
                left = wprof[lo_b - 1] if lo_b > 0 else -1
                right = wprof[hi_b + 1] if hi_b < nb - 1 else -1
                if right >= left:
                    hi_b += 1; acc += wprof[hi_b]
                else:
                    lo_b -= 1; acc += wprof[lo_b]
            price = c[p]
            pd_vals.append((price - bin_price[poc]) / price)
            vah, val = bin_price[hi_b], bin_price[lo_b]
            va_vals.append(1.0 if price > vah else (-1.0 if price < val else 0.0))
        poc_dist[sym] = pd.Series(pd_vals, index=day_ts).reindex(daily_index)
        va_pos[sym] = pd.Series(va_vals, index=day_ts).reindex(daily_index)
    return {"poc_dist": pd.DataFrame(poc_dist).reindex(daily_index),
            "va_pos": pd.DataFrame(va_pos).reindex(daily_index)}
