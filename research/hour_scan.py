"""E1: does the daily rebalance boundary (00:00 UTC) matter? Rebuild daily
bars from local 1h data at boundary hour h in {0,4,8,12,16,20} and rerun the
A+B book per boundary (sleeve C is calendar-day data, so the scan uses the
80% of the book that is boundary-sensitive). Internal comparison only --
same window (1h data: 2022-05 ->), IS-gated with a plateau rule.

E2: empirical breadth-scaling of sleeve C (IR ~ IC*sqrt(breadth)): IS Sharpe
on random half-coverage vs full 40-symbol coverage. Motivates the user-side
fetch of metrics for all ~140 symbols.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import altdata
import data
import echo
import engine
import lab
import strategies as st
from apex import hold_cadence

IS_END = lab.IS_END


def daily_at(h):
    C1 = data.load("close", "1h")
    V1 = data.load("volume", "1h")
    off = pd.Timedelta(hours=h)
    close = C1.resample("24h", offset=off).last()
    vol = V1.resample("24h", offset=off).sum(min_count=1)
    keep = close.columns[close.notna().sum() > 60]
    return close[keep], vol[keep]


def ab_book_net(close, vol):
    ret = engine.to_returns(close)
    elig = st.eligibility(close, vol, 60, 5e6, 30)
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90),
                              strength_vol=30, k=2.0)
    A = st.concentrate(st.ema_ensemble(
        st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    B = echo.residual_continuation(close, ret, elig, beta_window=60,
                                   formations=(3, 5, 8), cap=0.10)
    W = hold_cadence(0.5 * A + 0.5 * B, 3)
    wv = engine.vol_target(W, ret, "1d", target_vol=0.40, max_leverage=3.0)
    return engine.simulate(wv, ret, "1d")["net"]


def main():
    # ---- E1: boundary scan ---------------------------------------------------
    hours = [0, 4, 8, 12, 16, 20]
    rows = []
    for h in hours:
        close, vol = daily_at(h)
        net = ab_book_net(close, vol)
        i, _ = lab.split(net)
        m = engine.metrics(i, "1d")
        rows.append({"boundary_utc": h, "IS_sharpe": m["Sharpe"],
                     "IS_cagr": m["CAGR"], "IS_dd": m["maxDD"]})
    df = pd.DataFrame(rows)
    base = df.loc[df.boundary_utc == 0, "IS_sharpe"].iloc[0]
    print("E1  A+B book by daily-close boundary (1h-derived, window 2022-05->):")
    print(df.round(3).to_string(index=False))
    best = df.loc[df.IS_sharpe.idxmax()]
    verdict = "keep 00:00"
    if best.boundary_utc != 0 and best.IS_sharpe > base + 0.05:
        k = hours.index(int(best.boundary_utc))
        nb = [df.IS_sharpe[j] for j in (k - 1, (k + 1) % len(hours))]
        if all(v >= base for v in nb):
            verdict = f"ACCEPT boundary {int(best.boundary_utc)}:00 UTC"
    print(f"    gate: base(h=0)={base:.2f}, best h={int(best.boundary_utc)} "
          f"({best.IS_sharpe:.2f}) -> {verdict}")
    df.to_csv("../seismo/out/hour_scan.csv", index=False)

    # ---- E2: sleeve C breadth scaling ---------------------------------------
    close, vol, ret, elig = lab.load_data("1d")
    ls = altdata.load_metric("toptrader_pos_ls", reindex_like=close)
    covered = [c for c in ls.columns if ls[c].notna().sum() > 100]

    def c_sharpe(cols):
        mask = pd.DataFrame(False, index=ls.index, columns=ls.columns)
        mask[cols] = True
        C = echo.positioning_sleeve(close, ret, elig, ls.where(mask),
                                    fade=True, lag=1, beta_window=60, cap=0.10)
        wv = engine.vol_target(C, ret, "1d", target_vol=0.40, max_leverage=3.0)
        net = engine.simulate(wv, ret, "1d")["net"]
        return engine.metrics(lab.split(net)[0], "1d")["Sharpe"]

    full = c_sharpe(covered)
    rng = np.random.default_rng(5)
    halves = [c_sharpe(list(rng.choice(covered, len(covered) // 2,
                                       replace=False))) for _ in range(6)]
    print(f"\nE2  sleeve C breadth scaling: full n={len(covered)} IS Sharpe "
          f"{full:.2f}; half-coverage mean {np.mean(halves):.2f} "
          f"(6 draws: {[round(x, 2) for x in halves]})")
    print(f"    ratio full/half = {full / np.mean(halves):.2f} "
          f"(sqrt-breadth predicts {np.sqrt(2):.2f}) -> extrapolation to "
          f"~120 covered symbols: C standalone ~{full * np.sqrt(3):.1f} "
          f"IF the IC holds on the wider universe")


if __name__ == "__main__":
    main()
