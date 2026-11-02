"""Grid-trading study on hourly Binance USDT-M data (crypto only).

Pre-registered protocol (written before any grid result was seen):
  * Design on data <= 2024-12-31 (IS). Test once on 2025-01-01 .. 2026-06-07 (OOS) and on a
    random half of the coins held out from design (HOLDOUT, seed 11, IS period).
  * Costs: maker 0.02% (grid fills), taker 0.05% + 0.05% slippage (market trades),
    funding 11%/yr paid by longs / received by shorts.
  * Universe: crypto perps (tokenized stocks/metals excluded), trailing-30d median daily $ volume
    >= $10M, >= 30 days listed. One grid per eligible coin, equal capital, 1x max exposure.
  * PASS = IS Sharpe >= 1.0, OOS Sharpe >= 0.7, HOLDOUT Sharpe >= 0.7, max DD <= 35% (1x).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import data
from gridsim import GridParams, simulate
from final_v3 import EXCLUDE

IS_END = pd.Timestamp("2024-12-31 23:00", tz="UTC")
START = "2022-06-01"


def load(start=START):
    f = {k: data.load(k, "1h").loc[start:] for k in ("open", "high", "low", "close", "volume")}
    keep = [s for s in f["close"].columns if s not in EXCLUDE]
    f = {k: v[keep].astype("float64") for k, v in f.items()}
    dv = (f["close"] * f["volume"]).rolling(24, min_periods=20).sum()           # daily $ volume
    liq = dv.rolling(24 * 30, min_periods=24 * 20).median()
    age = f["close"].notna().cumsum()
    elig = (liq >= 1e7) & (age >= 24 * 30) & f["close"].notna()
    return f, elig


def split_coins(cols, seed=11):
    rng = np.random.default_rng(seed)
    cols = sorted(cols)
    idx = rng.permutation(len(cols))
    return [cols[i] for i in idx[: len(cols) // 2]], [cols[i] for i in idx[len(cols) // 2:]]


def hourly_portfolio(eq: pd.DataFrame, active: pd.DataFrame, capital: pd.DataFrame) -> pd.Series:
    """Equal allocation across the grids running at the previous hour's close: each grid's P&L is
    measured on the capital it was sized from (so a nearly-wiped grid can't post +9000% hours)."""
    pnl = eq.diff()
    w = active.shift(1, fill_value=False) & pnl.notna()
    return (pnl / capital.shift(1)).where(w).mean(axis=1).fillna(0.0)


def stats(r: pd.Series) -> dict:
    d = (1 + r).resample("D").prod() - 1
    if len(d) < 30 or d.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "mdd": np.nan, "vol": np.nan, "pos_m": np.nan}
    eq = (1 + d).cumprod()
    yrs = len(d) / 365
    m = (1 + d).resample("ME").prod() - 1
    return {"sharpe": d.mean() / d.std() * np.sqrt(365), "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "mdd": (eq / eq.cummax() - 1).min(), "vol": d.std() * np.sqrt(365), "pos_m": (m > 0).mean(),
            "worst_m": m.min()}


def run(f, elig, gp: GridParams, cols=None, g_frame=None, allow=None, bias_frame=None):
    cols = cols or list(f["close"].columns)
    al = elig[cols] if allow is None else (elig[cols] & allow.reindex(columns=cols).fillna(False))
    gf = None if g_frame is None else g_frame[cols]
    bf = None if bias_frame is None else bias_frame[cols]
    res = simulate(f["open"][cols], f["high"][cols], f["low"][cols], f["close"][cols], gp,
                   g_frame=gf, allow=al, bias_frame=bf)
    return res, hourly_portfolio(res["equity"], res["active"], res["capital"])


def evaluate(name, res, cols_dev, cols_hold, quiet=False):
    """IS on DEV coins (design), IS on HOLDOUT coins, OOS on all coins. Returns dict of stats."""
    eq, act, cap = res["equity"], res["active"], res["capital"]
    pf = lambda cols: hourly_portfolio(eq[cols], act[cols], cap[cols])
    dev, hold, allc = pf(cols_dev), pf(cols_hold), pf(list(eq.columns))
    out = {"name": name,
           "dev": stats(dev[dev.index <= IS_END]), "hold": stats(hold[hold.index <= IS_END]),
           "oos": stats(allc[allc.index > IS_END]), "all_is": stats(allc[allc.index <= IS_END]),
           "fills_per_day": res["fills"].sum() / max(1, act.sum().sum() / 24),
           "liq_per_grid_year": res["liquidations"].sum() / max(1, act.sum().sum() / 8760)}
    if not quiet:
        d, h, o = out["dev"], out["hold"], out["oos"]
        print(f"{name:<44} DEV-IS {d['sharpe']:5.2f} {d['cagr']:7.1%} dd{d['mdd']:6.1%} | HOLD-IS {h['sharpe']:5.2f} "
              f"{h['cagr']:7.1%} | OOS {o['sharpe']:5.2f} {o['cagr']:7.1%} dd{o['mdd']:6.1%} | fills/d {out['fills_per_day']:.1f} liq/yr {out['liq_per_grid_year']:.2f}")
    return out


def report(name, r_all):
    a, b = r_all[r_all.index <= IS_END], r_all[r_all.index > IS_END]
    si, so = stats(a), stats(b)
    print(f"{name:<44} IS {si['sharpe']:5.2f} {si['cagr']:7.1%} dd {si['mdd']:6.1%} | "
          f"OOS {so['sharpe']:5.2f} {so['cagr']:7.1%} dd {so['mdd']:6.1%}")
    return si, so


def buy_and_hold(f, elig, cols=None):
    cols = cols or list(f["close"].columns)
    r = f["close"][cols].pct_change(fill_method=None)
    return r.where(elig[cols].shift(1, fill_value=False)).mean(axis=1).fillna(0.0)


if __name__ == "__main__":
    f, elig = load()
    print(f"universe {f['close'].shape[1]} coins, median eligible {int(elig.sum(1).median())}, "
          f"{f['close'].index[0].date()}..{f['close'].index[-1].date()}")
    report("benchmark: equal-weight buy & hold", buy_and_hold(f, elig))


# ----------------------------------------------------------------------------- features (causal)
def hourly_vol(f, hours=168):
    """Trailing std of hourly log returns (known at the close of each bar)."""
    r = np.log(f["close"]).diff()
    return r.rolling(hours, min_periods=hours // 2).std()


def daily_trend_on_hours(f, lookbacks=(15, 30, 60, 90), k=2.0):
    """v3's risk-adjusted trend strength in [-1, 1] from daily closes, stamped on the 23:00 bar
    (whose close IS the daily close) and carried forward through the next day."""
    d = f["close"].resample("D").last()
    r = d.pct_change(fill_method=None)
    vol = r.rolling(30, min_periods=15).std()
    s = sum(np.tanh(k * d.pct_change(L, fill_method=None) / (vol * np.sqrt(L))) for L in lookbacks) / len(lookbacks)
    s.index = s.index + pd.Timedelta(hours=23)
    return s.reindex(f["close"].index).ffill()


def trailing_vr(f, q=8, hours=24 * 30):
    """Trailing variance ratio VR(q) of hourly log returns over `hours` (overlapping q-sums)."""
    r = np.log(f["close"]).diff()
    rq = r.rolling(q).sum()
    v1 = r.rolling(hours, min_periods=hours // 2).var()
    vq = rq.rolling(hours, min_periods=hours // 2).var()
    return vq / (q * v1)
