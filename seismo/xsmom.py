"""RIFT: cross-sectional residual-momentum long/short on USDT-perps (daily).

Second strategy family (built after SEISMO's anomaly decayed OOS).
Mechanism: persistent quality/trash dispersion between alt perps. The book
is dollar-neutral -- long the strongest residual momentum, short the
weakest -- so it does not depend on the direction of the alt market, which
is exactly what killed the cascade-reversion trade out-of-sample.

Discipline: tuned on the same IS fence (< 2024-07-01) with a coarse grid;
OOS is evaluated once for the frozen config.

Timing: signal from closes up to day t; trade at open(t+1); daily
open-to-open marks; turnover-based costs.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = "/tmp/claude-0/-home-user-binance-futures-klines/b68190b2-2a75-5b5b-b433-5f5a3e931263/scratchpad/cache/panel_1d.pkl"

LIQ_WIN, LIQ_MINP, LIQ_FLOOR = 30, 21, 5_000_000   # $5M/day median
BETA_WIN, VOL_WIN = 60, 30
IS_END = "2024-07-01"
COST_SIDE = 0.0010


def build_daily(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    opens, closes, dvols = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_USDT_1d.csv"))):
        sym = os.path.basename(path).replace("_USDT_1d.csv", "")
        df = pd.read_csv(path, usecols=["timestamp", "open", "close", "volume"])
        if len(df) < LIQ_MINP:
            continue
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index(idx).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        opens[sym], closes[sym] = df["open"], df["close"]
        dvols[sym] = df["close"] * df["volume"]
    O, C, DV = (pd.DataFrame(x) for x in (opens, closes, dvols))
    full = pd.date_range(C.index.min(), C.index.max(), freq="1D", tz="UTC")
    O, C, DV = (x.reindex(full) for x in (O, C, DV))
    liq = DV.rolling(LIQ_WIN, min_periods=LIQ_MINP).median()
    U = (liq.shift(1) >= LIQ_FLOOR) & C.notna()
    out = {"O": O, "C": C, "U": U}
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def signals(panel):
    C, U = panel["C"], panel["U"]
    R = np.log(C).diff()
    mkt = R.where(U).median(axis=1)
    Ex = R.rolling(BETA_WIN, min_periods=40).mean()
    Ey = mkt.rolling(BETA_WIN, min_periods=40).mean()
    Exy = R.mul(mkt, axis=0).rolling(BETA_WIN, min_periods=40).mean()
    beta = (Exy - Ex.mul(Ey, axis=0)).div(
        mkt.rolling(BETA_WIN, min_periods=40).var(), axis=0).clip(0, 3)
    resid = R.sub(beta.mul(mkt, axis=0))
    vol = R.rolling(VOL_WIN, min_periods=21).std()
    Of = panel["O"].ffill()
    RO = Of.pct_change().fillna(0.0)
    return {"C": C, "U": U, "R": R, "resid": resid, "vol": vol,
            "RO": RO, "idx": C.index, "cols": np.asarray(C.columns)}


def _leg_weights(vols, budget, w_cap=None):
    """Inverse-vol weights for one leg, optional per-name cap (x budget)."""
    iv = 1.0 / np.clip(vols, 1e-4, None)
    w = budget * iv / iv.sum()
    if w_cap is not None:
        cap = w_cap * budget
        for _ in range(4):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            free = ~over
            if not free.any() or w[free].sum() <= 0:
                break
            w[free] += excess * w[free] / w[free].sum()
    return w


def run_xs(sig, lookback=30, skip=2, n_side=10, rebal=7, use_resid=True,
           gross=2.0, cost_side=COST_SIDE, start=None, end=None,
           long_only=False, w_cap=None, gate_window=None):
    """Daily long/short backtest. gross = total notional / equity.

    gate_window: anomaly heartbeat -- live capital only deploys while the
    trailing mean of the last `gate_window` PAPER cycle returns is > 0.
    The paper book is always maintained.
    """
    idx = sig["idx"]
    U = sig["U"].values
    base = sig["resid"] if use_resid else sig["R"]
    mom = base.rolling(lookback).sum().shift(skip).values
    volv = sig["vol"].values
    ROv = sig["RO"].values
    n_bars, n_sym = ROv.shape

    i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start, tz="UTC"))
    i1 = n_bars if end is None else idx.searchsorted(pd.Timestamp(end, tz="UTC"))

    port = np.zeros(n_bars)                  # live daily returns
    pport = np.zeros(n_bars)                 # paper daily returns
    w = np.zeros(n_sym)                      # live weights
    wp = np.zeros(n_sym)                     # paper weights
    cyc_rets, pcyc_rets = [], []
    eq, peq = 1.0, 1.0
    cyc_start_eq, pcyc_start_eq = None, None
    live = gate_window is None
    next_rebal = i0
    for t in range(i0, i1 - 1):
        if next_rebal == t:
            score = np.where(U[t] & np.isfinite(mom[t]) & np.isfinite(volv[t]),
                             mom[t], np.nan)
            ok = np.flatnonzero(np.isfinite(score))
            w_new = np.zeros(n_sym)
            if len(ok) >= 2 * n_side + 5:
                order = ok[np.argsort(score[ok])]
                shorts, longs = order[:n_side], order[-n_side:]
                w_new[longs] = _leg_weights(volv[t][longs], gross / 2, w_cap)
                if not long_only:
                    w_new[shorts] = -_leg_weights(volv[t][shorts], gross / 2,
                                                  w_cap)
            if pcyc_start_eq is not None:
                pcyc_rets.append(peq / pcyc_start_eq - 1.0)
            pcyc_start_eq = peq
            if gate_window is not None:
                live = (len(pcyc_rets) >= gate_window and
                        float(np.mean(pcyc_rets[-gate_window:])) > 0)
            pport[t + 1] -= cost_side * np.abs(w_new - wp).sum()
            wp = w_new
            w_live_new = w_new if live else np.zeros(n_sym)
            port[t + 1] -= cost_side * np.abs(w_live_new - w).sum()
            w = w_live_new
            next_rebal = t + rebal
            if cyc_start_eq is not None:
                cyc_rets.append(eq / cyc_start_eq - 1.0)
            cyc_start_eq = eq
        # position return from open(t+1) to open(t+2) accrues at t+2
        if t + 2 < n_bars:
            r = np.nan_to_num(ROv[t + 2])
            if np.any(w):
                port[t + 2] += float(np.dot(w, r))
            if np.any(wp):
                pport[t + 2] += float(np.dot(wp, r))
        eq *= 1.0 + port[t + 1]
        peq *= 1.0 + pport[t + 1]
    return pd.Series(port, index=idx).iloc[i0:i1], np.array(cyc_rets)


def stats(port, cyc, label=""):
    eq = (1 + port).cumprod()
    yrs = max(len(port) / 365.25, 1e-9)
    dd = (eq / eq.cummax() - 1).min()
    shp = port.mean() / port.std() * np.sqrt(365) if port.std() > 0 else np.nan
    out = {"label": label, "total_return": eq.iloc[-1] - 1,
           "cagr": eq.iloc[-1] ** (1 / yrs) - 1, "max_dd": dd, "sharpe": shp,
           "cycles": len(cyc)}
    if len(cyc):
        g, l = cyc[cyc > 0].sum(), -cyc[cyc <= 0].sum()
        out["cycle_winrate"] = (cyc > 0).mean()
        out["cycle_pf"] = g / l if l > 0 else np.inf
    return out


def fmt(m):
    return (f"{m['label']:<44} totRet={m['total_return'] * 100:>8.1f}% "
            f"cagr={m['cagr'] * 100:>6.1f}% dd={m['max_dd'] * 100:>6.1f}% "
            f"shp={m['sharpe']:>5.2f} cyc={m['cycles']:>4} "
            f"WR={m.get('cycle_winrate', float('nan')) * 100:>5.1f}% "
            f"PF={m.get('cycle_pf', float('nan')):>5.2f}")


if __name__ == "__main__":
    import itertools
    panel = build_daily(force=True)
    sig = signals(panel)
    print("universe size mean/min/max:",
          round(panel["U"].sum(axis=1).mean(), 1),
          int(panel["U"].sum(axis=1).min()), int(panel["U"].sum(axis=1).max()))
    rows = []
    grid = {"lookback": [14, 30, 60], "skip": [0, 2], "n_side": [5, 10],
            "rebal": [3, 7], "use_resid": [False, True]}
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        port, cyc = run_xs(sig, end=IS_END, **kw)
        m = stats(port, cyc, ""); m.update(kw); rows.append(m)
    df = pd.DataFrame(rows).drop(columns="label")
    df.to_csv("seismo/out/grid_is_xsmom.csv", index=False)
    ok = df[df["cycles"] >= 50]
    cols = list(grid) + ["total_return", "cagr", "max_dd", "sharpe",
                         "cycle_winrate", "cycle_pf"]
    print("\ntop 12 by IS Sharpe:")
    print(ok.sort_values("sharpe", ascending=False)[cols].head(12)
          .round(3).to_string(index=False))
    print(f"\ncombos: {len(df)}, positive IS: {(ok['total_return'] > 0).mean():.0%}, "
          f"PF>1: {(ok['cycle_pf'] > 1).mean():.0%}")
