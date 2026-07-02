"""Independent re-implementation of ECHO v2 in THIS branch's engine.

Only the strategy *specification* is taken from the other branch
(sleeve signal math, smoothing, caps, 40/40/20 blend, 40% vol target):

  A: multi-lookback tanh trend, inverse-vol sized, EMA{5,10,15}, top-10
  B: beta-neutral residual continuation, F={3,5,8}, tanh z, cap 10%
  C: fade top-trader positioning crowding (lag 1), beta-neutral, cap 10%

Everything else is deliberately MY implementation, different from theirs:
  - universe: my $5M/day median-dollar-volume point-in-time filter
  - execution: signal at close t -> fill at next OPEN -> open-to-open marks
    (theirs: close-to-close with 1-bar lag)
  - costs: 10 bps/side on weight changes, charged at the fill
  - my own vol-targeting plumbing and metrics code

If ECHO's alpha is real it must survive this translation.
"""
import glob
import os

import numpy as np
import pandas as pd

from seismo.xsmom import build_daily

ALT = "/tmp/claude-0/-home-user-binance-futures-klines/b68190b2-2a75-5b5b-b433-5f5a3e931263/scratchpad/altdata"
COST = 0.0010
TARGET_VOL = 0.40


def load_toptrader(index, columns):
    cols = {}
    for f in glob.glob(os.path.join(ALT, "*_USDT_metrics_1d.csv")):
        base = os.path.basename(f)[: -len("_USDT_metrics_1d.csv")]
        df = pd.read_csv(f, usecols=["date", "toptrader_pos_ls"])
        s = pd.Series(df["toptrader_pos_ls"].values,
                      index=pd.to_datetime(df["date"], utc=True))
        cols[base] = s[~s.index.duplicated()]
    return pd.DataFrame(cols).reindex(index).reindex(columns=columns)


def norm_gross(w, gross=1.0):
    s = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(s, axis=0).mul(gross).fillna(0.0)


def xz(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0.0, np.nan), axis=0)


def build_weights():
    panel = build_daily()
    C, U = panel["C"], panel["U"]
    ret = C.pct_change()
    mkt = ret.where(U).mean(axis=1)                      # equal-weight factor
    mp = 30
    Ex = ret.rolling(60, min_periods=mp).mean()
    Ey = mkt.rolling(60, min_periods=mp).mean()
    Exy = ret.mul(mkt, axis=0).rolling(60, min_periods=mp).mean()
    beta = (Exy - Ex.mul(Ey, axis=0)).div(
        mkt.rolling(60, min_periods=mp).var(), axis=0)
    eps = ret.sub(beta.mul(mkt, axis=0)).where(U)

    # ---- sleeve A: trend --------------------------------------------------
    vol30 = ret.rolling(30, min_periods=15).std()
    sigA = sum(np.tanh(2.0 * C.pct_change(L) / (vol30 * np.sqrt(L)))
               for L in (15, 30, 60, 90)) / 4
    vol15 = ret.rolling(15, min_periods=8).std().replace(0.0, np.nan)
    wA = norm_gross((sigA / vol15).where(U, 0.0))
    wA = sum(wA.ewm(span=s).mean() for s in (5, 10, 15)) / 3
    keep = wA.abs().rank(axis=1, ascending=False, method="first") <= 10
    g0 = wA.abs().sum(axis=1)
    wA = wA.where(keep, 0.0)
    wA = wA.mul(g0 / wA.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    # ---- sleeve B: residual continuation ----------------------------------
    sigB = sum(np.tanh(xz(eps.rolling(F, min_periods=F).sum().where(U)))
               for F in (3, 5, 8)) / 3
    sigB = sigB.sub(sigB.mean(axis=1), axis=0).where(U)
    wB = norm_gross(sigB.fillna(0.0))
    wB = (sum(wB.ewm(span=s).mean() for s in (3, 5, 8)) / 3).clip(-0.10, 0.10)
    wB = norm_gross(wB)

    # ---- sleeve C: fade crowd positioning ----------------------------------
    ls = load_toptrader(C.index, C.columns).shift(1).where(U)
    sigC = -np.tanh(xz(ls))
    sigC = sigC.sub(sigC.mean(axis=1), axis=0).where(U)
    wC = (sigC / vol30.replace(0.0, np.nan)).where(U, 0.0)
    b = beta.where(U, 0.0)
    proj = (wC * b).sum(axis=1) / (b * b).sum(axis=1).replace(0.0, np.nan)
    wC = wC.sub(b.mul(proj, axis=0), fill_value=0.0).where(U, 0.0)
    wC = (sum(wC.ewm(span=s).mean() for s in (3, 5, 8)) / 3).clip(-0.10, 0.10)
    wC = norm_gross(wC)

    W = 0.4 * wA + 0.4 * wB + 0.2 * wC
    return panel, W


def simulate(panel, W, cost_side=COST):
    """My convention: W fixed at close t, filled at open(t+1), earns
    open(t+1)->open(t+2); costs on |dW| at the fill."""
    Of = panel["O"].ffill()
    RO = Of.pct_change().fillna(0.0)
    Wl = W.fillna(0.0)
    gross = (Wl.shift(2) * RO).sum(axis=1)
    cost = (Wl.diff().abs().sum(axis=1) * cost_side).shift(1).fillna(0.0)
    base = gross - cost
    rv = base.rolling(30, min_periods=15).std() * np.sqrt(365)
    scale = (TARGET_VOL / rv).shift(1).clip(upper=3.0).fillna(0.0)
    Ws = Wl.mul(scale, axis=0)
    gross_s = (Ws.shift(2) * RO).sum(axis=1)
    cost_s = (Ws.diff().abs().sum(axis=1) * cost_side).shift(1).fillna(0.0)
    return gross_s - cost_s


def stats(r, label):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.25
    g, l = r[r > 0].sum(), -r[r <= 0].sum()
    act = r[r != 0]
    return {"label": label, "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(365),
            "max_dd": (eq / eq.cummax() - 1).min(), "daily_pf": g / l,
            "daily_wr": (act > 0).mean(),
            "ann_vol": r.std() * np.sqrt(365), "total_x": eq.iloc[-1]}


if __name__ == "__main__":
    panel, W = build_weights()
    net = simulate(panel, W)
    net.to_csv("seismo/out/echo_replica_net.csv")

    theirs = pd.read_csv("seismo/out/echo_v2_net_10bps.csv", index_col=0)
    theirs.index = pd.to_datetime(theirs.index, utc=True)
    theirs = theirs.iloc[:, 0]
    both = pd.concat([net.rename("replica"), theirs.rename("theirs")],
                     axis=1).dropna()
    print(f"daily-return corr(replica, their code): "
          f"{both['replica'].corr(both['theirs']):+.2f}\n")

    rows = []
    for label, s, e in [("their IS (<=2024-12)", None, "2025-01-01"),
                        ("their OOS / common OOS (>=2025-01)", "2025-01-01", None),
                        ("RIFT OOS window (>=2024-07)", "2024-07-01", None),
                        ("full", None, None)]:
        r = net
        if s:
            r = r.loc[pd.Timestamp(s, tz="UTC"):]
        if e:
            r = r.loc[:pd.Timestamp(e, tz="UTC")]
        rows.append(stats(r, f"replica | {label}"))
    t = pd.DataFrame(rows)
    t.to_csv("seismo/out/echo_replica_stats.csv", index=False)
    print(t.round(3).to_string(index=False))
