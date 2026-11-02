"""Does crypto mean-revert at the horizons a grid trades? (Lo-MacKinlay variance ratios)

A grid is short volatility: on a martingale its expected P&L is zero before fees (Chen, Chen &
Jang 2025; Quantpedia "grid trading vs delta hedging"). Its only edges are drift (a long grid in a
rising coin) and MEAN REVERSION at the grid's scale: more back-and-forth crossings of the grid
levels than a random walk produces. The variance ratio VR(q) = Var(q-hour return) / (q * Var(1h))
measures exactly that: VR < 1 means reversal (good for grids), VR > 1 means trending (bad).

Uses the heteroskedasticity-robust z* of Lo & MacKinlay (1988), per symbol and calendar year.
Also tests idiosyncratic returns (coin minus beta*BTC), the relevant series for a hedged grid.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data

QS = (2, 4, 8, 24, 72, 168)            # 2h .. 1 week
MIN_OBS = 4000                          # hours per symbol-year


def vr_stats(r: np.ndarray, qs=QS):
    """VR(q) and robust z*(q) for a 1-D array of returns (no NaN)."""
    e = r - r.mean()
    T = len(e)
    s2 = e @ e
    out = {}
    maxj = max(qs) - 1
    rho = np.array([e[j:] @ e[:-j] / s2 for j in range(1, maxj + 1)])
    e2 = e * e
    delta = np.array([T * (e2[j:] @ e2[:-j]) / s2 ** 2 for j in range(1, maxj + 1)])
    for q in qs:
        j = np.arange(1, q)
        w = 2 * (1 - j / q)
        vr = 1 + (w * rho[: q - 1]).sum()
        theta = ((2 * (q - j) / q) ** 2 * delta[: q - 1]).sum()
        out[q] = (vr, np.sqrt(T) * (vr - 1) / np.sqrt(theta))
    return out


def table(ret: pd.DataFrame, label: str):
    rows = []
    for sym in ret.columns:
        s = ret[sym].dropna()
        for yr, g in s.groupby(s.index.year):
            if len(g) < MIN_OBS:
                continue
            st = vr_stats(g.to_numpy())
            for q, (vr, z) in st.items():
                rows.append((sym, yr, q, vr, z))
    df = pd.DataFrame(rows, columns=["sym", "year", "q", "vr", "z"])
    print(f"\n=== {label}: median VR across symbols (n symbol-years={df.groupby(['sym','year']).ngroups}) ===")
    med = df.pivot_table(index="year", columns="q", values="vr", aggfunc="median")
    cnt = df[df.q == 24].groupby("year").sym.count()
    med["n_sym"] = cnt
    print(med.round(3).to_string())
    print("share of symbol-years with significant mean reversion (z* < -1.96) / momentum (z* > 1.96):")
    sig = df.groupby("q").apply(lambda g: pd.Series({"reversal": (g.z < -1.96).mean(), "trend": (g.z > 1.96).mean()}),
                                include_groups=False)
    print(sig.round(3).T.to_string())
    return df


def main():
    c = data.load("close", "1h")
    r = np.log(c.astype("float64")).diff()
    r = r.where(r.abs() < 0.7)                         # drop obviously broken prints
    raw = table(r, "RAW hourly log returns")
    b = r["BTC"]
    beta = r.rolling(24 * 30, min_periods=24 * 10).cov(b).div(b.rolling(24 * 30, min_periods=24 * 10).var(), axis=0)
    idio = (r - beta.shift(1).mul(b, axis=0)).drop(columns=["BTC"])
    table(idio, "IDIOSYNCRATIC (coin - beta*BTC)")
    for s in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
        g = raw[raw.sym == s].pivot_table(index="year", columns="q", values="vr")
        print(f"\n{s} VR by year:\n{g.round(3).to_string()}")
    raw.to_csv("results/grid_vr.csv", index=False)


if __name__ == "__main__":
    main()
