"""Every honest lever on Profit Factor & Win Rate, IS-gated (fence 2024-12-31).

Acceptance rule (user choice: PROTECT RETURNS): a lever is accepted only if
it improves IS daily PF or WR AND keeps IS Sharpe and CAGR within 5% of the
base book. Everything is reported either way. One OOS read at the end for
the composed book. Leverage is excluded by construction: PF/WR are
scale-invariant.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import altdata
import echo
import engine
import lab
import strategies as st
from apex import hold_cadence
from trailing import portfolio_trailing

close, vol, ret, elig = lab.load_data("1d")
IS_END = lab.IS_END


def sleeves(spans_A=(5, 10, 15), spans_BC=(3, 5, 8)):
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90),
                              strength_vol=30, k=2.0)
    A = st.concentrate(st.ema_ensemble(
        st.signal_to_weights(sigT, ret, elig, 15), spans_A), 10)
    B = echo.residual_continuation(close, ret, elig, beta_window=60,
                                   formations=(3, 5, 8), cap=0.10,
                                   smooth_spans=spans_BC)
    ls = altdata.load_metric("toptrader_pos_ls", reindex_like=close)
    C = echo.positioning_sleeve(close, ret, elig, ls, fade=True, lag=1,
                                beta_window=60, cap=0.10, smooth=spans_BC)
    return A, B, C


def book_of(A, B, C):
    return hold_cadence(0.40 * A + 0.40 * B + 0.20 * C, 3)


def net_of(w, cost=0.0007, downside_vt=False):
    if not downside_vt:
        wv = engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0,
                               cost_per_side=cost)
        return engine.simulate(wv, ret, "1d", cost_per_side=cost)["net"]
    base = engine.simulate(w, ret, "1d", cost_per_side=cost)["net"]
    dn = base.clip(upper=0.0).rolling(30, min_periods=15).std() * np.sqrt(2 * 365)
    scale = (0.40 / dn).shift(1).clip(upper=3.0).fillna(0.0)
    wv = w.mul(scale, axis=0)
    return engine.simulate(wv, ret, "1d", cost_per_side=cost)["net"]


def pf_wr(r):
    r = r.dropna()
    r = r[r != 0]
    if len(r) == 0:
        return np.nan, np.nan
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return (g / l if l > 0 else np.inf), (r > 0).mean()


def evaluate(net, label):
    i, _ = lab.split(net)
    mi = engine.metrics(i, "1d")
    out = {"lever": label, "IS_sharpe": mi["Sharpe"], "IS_cagr": mi["CAGR"],
           "IS_dd": mi["maxDD"]}
    for tag, freq in [("d", None), ("w", "W"), ("m", "ME")]:
        rr = i if freq is None else (1 + i).resample(freq).prod() - 1
        pf, wr = pf_wr(rr)
        out[f"PF_{tag}"], out[f"WR_{tag}"] = pf, wr
    return out


def deadband(W, band):
    """Per-name no-trade band: keep the held weight when the new target
    moved less than band * typical position size (mirrors the bot's
    rebalance_band)."""
    Wv = W.values
    out = Wv.copy()
    held = np.zeros(Wv.shape[1])
    for t in range(len(Wv)):
        row = Wv[t]
        nnz = max((np.abs(row) > 1e-9).sum(), 1)
        typ = np.abs(row).sum() / nnz
        keep = np.abs(row - held) < band * typ
        out[t] = np.where(keep, held, row)
        held = out[t]
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def agreement_scale(W, A, B):
    sA, sB = np.sign(A.values), np.sign(B.values)
    mult = np.where(sA * sB > 0, 1.0, np.where(sA * sB < 0, 0.5, 0.75))
    return W * mult


def conviction_floor(W, q=0.25):
    absw = W.abs().replace(0.0, np.nan)
    thr = absw.quantile(q, axis=1)
    keep = absw.ge(thr, axis=0)
    g0 = W.abs().sum(axis=1)
    Wk = W.where(keep, 0.0)
    g1 = Wk.abs().sum(axis=1).replace(0.0, np.nan)
    return Wk.mul(g0 / g1, axis=0).fillna(0.0)


def funding_tilt(W, lam=0.2):
    f = altdata.load_funding("funding_sum", reindex_like=close)
    if f is None:
        return W
    fz = f.shift(1).sub(f.shift(1).mean(axis=1), axis=0).div(
        f.shift(1).std(axis=1).replace(0.0, np.nan), axis=0)
    mult = 1.0 + lam * np.tanh(-np.sign(W) * fz.fillna(0.0))
    Wt = W * mult
    g0 = W.abs().sum(axis=1).replace(0.0, np.nan)
    g1 = Wt.abs().sum(axis=1).replace(0.0, np.nan)
    return Wt.mul(g0 / g1, axis=0).fillna(0.0)


def main():
    A, B, C = sleeves()
    W = book_of(A, B, C)
    base_net = net_of(W)
    base = evaluate(base_net, "BASE APEX (7bps)")
    rows = [base]

    def gate(m):
        ok = ((m["PF_d"] > base["PF_d"] or m["WR_d"] > base["WR_d"])
              and m["IS_sharpe"] >= 0.95 * base["IS_sharpe"]
              and m["IS_cagr"] >= 0.95 * base["IS_cagr"])
        m["verdict"] = "ACCEPT" if ok else "reject"
        return m

    # C1 maker costs
    rows.append(gate(evaluate(net_of(W, cost=0.0004), "C1 maker costs 4bps")))
    # C2 deadband
    for band in (0.25, 0.5):
        rows.append(gate(evaluate(net_of(deadband(W, band)),
                                  f"C2 deadband {band}")))
    # C3 longer smoothing
    A2, B2, C2s = sleeves(spans_A=(10, 20, 30), spans_BC=(6, 10, 16))
    rows.append(gate(evaluate(net_of(book_of(A2, B2, C2s)),
                              "C3 smoothing 2x spans")))
    # S1 agreement scaling
    rows.append(gate(evaluate(net_of(agreement_scale(W, A, B)),
                              "S1 A/B agreement scaling")))
    # S2 conviction floor
    for q in (0.2, 0.3):
        rows.append(gate(evaluate(net_of(conviction_floor(W, q)),
                                  f"S2 conviction floor q={q}")))
    # S3 funding tilt
    for lam in (0.1, 0.2):
        rows.append(gate(evaluate(net_of(funding_tilt(W, lam)),
                                  f"S3 funding tilt lam={lam}")))
    # S4 skip dust days
    g = W.abs().sum(axis=1)
    W_skip = W.mul((g > 0.5 * g[g > 0].median()).astype(float), axis=0)
    rows.append(gate(evaluate(net_of(W_skip), "S4 skip dust days")))
    # P1 downside-vol targeting
    rows.append(gate(evaluate(net_of(W, downside_vt=True),
                              "P1 downside-vol targeting")))
    # P2 trailing throttle
    rows.append(gate(evaluate(net_of(portfolio_trailing(W, base_net, 0.25, 0.5)),
                              "P2 trailing throttle 25%/x0.5")))

    df = pd.DataFrame(rows)
    cols = ["lever", "verdict", "PF_d", "WR_d", "PF_w", "WR_w", "PF_m", "WR_m",
            "IS_sharpe", "IS_cagr", "IS_dd"]
    df["verdict"] = df.get("verdict", "").fillna("")
    print(df[cols].round(3).to_string(index=False))
    df.to_csv("../seismo/out/pf_push_results.csv", index=False)
    return df


if __name__ == "__main__":
    main()
