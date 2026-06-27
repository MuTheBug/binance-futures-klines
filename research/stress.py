"""
STRESS-TEST the ECHO Engine. Tries hard to BREAK it: crises, cost/slippage/lag shocks,
funding spikes, reduced breadth, removing the biggest winners, bot-downtime (missed
rebalances), and worst-case bootstrap paths at leverage. Honest by construction.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lab, engine, echo, strategies as st

IS_END = lab.IS_END


def build_blend(close, vol, ret, elig, blend=0.5):
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0)
    wA = st.concentrate(st.ema_ensemble(st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    wB = echo.residual_continuation(close, ret, elig, beta_window=60, formations=(3, 5, 8), cap=0.10)
    return blend * wA + (1 - blend) * wB


def vt_net(w, ret, cost=engine.COST_PER_SIDE, extra_lag=0, funding_annual=0.0):
    wv = engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0)
    if extra_lag:
        wv = wv.shift(extra_lag)
    sim = engine.simulate(wv, ret, "1d", cost_per_side=cost, funding_annual=funding_annual)
    return sim["net"]


def stats(net, K=1.0):
    n = (net * K).dropna()
    eq = (1 + n).cumprod()
    m = engine.metrics(n, "1d")
    return dict(CAGR=m["CAGR"], Sharpe=m["Sharpe"], maxDD=m["maxDD"], Calmar=m["Calmar"],
                PF=m["profit_factor"], win=m["win_rate"])


def underwater(net, K=1.0):
    eq = (1 + (net * K).fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    # longest stretch below previous peak
    below = dd < -1e-9
    longest = cur = 0
    for b in below:
        cur = cur + 1 if b else 0
        longest = max(longest, cur)
    return dd.min(), longest


def main():
    close, vol, ret, elig = lab.load_data("1d")
    w = build_blend(close, vol, ret, elig)
    net = vt_net(w, ret)

    print("=" * 90)
    print("BASELINE (vol-targeted blend, 14bps RT, ~1x gross)")
    for K in (1.0, 2.0):
        s = stats(net, K); dd, uw = underwater(net, K)
        print(f"  {K:g}x: CAGR {s['CAGR']*100:5.0f}%  Sharpe {s['Sharpe']:.2f}  maxDD {s['maxDD']*100:5.0f}%  "
              f"Calmar {s['Calmar']:.2f}  PF {s['PF']:.2f}  longest-underwater {uw}d")

    # ---------- 1) CRISIS EPISODES ----------
    print("\n" + "=" * 90)
    print("1) CRISIS EPISODES — strategy return during named crypto crashes (1x / 2x)")
    crises = {
        "May-2021 leverage flush": ("2021-05-10", "2021-05-23"),
        "LUNA/UST collapse":       ("2022-05-05", "2022-05-20"),
        "3AC/deleverage Jun-2022": ("2022-06-10", "2022-06-30"),
        "FTX collapse":            ("2022-11-06", "2022-11-21"),
        "Aug-2024 carry unwind":   ("2024-08-01", "2024-08-07"),
        "Feb-2025 selloff":        ("2025-02-24", "2025-03-04"),
    }
    for name, (a, b) in crises.items():
        seg = net[(net.index >= a) & (net.index <= b)]
        if len(seg) == 0:
            continue
        r1 = (1 + seg).prod() - 1
        r2 = (1 + 2 * seg).prod() - 1
        print(f"  {name:26} {a}..{b}: 1x {r1*100:+6.1f}%   2x {r2*100:+6.1f}%")

    # ---------- 2) WORST ROLLING WINDOWS ----------
    print("\n" + "=" * 90)
    print("2) WORST ROLLING WINDOWS (1x / 2x)")
    for win, lbl in [(30, "1mo"), (90, "3mo"), (180, "6mo"), (365, "12mo")]:
        roll1 = (1 + net.fillna(0)).rolling(win).apply(np.prod, raw=True) - 1
        roll2 = (1 + 2 * net.fillna(0)).rolling(win).apply(np.prod, raw=True) - 1
        print(f"  worst {lbl:4}: 1x {roll1.min()*100:+6.0f}%   2x {roll2.min()*100:+6.0f}%")

    # ---------- 3) COST / SLIPPAGE / EXECUTION-LAG SHOCKS ----------
    print("\n" + "=" * 90)
    print("3) COST / SLIPPAGE / LAG / FUNDING SHOCKS  (Sharpe IS | OOS)")
    def shp(nn):
        i, o = lab.split(nn); return engine.metrics(i, "1d")["Sharpe"], engine.metrics(o, "1d")["Sharpe"]
    scenarios = [
        ("base 7bps/side",            dict(cost=0.0007)),
        ("15bps/side",                dict(cost=0.0015)),
        ("30bps/side (thin alts)",    dict(cost=0.0030)),
        ("+1 day execution lag",      dict(cost=0.0010, extra_lag=1)),
        ("+2 day execution lag",      dict(cost=0.0010, extra_lag=2)),
        ("funding drag 30%/yr",       dict(cost=0.0010, funding_annual=0.30)),
        ("worst-case 30bps+1lag+fund", dict(cost=0.0030, extra_lag=1, funding_annual=0.30)),
    ]
    for name, kw in scenarios:
        i, o = shp(vt_net(w, ret, **kw))
        print(f"  {name:30} IS {i:5.2f} | OOS {o:5.2f}")

    # ---------- 4) BREADTH / CONCENTRATION JACKKNIFE ----------
    print("\n" + "=" * 90)
    print("4) BREADTH & CONCENTRATION — does it depend on a few names? (Sharpe IS | OOS, FULL CAGR)")
    def run_universe(mask_cols=None, topN=None):
        e2 = elig.copy()
        if mask_cols:
            for c in mask_cols:
                if c in e2.columns:
                    e2[c] = False
        if topN:
            dollar = (close * vol).rolling(30, min_periods=10).median()
            rank = dollar.rank(axis=1, ascending=False)
            e2 = e2 & (rank <= topN)
        w2 = build_blend(close, vol, ret, e2)
        nn = vt_net(w2, ret)
        i, o = shp(nn); f = engine.metrics(nn, "1d")["CAGR"]
        return i, o, f
    for lbl, kw in [("full universe", {}), ("exclude BTC+ETH", dict(mask_cols=["BTC", "ETH"])),
                    ("top-30 liquid only", dict(topN=30)), ("top-20 liquid only", dict(topN=20))]:
        i, o, f = run_universe(**kw)
        print(f"  {lbl:22} IS {i:5.2f} | OOS {o:5.2f} | FULL CAGR {f*100:4.0f}%")

    # ---------- 5) BOT DOWNTIME — missed rebalances ----------
    print("\n" + "=" * 90)
    print("5) BOT DOWNTIME — rebalance every k days instead of daily (Sharpe IS | OOS)")
    wv = engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0)
    for k in (1, 2, 3, 5):
        wk = wv.copy()
        if k > 1:                          # only update weights every k-th bar (hold in between)
            keep = np.zeros(len(wk), bool); keep[::k] = True
            wk = wk.where(pd.Series(keep, index=wk.index), np.nan).ffill()
        nn = engine.simulate(wk, ret, "1d")["net"]
        i, o = shp(nn)
        print(f"  rebalance every {k} day(s): IS {i:5.2f} | OOS {o:5.2f}")

    # ---------- 6) BOOTSTRAP TAIL / RISK OF RUIN ----------
    print("\n" + "=" * 90)
    print("6) BOOTSTRAP TAIL — worst-case 1-year outcomes (block=20d, 40k paths) w/ liquidation")
    r = net.dropna().to_numpy(); n = len(r); rng = np.random.default_rng(5)
    H, block, P = 365, 20, 40000
    nb = int(np.ceil(H / block)); s0 = rng.integers(0, n - block, size=(P, nb))
    idx = (s0[:, :, None] + np.arange(block)[None, None, :]).reshape(P, -1)[:, :H]
    paths = r[idx]
    for K in (1.0, 2.0, 3.0):
        lev = paths * K; thr = -(1 - 0.005) / K
        ruined = (lev <= thr).any(axis=1)
        gk = np.where(lev <= thr, -1.0, lev)
        eq = np.cumprod(1 + gk, axis=1); final = eq[:, -1].copy(); final[ruined] = 0.0
        dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
        print(f"  {K:g}x: 1yr final  p1 {np.quantile(final,.01):4.2f}x  p5 {np.quantile(final,.05):4.2f}x  "
              f"median {np.median(final):4.2f}x | worst path maxDD {dd.min()*100:.0f}% "
              f"median maxDD {np.median(dd)*100:.0f}% | P(ruin) {ruined.mean()*100:.1f}%")
    print("\nDONE.")


if __name__ == "__main__":
    main()
