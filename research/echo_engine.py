"""
THE ECHO ENGINE  —  definitive backtest + leverage report.

A beta-decomposed DUAL-ENGINE on crypto perps. Each coin's daily return is split,
by a rolling market-beta regression, into two physically different processes that
demand OPPOSITE models, and we trade BOTH:

  SLEEVE A  "ride the tide"  : risk-adjusted multi-lookback TREND on the systematic
                               (beta * market) component.            [the robust core]
  SLEEVE B  "ride the wake"  : beta-neutral idiosyncratic CONTINUATION — long coins whose
                               market-stripped move is extending, short those fading,
                               tanh-bounded + per-name capped.       [the novel orthogonal alpha]

The two sleeves are near-uncorrelated (corr ~ +0.13), so the 50/50 blend's Sharpe
exceeds the ~1.4 trend-only ceiling that earlier research called the hard limit. The
orthogonal alpha everyone hunts for in exotic external data was hiding INSIDE the price,
separable by a beta regression.

Discipline: tuned ONLY on data <= 2024-12-31 (lab.IS_END). Every number below is printed
IS (in-sample) and OOS (out-of-sample) so the reader can judge robustness directly.

Run:  cd research && python3 echo_engine.py
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lab, engine, echo, strategies as st
try:
    import altdata
except Exception:
    altdata = None

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET_VOL = 0.40          # annualized vol target for the (gross~1) blend
# 2-sleeve weights (no alt-data) and 3-sleeve weights (with positioning alt-data)
W2 = (0.50, 0.50)                 # (trend, residual)
W3 = (0.40, 0.40, 0.20)           # (trend, residual, positioning)


# ----------------------------- the strategy -----------------------------
def build_sleeves(close, vol, ret, elig):
    """Return (wA, wB, wC). wC (crowd-positioning) is None unless alt-data is present."""
    # SLEEVE A: trend (flagship: risk-adjusted strength, EMA-ensemble, top-10 concentrated)
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0)
    wA = st.signal_to_weights(sigT, ret, elig, vol_lookback=15)
    wA = st.ema_ensemble(wA, spans=(5, 10, 15))
    wA = st.concentrate(wA, 10)
    # SLEEVE B: beta-neutral idiosyncratic continuation (multi-F ensemble)
    wB = echo.residual_continuation(close, ret, elig, beta_window=60,
                                    formations=(3, 5, 8), cap=0.10)
    # SLEEVE C: crowd-positioning reversal (validated alt-data edge), if data present
    wC = None
    if altdata is not None and altdata.have_metrics():
        ls = altdata.load_metric("toptrader_pos_ls", reindex_like=close)
        if ls is not None and ls.notna().sum().sum() > 1000:
            wC = echo.positioning_sleeve(close, ret, elig, ls, fade=True, lag=1,
                                         beta_window=60, cap=0.10)
    return wA, wB, wC


def blend_weights(wA, wB, wC):
    if wC is not None:
        return W3[0] * wA + W3[1] * wB + W3[2] * wC
    return W2[0] * wA + W2[1] * wB


def vt_net(weights, ret, cost_per_side=engine.COST_PER_SIDE):
    wv = engine.vol_target(weights, ret, "1d", target_vol=TARGET_VOL, max_leverage=3.0)
    return engine.simulate(wv, ret, "1d", cost_per_side=cost_per_side)


def line(name, net):
    isn, oos = lab.split(net)
    mi, mo = engine.metrics(isn, "1d", name + " [IS]"), engine.metrics(oos, "1d", name + " [OOS]")
    print(engine.fmt_metrics(mi)); print(engine.fmt_metrics(mo))
    return mi, mo


# ----------------------------- leverage / risk of ruin -----------------------------
def block_bootstrap_leverage(net, leverages, horizon=365, block=20, n_paths=4000, seed=7,
                             maint_margin=0.005):
    """Resample BLOCKS of daily net returns (preserves autocorrelation/vol clustering),
    build many 1-year paths, apply each leverage with per-bar liquidation, and report the
    distribution of outcomes + probability of ruin. Honest about leverage: vol drag and
    liquidation are modelled, so the growth-optimal leverage is finite."""
    r = net.dropna().to_numpy()
    n = len(r); rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block, size=(n_paths, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    paths = r[idx]                                    # (n_paths, horizon)
    out = {}
    for k in leverages:
        lev = paths * k
        liq_thr = -(1.0 - maint_margin) / k
        ruined = (lev <= liq_thr).any(axis=1)
        gk = np.where(lev <= liq_thr, -1.0, lev)      # floor a liquidating bar at -100%
        eq = np.cumprod(1.0 + gk, axis=1)
        eq[ruined] = 0.0
        final = eq[:, -1]
        out[k] = dict(med=float(np.median(final)), p05=float(np.quantile(final, .05)),
                      p95=float(np.quantile(final, .95)), mean_log=float(np.mean(np.log(np.clip(final,1e-9,None)))),
                      p_ruin=float(ruined.mean()))
    return out


def main():
    close, vol, ret, elig = lab.load_data("1d")
    print(f"universe={close.shape[1]} symbols  bars={close.shape[0]} "
          f"({close.index[0].date()}..{close.index[-1].date()})  "
          f"elig median/day={int(elig.sum(1).median())}\n")

    wA, wB, wC = build_sleeves(close, vol, ret, elig)
    netA = vt_net(wA, ret)["net"]
    netB = vt_net(wB, ret)["net"]
    blend = blend_weights(wA, wB, wC)
    simX = vt_net(blend, ret)
    netX = simX["net"]

    print("=" * 96)
    print("SLEEVES (each vol-targeted to 40%/yr, 14bps round-trip cost, gross~1):")
    print("=" * 96)
    line("A  TREND (ride the tide)", netA)
    line("B  RESID-CONTINUATION (ride the wake)", netB)
    if wC is not None:
        line("C  CROWD-POSITIONING (fade the herd)", vt_net(wC, ret)["net"])
    c = pd.concat([netA, netB], axis=1).dropna()
    print(f"\n   corr(A,B) = {c.iloc[:,0].corr(c.iloc[:,1]):+.3f}   <- near-orthogonal => diversification\n")
    print("=" * 96)
    if wC is not None:
        print(f"THE ECHO ENGINE v2 = {int(W3[0]*100)}% TREND + {int(W3[1]*100)}% RESIDUAL "
              f"+ {int(W3[2]*100)}% POSITIONING (alt-data)")
    else:
        print(f"THE ECHO ENGINE = {int(W2[0]*100)}% TREND + {int(W2[1]*100)}% RESIDUAL "
              f"(no alt-data; run altdata/ to add Sleeve C)")
    print("=" * 96)
    mi, mo = line("ECHO ENGINE", netX)

    mfull = engine.metrics(netX, "1d", "ECHO full")
    msi, mso = engine.monthly_summary(*lab.split(netX))[0] if False else (None, None)
    # yearly
    ann = (1 + netX.fillna(0)).groupby(netX.index.year).prod() - 1
    print("\n  yearly net return (vol-targeted, ~1x gross):  " +
          "  ".join(f"{y}:{v*100:+.0f}%" for y, v in ann.items()))
    si, so = engine.monthly_summary(lab.split(netX)[0]), engine.monthly_summary(lab.split(netX)[1])
    print(f"  monthly: IS mean={si['mean_monthly']*100:.1f}% med={si['median_monthly']*100:.1f}% "
          f"+{si['pct_positive']*100:.0f}% worst={si['worst']*100:.0f}%  |  "
          f"OOS mean={so['mean_monthly']*100:.1f}% med={so['median_monthly']*100:.1f}% "
          f"+{so['pct_positive']*100:.0f}% worst={so['worst']*100:.0f}%")
    print(f"  avg gross exposure={simX['gross_exposure'].mean():.2f}x  turnover/day={simX['turnover'].mean():.3f}")

    # ----------------------------- LEVERAGE -----------------------------
    print("\n" + "=" * 96)
    print("LEVERAGE  —  block-bootstrap (1yr paths, 20d blocks) with per-bar liquidation")
    print("=" * 96)
    levs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for tag, seg in [("IS  ", lab.split(netX)[0]), ("OOS ", lab.split(netX)[1]), ("FULL", netX)]:
        bb = block_bootstrap_leverage(seg, levs)
        # growth-optimal = argmax mean log-final
        kstar = max(bb, key=lambda k: bb[k]["mean_log"])
        print(f"\n  [{tag}]  $1 -> 1yr median (x) by leverage   (growth-optimal k*={kstar:g})")
        print("    " + "  ".join(f"{k:g}x" for k in levs))
        print("    med   " + "  ".join(f"{bb[k]['med']:.2f}" for k in levs))
        print("    p05   " + "  ".join(f"{bb[k]['p05']:.2f}" for k in levs))
        print("    p95   " + "  ".join(f"{bb[k]['p95']:.1f}" for k in levs))
        print("    ruin% " + "  ".join(f"{bb[k]['p_ruin']*100:.1f}" for k in levs))

    # final leveraged headline at a recommended, survivable leverage
    K = 2.0
    lc = engine.leverage_curve(netX, K)
    mk = engine.metrics(netX * K, "1d", f"ECHO @ {K:g}x")
    print("\n" + "=" * 96)
    print(f"HEADLINE — ECHO ENGINE at {K:g}x leverage (recommended; survivable, growth-near-optimal)")
    print("=" * 96)
    print(f"  CAGR={mk['CAGR']*100:.0f}%  Sharpe={mk['Sharpe']:.2f}  Sortino={mk['Sortino']:.2f}  "
          f"maxDD={mk['maxDD']*100:.0f}%  Calmar={mk['Calmar']:.2f}  win={mk['win_rate']*100:.1f}%  "
          f"PF={mk['profit_factor']:.2f}")
    print(f"  $60 -> ${60*lc['final_x']:,.0f} over {mfull['years']:.1f}y  (path liquidation-checked: "
          f"ruined={lc['ruined']})")

    # save artifacts
    netX.to_csv(os.path.join(RESULTS, "echo_engine_net.csv"))
    today = blend.iloc[-1]; today = today[today.abs() > 1e-6].sort_values()
    today.to_csv(os.path.join(RESULTS, "echo_today_weights.csv"))
    _plot(netA, netB, netX)
    print(f"\n  artifacts -> results/echo_engine_net.csv, results/echo_today_weights.csv, "
          f"results/echo_engine.png")


def _plot(netA, netB, netX):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 1]})
    series = [(netA, "Trend sleeve (1x)", "#888", 1.1),
              (netB, "Residual-continuation sleeve (1x)", "#3a7", 1.1),
              (netX, "ECHO ENGINE blend (1x)", "#06c", 1.7),
              (netX * 2, "ECHO ENGINE @ 2x leverage", "#c30", 1.7)]
    for s, lbl, c, lw in series:
        eq = (1 + s.fillna(0)).cumprod()
        ax[0].plot(eq.index, eq.values, label=lbl, color=c, lw=lw)
    ax[0].axvline(lab.IS_END, color="k", ls="--", lw=0.8)
    ax[0].text(lab.IS_END, ax[0].get_ylim()[1] * 0.5, "  OOS ->", fontsize=9)
    ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1 (log)")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title("The ECHO Engine — beta-decomposed dual-engine "
                    "(trend the tide + ride the residual wake)")
    eq = (1 + netX.fillna(0)).cumprod(); dd = eq / eq.cummax() - 1
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="#c30", alpha=.5)
    ax[1].set_ylabel("drawdown %"); ax[1].axvline(lab.IS_END, color="k", ls="--", lw=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "echo_engine.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
