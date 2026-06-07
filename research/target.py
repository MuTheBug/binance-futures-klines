"""
Two questions, answered honestly:
  (1) How does the strategy degrade when CONCENTRATED to K positions (runnable on $60)?
  (2) What leverage is needed to chase 40%/month, and what is the probability of RUIN?

Leverage/ruin uses block-bootstrap of the strategy's own daily returns (blocks
preserve autocorrelation / volatility clustering) with per-bar liquidation modelling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import data, engine, strategies as st
import lab

INTERVAL = "1d"
LOOKBACKS = (15, 30, 60, 90)
TARGET_VOL = 0.40
MAINT = 0.005   # 0.5% maintenance margin (optimistic; real intrabar can liquidate sooner)


def flagship_net(elig, close, vol, ret, k=None, target_vol=TARGET_VOL, max_lev=3.0):
    w = st.ts_momentum_multi(close, vol, lookbacks=LOOKBACKS, vol_lookback=LOOKBACKS[0], elig=elig)
    if k is not None:
        w = st.concentrate(w, k)
    w = engine.vol_target(w, ret, INTERVAL, target_vol=target_vol, max_leverage=max_lev)
    return engine.simulate(w, ret, INTERVAL)["net"], w


def block_bootstrap(daily: np.ndarray, k: float, n_paths=4000, horizon=360, block=15,
                    maint=MAINT, seed=0):
    """Return dict of distribution stats for leverage k over `horizon` days."""
    n = len(daily)
    liq = -(1.0 - maint) / k
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(horizon / block))
    fin = np.empty(n_paths)
    ruined = np.zeros(n_paths, dtype=bool)
    mdd = np.empty(n_paths)
    for p in range(n_paths):
        starts = rng.integers(0, n - block, nblocks)
        path = np.concatenate([daily[s:s + block] for s in starts])[:horizon]
        eq = 1.0; peak = 1.0; dd = 0.0; rin = False
        for x in path:
            if x <= liq:
                eq = 0.0; rin = True; break
            eq *= (1.0 + k * x)
            if eq > peak: peak = eq
            d = eq / peak - 1.0
            if d < dd: dd = d
        fin[p] = eq; ruined[p] = rin; mdd[p] = (-1.0 if rin else dd)
    pos = fin[fin > 0]
    monthly = np.where(fin > 0, fin, 1e-9) ** (1.0 / 12.0) - 1.0   # geometric monthly equiv
    return {
        "k": k,
        "p_ruin": float(ruined.mean()),
        "median_monthly": float(np.median(monthly)),
        "p_monthly_ge_40": float((monthly >= 0.40).mean()),
        "median_year_x": float(np.median(fin)),
        "p_lose_half": float((fin < 0.5).mean()),
        "p_10x": float((fin >= 10).mean()),
        "median_maxDD": float(np.median(mdd)),
    }


def main():
    close, vol, ret, elig = lab.load_data(INTERVAL)

    print("="*100)
    print("(1) CONCENTRATION: full book vs top-K positions (runnable on a tiny account)")
    print("    K positions => need ~K (L/S) mini-futures open at once. $60 realistically supports K<=~8-10.")
    print("="*100)
    for k in [None, 15, 10, 8, 6, 4, 3]:
        net, w = flagship_net(elig, close, vol, ret, k=k)
        is_net, oos_net = lab.split(net)
        mi = engine.metrics(is_net, INTERVAL); mo = engine.metrics(oos_net, INTERVAL)
        lab_k = "ALL" if k is None else f"top{k}"
        avg_pos = (w.abs() > 1e-6).sum(axis=1)
        print(f"  {lab_k:>5}  IS: CAGR={mi['CAGR']*100:>6.1f}% Sh={mi['Sharpe']:>4.2f} DD={mi['maxDD']*100:>5.0f}% "
              f"| OOS: CAGR={mo['CAGR']*100:>6.1f}% Sh={mo['Sharpe']:>4.2f} DD={mo['maxDD']*100:>5.0f}% "
              f"| avg #pos={avg_pos[avg_pos>0].mean():.1f}")

    # choose a runnable concentration for the leverage study
    K_RUN = 8
    net_run, _ = flagship_net(elig, close, vol, ret, k=K_RUN)
    full = net_run.dropna().to_numpy()
    oos = lab.split(net_run)[1].dropna().to_numpy()
    print(f"\nChosen runnable config for leverage study: top-{K_RUN}, vol-target {TARGET_VOL:.0%}.")
    print(f"  base daily mean={np.mean(full)*100:.3f}% std={np.std(full)*100:.2f}%  "
          f"(OOS mean={np.mean(oos)*100:.3f}% std={np.std(oos)*100:.2f}%)")

    print("\n" + "="*100)
    print("(2) LEVERAGE vs RUIN — block-bootstrap, 1-year (360d) paths, w/ liquidation")
    print("    Two bases: FULL sample (optimistic, includes strong IS years) and OOS-ONLY (realistic).")
    print("="*100)
    for label, base in [("FULL", full), ("OOS ", oos)]:
        print(f"\n  --- base = {label} daily returns ---")
        print(f"  {'lev':>4} {'med.monthly':>11} {'P(mo>=40%)':>11} {'med.year':>9} "
              f"{'P(ruin/yr)':>11} {'P(lose½)':>9} {'P(>=10x)':>9} {'med.maxDD':>10}")
        for k in [1, 2, 3, 5, 8, 12, 20, 30]:
            r = block_bootstrap(base, k)
            print(f"  {k:>4} {r['median_monthly']*100:>10.1f}% {r['p_monthly_ge_40']*100:>10.1f}% "
                  f"{r['median_year_x']:>8.2f}x {r['p_ruin']*100:>10.1f}% {r['p_lose_half']*100:>8.1f}% "
                  f"{r['p_10x']*100:>8.1f}% {r['median_maxDD']*100:>9.0f}%")

    # what leverage targets 40%/month on each base, and its ruin prob?
    print("\n" + "="*100)
    print("(3) Leverage required for MEDIAN 40%/month, and the ruin it implies")
    print("="*100)
    for label, base in [("FULL", full), ("OOS ", oos)]:
        best = None
        for k in np.arange(1, 60, 0.5):
            r = block_bootstrap(base, k, n_paths=1500)
            if r["median_monthly"] >= 0.40:
                best = r; break
        if best:
            print(f"  base={label}: need lev≈{best['k']:.1f}x for median 40%/mo  =>  "
                  f"P(ruin within 1yr)={best['p_ruin']*100:.0f}%, median maxDD={best['median_maxDD']*100:.0f}%, "
                  f"P(lose half)={best['p_lose_half']*100:.0f}%")
        else:
            print(f"  base={label}: 40%/mo median NOT reachable within 1–60x before ruin dominates.")


if __name__ == "__main__":
    main()
