"""
Validation gauntlet for the frozen CBS v2 swing strategy (run AFTER freezing;
nothing here feeds back into parameter choices).

  1. Plateau neighborhood: IS AND OOS metrics for every one-step neighbor of
     the frozen config — reported for transparency (if OOS failure were
     config luck, some neighbor would rescue it; it doesn't).
  2. Cost / funding stress (IS + OOS).
  3. Leverage ladder: the ENGINE re-run with risk scaled k-fold (real
     path-dependent compounding, heat/gross caps scaled, liquidation check).
  4. Block-bootstrap ruin analysis: 1-year paths resampled from the FULL
     4h net series in 1-week blocks, levered k-fold with per-bar liquidation.
  5. Sleeve analysis: correlation + blend with the v3 daily cross-sectional
     trend strategy (research/final_v3.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import swing_signals
from swing_engine import Config, run, metrics, fmt, BPY_4H
from run_swing import windows, FINAL_SIG, FINAL_CFG

RNG = np.random.default_rng(7)


def leverage_ladder(base, sig, i0, i_split, iT):
    print("=" * 100)
    print("LEVERAGE LADDER — engine re-run with per-trade risk x k (caps scaled, gross capped at 4x)")
    print("=" * 100)
    for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        cfg = Config(**{**FINAL_CFG,
                        "risk_frac": FINAL_CFG["risk_frac"] * k,
                        "heat_cap": FINAL_CFG["heat_cap"] * k,
                        "gross_cap": min(FINAL_CFG["gross_cap"] * k, 4.0),
                        "pos_frac_cap": min(0.30 * k, 0.60)})
        oi = run(sig, cfg, start=i0, end=i_split)
        oo = run(sig, cfg, start=i_split, end=iT)
        of = run(sig, cfg, start=i0, end=iT)
        mi, mo, mf = (metrics(o["net"], "") for o in (oi, oo, of))
        ruin = "RUINED" if (oi["ruined"] or oo["ruined"] or of["ruined"]) else ""
        gr = of["gross"].mean()
        print(f"  k={k:>3.1f} avgGross={gr:>4.2f}x | IS CAGR={mi.get('CAGR',0)*100:>6.1f}% "
              f"Shrp={mi.get('Sharpe',0):>5.2f} DD={mi.get('maxDD',0)*100:>6.1f}% | "
              f"OOS CAGR={mo.get('CAGR',0)*100:>6.1f}% DD={mo.get('maxDD',0)*100:>6.1f}% | "
              f"FULL CAGR={mf.get('CAGR',0)*100:>6.1f}% Shrp={mf.get('Sharpe',0):>5.2f} "
              f"DD={mf.get('maxDD',0)*100:>6.1f}% {ruin}")


def bootstrap_ruin(net: pd.Series, block: int = 42, n_paths: int = 3000,
                   maint: float = 0.005):
    """1-year paths (BPY_4H bars) resampled in `block`-bar blocks; lever k;
    per-bar liquidation when k*loss wipes equity to maintenance."""
    r = net.dropna().to_numpy()
    nblocks = int(np.ceil(BPY_4H / block))
    starts = RNG.integers(0, max(1, len(r) - block), size=(n_paths, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)
    paths = r[idx][:, :BPY_4H]
    print("=" * 100)
    print(f"BLOCK-BOOTSTRAP ({n_paths} 1-yr paths, {block}-bar blocks, from FULL 4h net)")
    print("=" * 100)
    for k in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        liq = -(1.0 - maint) / k
        lev = k * paths
        ruined = (lev <= liq).any(axis=1)
        eq = np.where(ruined, 0.0, np.prod(1.0 + np.clip(lev, liq, None), axis=1))
        med = np.median(eq)
        p5, p95 = np.percentile(eq, [5, 95])
        mo = med ** (1 / 12) - 1
        print(f"  k={k:>3.1f} | median 1yr: x{med:>5.2f} ({mo*100:+5.1f}%/mo) | "
              f"5-95%: x{p5:.2f}..x{p95:.2f} | P(ruin)={ruined.mean()*100:4.1f}%")


def neighborhood(base, i0, i_split, iT):
    print("=" * 100)
    print("PLATEAU NEIGHBORHOOD — IS (used for tuning) and OOS (transparency; NOT used to select)")
    print("=" * 100)
    steps = [("frozen", {}, {}),
             ("N=20", dict(N=20), {}), ("N=28", dict(N=28), {}),
             ("thr=0.0", dict(ts_thr=0.0), {}), ("thr=0.2", dict(ts_thr=0.2), {}),
             ("stop=3.0", {}, dict(stop_atr=3.0)), ("stop=4.0", {}, dict(stop_atr=4.0)),
             ("trail=4", {}, dict(trail_atr=4.0)), ("trail=6", {}, dict(trail_atr=6.0)),
             ("retest=12", {}, dict(retest_bars=12)), ("retest=24", {}, dict(retest_bars=24)),
             ("mp=14 rf=.0054", {}, dict(max_positions=14, risk_frac=0.0054)),
             ("mp=26 rf=.0029", {}, dict(max_positions=26, risk_frac=0.0029))]
    cache = {}
    for name, skw, ckw in steps:
        skw_full = {**FINAL_SIG, **skw}
        key = tuple(sorted(skw_full.items()))
        if key not in cache:
            cache[key] = base.build(**skw_full)
        cfg = Config(**{**FINAL_CFG, **ckw})
        mi = metrics(run(cache[key], cfg, start=i0, end=i_split)["net"], "")
        mo = metrics(run(cache[key], cfg, start=i_split, end=iT)["net"], "")
        print(f"  {name:<16} IS: Shrp={mi.get('Sharpe',0):>5.2f} CAGR={mi.get('CAGR',0)*100:>6.1f}% "
              f"DD={mi.get('maxDD',0)*100:>6.1f}% | OOS: Shrp={mo.get('Sharpe',0):>5.2f} "
              f"CAGR={mo.get('CAGR',0)*100:>6.1f}% DD={mo.get('maxDD',0)*100:>6.1f}%")


def stress(base, sig, i0, i_split, iT):
    print("=" * 100)
    print("COST / FUNDING STRESS (frozen config)")
    print("=" * 100)
    for name, ckw in [("base", {}),
                      ("costs x1.5", dict(cost_per_side=0.00105, maker_fee=0.0003, stop_slip=0.00075)),
                      ("costs x2", dict(cost_per_side=0.0014, maker_fee=0.0004, stop_slip=0.001)),
                      ("all-taker entries", dict(maker_fee=0.0007)),
                      ("funding x2", dict(funding_8h=2e-4)),
                      ("dd throttle k=2", dict(dd_k=2.0))]:
        cfg = Config(**{**FINAL_CFG, **ckw})
        mi = metrics(run(sig, cfg, start=i0, end=i_split)["net"], "")
        mo = metrics(run(sig, cfg, start=i_split, end=iT)["net"], "")
        print(f"  {name:<18} IS: Shrp={mi.get('Sharpe',0):>5.2f} CAGR={mi.get('CAGR',0)*100:>6.1f}% "
              f"DD={mi.get('maxDD',0)*100:>6.1f}% | OOS: Shrp={mo.get('Sharpe',0):>5.2f} "
              f"CAGR={mo.get('CAGR',0)*100:>6.1f}% DD={mo.get('maxDD',0)*100:>6.1f}%")


def sleeves(base, sig, i0, iT):
    print("=" * 100)
    print("SLEEVE ANALYSIS — CBS (this) + v3 daily L/S trend (final_v3.py), generic funding")
    print("=" * 100)
    import lab, engine, strategies as st
    from final_v3 import build, EXCLUDE, I, TV
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(EXCLUDE)
    v3 = engine.simulate(engine.vol_target(build(c, v, e), r, I, target_vol=TV), r, I)["net"]
    out = run(sig, Config(**FINAL_CFG), start=i0, end=iT)
    cbs = out["equity"].resample("1D").last().dropna().pct_change().dropna()
    common = v3.index.intersection(cbs.index)
    v3c, cbsc = v3[common], cbs[common]
    split = pd.Timestamp("2024-12-31", tz="UTC")
    print(f"  daily corr: FULL={v3c.corr(cbsc):+.3f} "
          f"IS={v3c[common<=split].corr(cbsc[common<=split]):+.3f} "
          f"OOS={v3c[common>split].corr(cbsc[common>split]):+.3f}")
    for wt in (1.0, 0.7, 0.5, 0.0):
        b = wt * v3c + (1 - wt) * cbsc
        mi = engine.metrics(b[common <= split], I, "")
        mo = engine.metrics(b[common > split], I, "")
        print(f"  v3={wt:>3.0%} CBS={1-wt:>3.0%} | IS: Shrp={mi['Sharpe']:>5.2f} "
              f"CAGR={mi['CAGR']*100:>6.1f}% DD={mi['maxDD']*100:>6.1f}% | "
              f"OOS: Shrp={mo['Sharpe']:>5.2f} CAGR={mo['CAGR']*100:>6.1f}% "
              f"DD={mo['maxDD']*100:>6.1f}%")


if __name__ == "__main__":
    print("loading ...")
    base = swing_signals.Base()
    sig = base.build(**FINAL_SIG)
    i0, i_split, iT = windows(base)

    neighborhood(base, i0, i_split, iT)
    stress(base, sig, i0, i_split, iT)
    leverage_ladder(base, sig, i0, i_split, iT)
    full = run(sig, Config(**FINAL_CFG), start=i0, end=iT)
    bootstrap_ruin(full["net"])
    sleeves(base, sig, i0, iT)
