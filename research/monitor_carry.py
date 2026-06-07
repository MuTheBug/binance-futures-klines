"""
MONTHLY MONITOR: has the funding-carry edge started working again (crowding unwind)?
Fetches FRESH daily klines + funding from Binance for the liquid crypto universe, rebuilds
the carry & trend sleeves with the tested engine, and reports trailing Sharpe + a verdict:
redeploy carry (trend+0.3*carry) only if carry's trailing edge has turned durably positive.

Self-contained (own data fetch). Falls back to repo cache if network is unavailable.
Run: python3 monitor_carry.py
"""
from __future__ import annotations
import os, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
import engine, strategies as st
import carry as cy
import final_v3 as fv

I = "1d"; TV = 0.40
FAPI = "https://fapi.binance.com/fapi/v1"
# liquid crypto universe (majors + liquid alts); failures are skipped
SYMS = ["BTC","ETH","SOL","XRP","BNB","DOGE","ADA","AVAX","LINK","LTC","BCH","XLM","ATOM",
        "ETC","FIL","APT","ARB","OP","NEAR","INJ","AAVE","UNI","CRV","LDO","FET","TIA",
        "SUI","SEI","GALA","ALGO","HBAR","ICP","DOT","TRX","XMR","ZEC","1000PEPE","1000SHIB",
        "WIF","ORDI","TAO","RENDER","ENA","ONDO","JTO","PENDLE","WLD","CHZ"]
DAYS = 320


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "monitor"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def _fetch_sym(base):
    sym = base + "USDT"
    try:
        kl = _get(f"{FAPI}/klines?symbol={sym}&interval=1d&limit={DAYS}")
        fr = _get(f"{FAPI}/fundingRate?symbol={sym}&limit=1000")
    except Exception:
        return base, None
    close = pd.Series({pd.to_datetime(k[0], unit="ms", utc=True).floor("D"): float(k[4]) for k in kl})
    vol = pd.Series({pd.to_datetime(k[0], unit="ms", utc=True).floor("D"): float(k[5]) for k in kl})
    if fr:
        fdf = pd.DataFrame(fr)
        day = pd.to_datetime(fdf["fundingTime"], unit="ms", utc=True).dt.floor("D")
        fund = fdf.assign(d=day, fr=fdf["fundingRate"].astype(float)).groupby("d")["fr"].sum()
    else:
        fund = pd.Series(dtype=float)
    return base, (close, vol, fund)


def fetch_fresh():
    closes, vols, funds = {}, {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_fetch_sym, b) for b in SYMS]):
            base, res = fut.result()
            if res is None:
                continue
            closes[base], vols[base], funds[base] = res
    if not closes:
        raise RuntimeError("no data fetched (network?)")
    c = pd.DataFrame(closes).sort_index(); v = pd.DataFrame(vols).reindex_like(c)
    F = pd.DataFrame(funds).reindex(c.index).reindex(columns=c.columns)
    return c, v, F


def tsharpe(net, win):
    s = net.dropna().iloc[-win:]
    return float(s.mean() / s.std() * np.sqrt(365)) if len(s) > 20 and s.std() > 0 else float("nan")


def main():
    print(f"=== Carry monitor — {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC ===")
    try:
        c, v, F = fetch_fresh()
        src = "LIVE Binance"
    except Exception as e:
        print(f"  live fetch failed ({e}); falling back to repo cache.")
        import lab, funding
        c, v, r0, e0 = lab.load_data(I)
        c, v = c.iloc[-DAYS:], v.iloc[-DAYS:]
        F = funding.load_daily_funding(c.index).reindex(columns=c.columns)
        src = "repo cache (STALE)"
    r = c.pct_change()
    elig = c.notna() & (c.rolling(90, min_periods=90).count() >= 90) & F.notna()
    cy.FMAT = F
    print(f"  data source: {src}  | symbols={elig.iloc[-1].sum()}  last bar={c.index[-1]:%Y-%m-%d}")

    trend_w = fv.build(c, v, elig)
    carry_w = cy.carry_weights(F, r, elig, N=3, K=12, smooth=10)
    trend = engine.simulate(engine.vol_target(trend_w, r, I, target_vol=TV, funding_matrix=F), r, I, funding_matrix=F)["net"]
    carry = engine.simulate(engine.vol_target(carry_w, r, I, target_vol=TV, funding_matrix=F), r, I, funding_matrix=F)["net"]
    combo = trend + 0.3 * carry

    print("\n  trailing Sharpe (annualized):")
    print(f"    {'window':>8} {'trend':>8} {'carry':>8} {'trend+0.3carry':>16}")
    for w in [60, 90, 120]:
        print(f"    {w:>6}d  {tsharpe(trend,w):>8.2f} {tsharpe(carry,w):>8.2f} {tsharpe(combo,w):>16.2f}")

    s90, s120 = tsharpe(carry, 90), tsharpe(carry, 120)
    redeploy = (s90 > 0.5) and (s120 > 0.0)
    print("\n  >>> VERDICT:")
    if redeploy:
        print(f"    REDEPLOY CARRY ✅  carry trailing 90d Sharpe={s90:.2f} (>0.5) & 120d={s120:.2f} (>0).")
        print(f"    Switch to trend+0.3*carry (see carry.py / factor_timing.py). Combined 90d Sharpe={tsharpe(combo,90):.2f}.")
    else:
        print(f"    KEEP TREND-ONLY ⛔  carry trailing 90d Sharpe={s90:.2f} (need >0.5, 120d>0). Carry still crowded.")
    print("    (Threshold is a heuristic; confirm with a fresh IS/OOS backtest before changing capital.)")


if __name__ == "__main__":
    main()
