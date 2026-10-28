"""v4 search: hunt for ORTHOGONAL sleeves to blend with the v3 trend book.

Pre-committed rules (written BEFORE running anything):
  * Tune only on IS (<= 2024-12-31). OOS (2025-01 -> 2026-06) is looked at once per idea.
  * A sleeve PASSES only if: IS Sharpe >= 0.8 AND OOS Sharpe >= 0.5 (after 14bps RT costs and
    pessimistic funding), AND the corr to v3 is < 0.3.
  * A blend is ACCEPTED only if it beats v3-alone on BOTH IS and OOS Sharpe.
  * Funding is not available (Binance API blocked from this box), so shorts in new / pumped
    coins are charged a pessimistic 50%/yr (short PAYS), majors get the generic 11%/yr on net long.

Hypotheses (each one sentence, primitive in brackets):
  H1 listing-decay [event-driven, short]: freshly listed perps drift down over their first
     weeks because of airdrop/unlock supply and low-float/high-FDV launches; short them,
     hedged with BTC. Counterparty: hype buyers and airdrop farmers selling to them.
  H2 pump-fade [mean reversion]: coins with an extreme 1-day idiosyncratic pump revert.
  H3 betting-against-beta [low-risk anomaly]: high-beta alts underperform per unit risk.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import data, engine, strategies as st, lab
from final_v3 import EXCLUDE, build as build_v3

I = "1d"
SHORT_FUNDING_STRESS = 0.50 / 365      # short pays 50%/yr in the listing/pump sleeves
GENERIC = engine.FUNDING_ANNUAL / 365


def load_all():
    c = data.load("close", I); v = data.load("volume", I)
    c = c[c.index >= "2020-05-31"]; v = v[v.index >= "2020-05-31"]
    c = c.drop(columns=[x for x in c.columns if x in EXCLUDE]); v = v[c.columns]
    r = engine.to_returns(c)
    return c, v, r


def report(name, net):
    a, b = lab.split(net)
    mi, mo = engine.metrics(a, I, name + " IS"), engine.metrics(b, I, name + " OOS")
    print(engine.fmt_metrics(mi)); print(engine.fmt_metrics(mo))
    return mi, mo


def run(w, r, F=None, tv=0.40):
    wt = engine.vol_target(w, r, I, target_vol=tv, funding_matrix=F) if tv else w
    return engine.simulate(wt, r, I, funding_matrix=F)


# ---------------------------------------------------------------- H1 listing decay
def listing_age(c):
    first = c.apply(lambda s: s.first_valid_index())
    age = pd.DataFrame(np.arange(len(c))[:, None] - c.index.get_indexer(first)[None, :],
                       index=c.index, columns=c.columns).astype(float)
    age = age.where(c.notna())
    # coins already trading when the dataset starts are not "new listings"
    old = first <= pd.Timestamp("2020-06-15", tz="UTC")
    age.loc[:, old.values] = np.nan
    return age


def h1_listing(c, v, r, start=2, end=60, min_dvol=5e6, hedge=True, cap=0.25):
    age = listing_age(c)
    dvol = (c * v).rolling(3, min_periods=1).median()
    live = (age >= start) & (age <= end) & (dvol >= min_dvol)
    vol = r.rolling(10, min_periods=2).std().fillna(r.abs().rolling(2, min_periods=1).mean())
    raw = (-1.0 / vol.replace(0, np.nan)).where(live, 0.0).fillna(0.0)
    s = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(s, axis=0).fillna(0.0).clip(lower=-cap)        # per-name cap
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if hedge:                                                   # beta hedge w/ BTC (assume beta~1)
        w["BTC"] = w["BTC"] - w.sum(axis=1)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    F = pd.DataFrame(GENERIC, index=c.index, columns=c.columns)
    F = F.mask(live, -SHORT_FUNDING_STRESS)                     # shorts pay on new listings
    return w, F, live


# ---------------------------------------------------------------- H2 pump fade
def h2_pump(c, v, r, thr=0.25, hold=5, e=None):
    idio = r.sub(r["BTC"], axis=0)
    trig = (idio > thr) & e
    held = trig.astype(float).rolling(hold, min_periods=1).max().astype(bool) & e
    vol = r.rolling(20, min_periods=5).std()
    raw = (-1.0 / vol).where(held, 0.0).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w["BTC"] = w["BTC"] - w.sum(axis=1)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    F = pd.DataFrame(GENERIC, index=c.index, columns=c.columns).mask(held, -SHORT_FUNDING_STRESS)
    return w, F


# ---------------------------------------------------------------- H3 betting against beta
def h3_bab(c, v, r, e, lb=60, q=0.3):
    m = r["BTC"]
    beta = r.rolling(lb, min_periods=lb // 2).cov(m).div(m.rolling(lb, min_periods=lb // 2).var(), axis=0)
    beta = beta.where(e)
    rk = beta.rank(axis=1, pct=True)
    lo, hi = rk <= q, rk >= 1 - q
    bl = beta.where(lo).mean(axis=1); bh = beta.where(hi).mean(axis=1)
    wl = lo.astype(float).div(lo.sum(axis=1).replace(0, np.nan), axis=0) / bl.values[:, None]
    wh = hi.astype(float).div(hi.sum(axis=1).replace(0, np.nan), axis=0) / bh.values[:, None]
    w = (wl - wh).fillna(0.0)                                   # beta-neutral L/S
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def main(which):
    c, v, r = load_all()
    e = st.eligibility(c, v, 60, 5e6, 30)
    wv3 = build_v3(c, v, e)
    v3 = engine.simulate(engine.vol_target(wv3, r, I, target_vol=0.40), r, I)["net"]
    print("baseline (generic funding):"); report("v3", v3); print()

    if "h1" in which:
        for start, end, hedge in [(2, 30, True), (2, 60, True), (2, 90, True), (2, 60, False), (7, 60, True)]:
            w, F, live = h1_listing(c, v, r, start, end, hedge=hedge)
            net = run(w, r, F)["net"]
            print(f"H1 listing short d{start}-{end} hedge={hedge}  "
                  f"avg #names IS={live[live.index<=lab.IS_END].sum(1).mean():.1f} OOS={live[live.index>lab.IS_END].sum(1).mean():.1f}")
            report("H1", net)
            print(f"   corr to v3: {net.corr(v3):+.2f}\n")
    if "h2" in which:
        for thr, hold in [(0.2, 3), (0.25, 5), (0.3, 10)]:
            w, F = h2_pump(c, v, r, thr, hold, e)
            net = run(w, r, F)["net"]
            print(f"H2 pump-fade thr={thr} hold={hold}"); report("H2", net)
            print(f"   corr to v3: {net.corr(v3):+.2f}\n")
    if "h3" in which:
        for lb in (30, 60, 120):
            net = run(h3_bab(c, v, r, e, lb), r)["net"]
            print(f"H3 BAB lb={lb}"); report("H3", net)
            print(f"   corr to v3: {net.corr(v3):+.2f}\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["h1", "h2", "h3"])
