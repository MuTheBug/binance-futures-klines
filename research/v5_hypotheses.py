"""Part B of V5_PROTOCOL.md: the eight pre-registered hypotheses, one specification each.

    python3 v5_hypotheses.py price       # B5-B8 (need prices and funding only)
    python3 v5_hypotheses.py positioning # B1-B4 (need the positioning metrics download)
Design window only. Pass bars are in the protocol and are applied mechanically below.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

import v5_lib as L
import vision_panel as vp

MET_START = pd.Timestamp("2020-10-15", tz="UTC")      # metrics begin 2020-09-01; 30-45d warm-up


def verdict_standalone(name, net, v4net, start=L.START):
    a, b = (L.stats(x) for x in L.split(net, start))
    corr = float(net.loc[start:L.IS_END].corr(v4net.loc[start:L.IS_END]))
    ok = a["sharpe"] >= 1.0 and b["sharpe"] >= 0.7 and abs(corr) < 0.3
    print(f"{name:<34} IS {a['sharpe']:5.2f} ({a['cagr']:+7.1%}, DD {a['mdd']:6.1%}) | VAL {b['sharpe']:5.2f} "
          f"({b['cagr']:+7.1%}, DD {b['mdd']:6.1%}) | corr v4 {corr:+.2f} | {'PASS' if ok else 'fail'}")
    return dict(name=name, is_sharpe=a["sharpe"], val_sharpe=b["sharpe"], corr=corr, passed=ok)


def verdict_mod(name, net, base, start=L.START):
    a, b = (L.stats(x) for x in L.split(net, start))
    a0, b0 = (L.stats(x) for x in L.split(base, start))
    ok = (a["sharpe"] - a0["sharpe"] >= 0.15 and b["sharpe"] - b0["sharpe"] >= 0.15
          and b["mdd"] >= b0["mdd"] - 0.10)
    print(f"{name:<34} IS {a['sharpe']:5.2f} vs base {a0['sharpe']:5.2f} | VAL {b['sharpe']:5.2f} vs base {b0['sharpe']:5.2f} "
          f"| VAL DD {b['mdd']:6.1%} vs {b0['mdd']:6.1%} | {'PASS' if ok else 'fail'}")
    return dict(name=name, is_sharpe=a["sharpe"], val_sharpe=b["sharpe"], is_base=a0["sharpe"], val_base=b0["sharpe"], passed=ok)


# ------------------------------------------------------------------------------------ B5-B8
def b5_hook(F):
    f3 = F.rolling(3, min_periods=2).mean() * 365.0           # trailing 3-day funding, annualised
    def hook(sig):
        return sig.mask((sig > 0) & (f3 > 0.50), 0.0).mask((sig < 0) & (f3 < -0.50), 0.0)
    return hook


def b6_small_caps():
    p1 = L.load_panel(min_qv=1e6)
    liq = p1["qv"].rolling(30, min_periods=15).median()
    cost = pd.DataFrame(np.where(liq < 5e6, 0.0020, L.COST), index=liq.index, columns=liq.columns)
    return L.v4(p1, cost=cost)[0], p1


def b7_listing_short(p, start=2, end=30, cap=0.25):
    c, r, qv, F = p["c"], p["r"], p["qv"], p["F"]
    first = c.apply(lambda s: s.first_valid_index())
    pos = pd.Series(c.index.get_indexer(first.fillna(c.index[0])), index=c.columns)
    age = pd.DataFrame(np.arange(len(c))[:, None] - pos.to_numpy()[None, :], index=c.index, columns=c.columns).astype(float)
    age = age.where(c.notna())
    not_new = (first <= pd.Timestamp("2020-02-01", tz="UTC")) | first.isna()
    age.loc[:, not_new.to_numpy()] = np.nan
    by_name = np.array([vp.is_crypto_name(x) and ".S" not in x for x in c.columns])
    live = (age >= start) & (age <= end) & (qv.rolling(3, min_periods=1).median() >= 5e6) & by_name[None, :]
    vol = r.rolling(10, min_periods=2).std().fillna(r.abs().rolling(2, min_periods=1).mean())
    raw = (-1.0 / vol.replace(0, np.nan)).where(live, 0.0).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).clip(lower=-cap)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w["BTC"] = w["BTC"] - w.sum(axis=1)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    net, _ = L.run(w, r, L.COST, F)
    return net, live


def b8_taker(p):
    tbs = p["tb"] / p["qv"] - 0.5
    f = tbs.rolling(5, min_periods=4).mean()
    z = (f - f.rolling(60, min_periods=40).mean()) / f.rolling(60, min_periods=40).std()
    return L.xs_sleeve(z, p)[0]


# ------------------------------------------------------------------------------------ B1-B4
def positioning(p):
    lsa, ttp, oi = (vp.load(f"m_{k}").reindex_like(p["c"]) for k in ("lsa", "ttp", "oi"))
    cov = (lsa.notna() & p["e"]).loc[MET_START:].sum().sum() / p["e"].loc[MET_START:].sum().sum()
    print(f"positioning data available on {cov:.1%} of eligible coin-days since {MET_START.date()}")
    ff = lambda x: x.ffill(limit=2)
    z_lsa, z_ttp = L.z30(np.log(ff(lsa))), L.z30(np.log(ff(ttp)))
    c, r = p["c"], p["r"]
    sig30 = r.rolling(30, min_periods=15).std()
    m = np.tanh(2.0 * c.pct_change(7, fill_method=None) / (sig30 * np.sqrt(7)))
    g = np.log(ff(oi)).diff(7).where(p["e"])
    n = g.notna().sum(axis=1).replace(0, np.nan)
    g_rank = 2.0 * (g.rank(axis=1, pct=True) - 0.5 / n.to_numpy()[:, None]) - 1.0
    return {"B1 retail crowding (contrarian)": -z_lsa, "B2 top traders (follow)": z_ttp,
            "B3 smart minus retail": z_ttp - z_lsa, "B4 OI-confirmed move": m * g_rank}


if __name__ == "__main__":
    which = sys.argv[1]
    p = L.load_panel()
    v4net, v3net, _ = L.v4(p)
    out = []
    if which == "price":
        print("--- modifications (must beat baseline by >= 0.15 Sharpe on IS and VAL)")
        n5, _ = L.run(L.v3_weights(p["c"], p["e"], b5_hook(p["F"])), p["r"], L.COST, p["F"])
        out.append(verdict_mod("B5 funding veto on v3", n5, v3net))
        n6, p1 = b6_small_caps()
        print(f"   (B6 universe: median eligible {int(p1['e'].sum(1).loc[L.START:].median())} vs {int(p['e'].sum(1).loc[L.START:].median())})")
        out.append(verdict_mod("B6 small-cap extension of v4", n6, v4net))
        print("--- standalone sleeves (IS >= 1.0, VAL >= 0.7, |corr v4| < 0.3)")
        n7, live = b7_listing_short(p)
        print(f"   (B7: avg names shorted IS {live.loc[L.START:L.IS_END].sum(1).mean():.1f}, VAL {live.loc[L.IS_END:].sum(1).mean():.1f})")
        out.append(verdict_standalone("B7 new-listing short", n7, v4net))
        out.append(verdict_standalone("B8 taker-flow persistence", b8_taker(p), v4net))
    else:
        print("--- standalone sleeves (IS >= 1.0, VAL >= 0.7, |corr v4| < 0.3); IS starts", MET_START.date())
        for name, sig in positioning(p).items():
            net, _ = L.xs_sleeve(sig.loc[MET_START:], {**p, "e": p["e"].loc[MET_START:], "r": p["r"].loc[MET_START:],
                                                         "F": p["F"].loc[MET_START:]})
            out.append(verdict_standalone(name, net, v4net, start=MET_START + pd.Timedelta(days=45)))
    pd.DataFrame(out).to_csv(f"results/v5_hypotheses_{which}.csv", index=False)
