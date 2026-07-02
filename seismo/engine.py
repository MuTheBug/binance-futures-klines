"""SEISMO: liquidation-cascade "seismology" strategy.

Thesis
------
Market-wide liquidation cascades on perp futures behave like earthquakes:
a main shock of forced, indiscriminate selling that is *flow*, not
information. The coins whose 4h crash is least explained by their own
market beta (most negative *residual*) were hit hardest by forced flow and
snap back over the following hours. The strategy is flat almost all the
time and only deploys leveraged long baskets in the minutes after a
detected main shock.

Anti-lookahead rules
--------------------
- every signal at bar t uses data up to and including close(t)
- fills happen at open(t+1)
- the tradable universe at t is computed from dollar volume shifted 1 bar
- rolling stats use >=3-week min_periods so young listings can't pollute

Trigger horizon (4h), z/beta lookback (30d) and the liquidity floor are
fixed design constants, NOT tuned. Only (z_thr, breadth_thr, n_picks,
hold_hours) are tuned, on the in-sample window only, over a coarse grid.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLL = 24 * 30            # 30d lookback for z-score and beta
MINP = 24 * 21            # 3 weeks min history
TRIG_H = 4                # main-shock measurement horizon (hours), fixed
MIN_UNIVERSE = 15         # need a real cross-section for breadth/median
COST_SIDE = 0.0010        # 10 bps per side: taker fee + slippage


def compute_signals(panel):
    C, O, U = panel["C"], panel["O"], panel["universe"]
    R1 = C.pct_change()
    m1 = R1.where(U).median(axis=1)                     # market hourly return
    n_univ = U.sum(axis=1)

    logm1 = np.log1p(m1)
    M4 = logm1.rolling(TRIG_H).sum()                    # 4h market log-return
    z = (M4 - M4.rolling(ROLL, min_periods=MINP).mean()) / \
        M4.rolling(ROLL, min_periods=MINP).std()

    R4 = C.pct_change(TRIG_H)
    breadth_dn = (R4 < 0).where(U).sum(axis=1) / n_univ.replace(0, np.nan)

    # rolling beta of each coin's hourly return vs the market
    Ex = R1.rolling(ROLL, min_periods=MINP).mean()
    Ey = m1.rolling(ROLL, min_periods=MINP).mean()
    Exy = R1.mul(m1, axis=0).rolling(ROLL, min_periods=MINP).mean()
    vary = m1.rolling(ROLL, min_periods=MINP).var()
    beta = (Exy - Ex.mul(Ey, axis=0)).div(vary, axis=0).clip(0.0, 3.0)

    # residual: how much worse the coin fell than its beta explains
    resid = np.log1p(R4).sub(beta.mul(M4, axis=0))

    # rolling beta of the alt-market factor on BTC (for the tradable hedge)
    rbtc = R1["BTC"]
    cov_mb = (m1 * rbtc).rolling(ROLL, min_periods=MINP).mean() - \
        m1.rolling(ROLL, min_periods=MINP).mean() * \
        rbtc.rolling(ROLL, min_periods=MINP).mean()
    beta_mkt_btc = (cov_mb / rbtc.rolling(ROLL, min_periods=MINP).var()).clip(0.0, 3.0)

    Of = O.ffill()
    RO = Of.pct_change().fillna(0.0)                    # open-to-open marks

    vol1h = R1.rolling(72, min_periods=48).std()        # per-coin hourly vol

    return {
        "C": C, "O": O, "Of": Of, "RO": RO, "U": U,
        "z": z, "breadth_dn": breadth_dn, "n_univ": n_univ,
        "resid": resid, "M4": M4, "m1": m1, "vol1h": vol1h,
        # numpy caches for the hot backtest loop
        "np": {
            "RO": RO.values, "Of": Of.values, "Ono": O.notna().values,
            "U": U.values, "resid": resid.values, "vol": vol1h.values,
            "m1": m1.values, "cols": np.asarray(C.columns),
            "beta": beta.values, "beta_mkt_btc": beta_mkt_btc.values,
            "i_btc": int(np.flatnonzero(np.asarray(C.columns) == "BTC")[0]),
        },
    }


@dataclass
class Params:
    z_thr: float = 2.5        # main-shock severity (z of 4h market return)
    breadth_thr: float = 0.85  # fraction of universe down over 4h
    n_picks: int = 5           # overshooters bought per event
    hold_hours: int = 24       # holding period
    stop_frac: float | None = 0.08  # hard basket stop (unlevered), None = off
    wait_calm: int = 0         # 0 = enter next open; else wait up to this many
                               # hours for the first non-falling market bar
    aftershock_gap: float | None = 24.0   # only trade triggers whose previous
    aftershock_win: float | None = 96.0   # trigger is (gap, win] hours old;
                                          # None disables the aftershock gate
    leverage: float = 3.0      # max gross notional / equity while deployed
    target_risk: float | None = None  # if set: per-event 1-sigma equity risk
                                      # budget; leverage is scaled down when
                                      # predicted basket vol is high
    cost_side: float = COST_SIDE
    side: str = "long_overshoot"   # or "hedged_overshoot" / "short_resilient"
                                   # / "long_market"
    gate_window: int | None = None  # anomaly heartbeat: trade live only when
                                    # the mean levered return of the last
                                    # gate_window PAPER events is positive
    random_seed: int | None = None  # if set: random picks (ablation)


def detect_events(sig, p: Params, start=None, end=None):
    """Tradable trigger bars (overlap is resolved in run()).

    The aftershock gate looks at the age of the previous raw trigger, which
    only uses past information, so it is computed over the full history and
    the [start, end) restriction is applied afterwards.
    """
    z, b, n = sig["z"], sig["breadth_dn"], sig["n_univ"]
    mask = (z <= -p.z_thr) & (b >= p.breadth_thr) & (n >= MIN_UNIVERSE)
    raw = list(mask.index[mask])
    if p.aftershock_gap is None:
        out = raw
    else:
        out = []
        for k in range(1, len(raw)):
            age = (raw[k] - raw[k - 1]).total_seconds() / 3600.0
            if p.aftershock_gap < age <= p.aftershock_win:
                out.append(raw[k])
    if start is not None:
        s = pd.Timestamp(start, tz="UTC")
        out = [t for t in out if t >= s]
    if end is not None:
        e = pd.Timestamp(end, tz="UTC")
        out = [t for t in out if t < e]
    return out


def run(sig, p: Params, start=None, end=None):
    """Backtest. Returns (hourly portfolio return series, trades df, events df)."""
    idx = sig["z"].index
    n_bars = len(idx)
    c = sig["np"]
    ROv, Ofv, Ono, Uv = c["RO"], c["Of"], c["Ono"], c["U"]
    residv, volv, m1v, cols = c["resid"], c["vol"], c["m1"], c["cols"]
    events = detect_events(sig, p, start, end)
    ev_pos = idx.get_indexer(pd.DatetimeIndex(events)) if events else []
    rng = np.random.default_rng(p.random_seed) if p.random_seed is not None else None

    port = np.zeros(n_bars)
    trades, ev_rows = [], []
    busy_until = -1
    paper = []   # levered returns of ALL events (heartbeat gate reads this)

    for t, i in zip(events, ev_pos):
        i_sig = i  # bar whose close defines the signal / picks
        if p.wait_calm > 0:
            # aftershock protocol: stand aside until the market prints its
            # first non-falling hourly bar, then deploy at the next open
            calm = None
            for j in range(i + 1, min(i + 1 + p.wait_calm, n_bars)):
                if m1v[j] >= 0:
                    calm = j
                    break
            if calm is None:
                continue
            i_sig = calm
        i_in = i_sig + 1
        if i_in + 1 >= n_bars or i_in <= busy_until:
            continue
        i_out = min(i_in + p.hold_hours, n_bars - 1)

        # candidates: in universe at trigger, residual defined at signal bar,
        # and a real (non-stale) open print on the entry bar
        r = np.where(Uv[i] & Ono[i_in], residv[i_sig], np.nan)
        cand = np.flatnonzero(np.isfinite(r))
        if len(cand) < p.n_picks:
            continue
        order = cand[np.argsort(r[cand], kind="stable")]
        hedged = p.side == "hedged_overshoot"
        if p.side == "long_overshoot" or hedged:
            picks, sgn = order[:p.n_picks], 1.0
        elif p.side == "short_resilient":
            picks, sgn = order[-p.n_picks:], -1.0
        elif p.side == "long_market":
            picks, sgn = np.flatnonzero(cols == "BTC"), 1.0
        else:
            raise ValueError(p.side)
        if rng is not None:
            picks = rng.choice(cand, size=p.n_picks, replace=False)

        # hedge ratio: mean pick beta (to the alt-market factor) chained
        # with the factor's beta to BTC -> short hb units of BTC per unit long
        hb = 0.0
        if hedged:
            if not (Ono[i_in, c["i_btc"]] and np.isfinite(Ofv[i_out, c["i_btc"]])):
                continue   # hedge instrument not tradable -> no trade
            hb = float(np.nan_to_num(
                np.nanmean(c["beta"][i_sig, picks]) * c["beta_mkt_btc"][i_sig],
                nan=1.0))
            hb = min(max(hb, 0.0), 3.0)

        # hourly mark-to-market: open(t+1) -> open(t+1+H)
        legs = ROv[i_in + 1: i_out + 1][:, picks]
        btc = ROv[i_in + 1: i_out + 1][:, c["i_btc"]]
        if p.stop_frac is not None and len(legs):
            # basket stop: breach observed on an hourly mark, exit next open
            cum = np.cumprod(1.0 + sgn * legs.mean(axis=1) - hb * btc) - 1.0
            breach = np.flatnonzero(cum <= -p.stop_frac)
            if len(breach):
                i_out = min(i_in + 1 + breach[0] + 1, n_bars - 1)
                legs = ROv[i_in + 1: i_out + 1][:, picks]
                btc = ROv[i_in + 1: i_out + 1][:, c["i_btc"]]

        lev = p.leverage
        if p.target_risk is not None:
            pv = np.nanmean(volv[i_sig, picks])
            if np.isfinite(pv) and pv > 0:
                pred = pv * np.sqrt(p.hold_hours)   # predicted basket move
                lev = min(p.leverage, p.target_risk / pred)

        # anomaly heartbeat: deploy real capital only while the trailing
        # paper track of this very strategy is profitable
        live = True
        if p.gate_window is not None:
            live = len(paper) >= p.gate_window and \
                float(np.mean(paper[-p.gate_window:])) > 0

        w = lev / len(picks)
        if live:
            port[i_in + 1: i_out + 1] += sgn * w * legs.sum(axis=1) - lev * hb * btc
            cost_mult = lev * (1.0 + hb) * p.cost_side   # both legs pay costs
            port[i_in] -= cost_mult                       # entry cost
            port[i_out] -= cost_mult                      # exit cost
        busy_until = i_out

        e_px, x_px = Ofv[i_in, picks], Ofv[i_out, picks]
        leg_rets = sgn * (x_px / e_px - 1.0) - 2 * p.cost_side
        for s, lr in zip(picks, leg_rets):
            trades.append({"event": t, "symbol": cols[s], "ret_net": lr,
                           "entry": idx[i_in], "exit": idx[i_out], "live": live})
        ev_ret = float(leg_rets.mean())
        if hedged:
            b_e, b_x = Ofv[i_in, c["i_btc"]], Ofv[i_out, c["i_btc"]]
            hedge_ret = -hb * (b_x / b_e - 1.0) - 2 * p.cost_side * hb
            trades.append({"event": t, "symbol": "HEDGE_BTC",
                           "ret_net": hedge_ret,
                           "entry": idx[i_in], "exit": idx[i_out], "live": live})
            ev_ret += hedge_ret
        ev_rows.append({"event": t, "n": len(picks), "lev": lev, "hedge": hb,
                        "basket_ret_net": ev_ret,
                        "levered_ret": ev_ret * lev, "live": live})
        paper.append(ev_ret * lev)

    return pd.Series(port, index=idx), pd.DataFrame(trades), pd.DataFrame(ev_rows)


def metrics(port, trades, events, label=""):
    eq = (1.0 + port).cumprod()
    yrs = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] - 1.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1.0
    dd = (eq / eq.cummax() - 1.0).min()
    live = port != 0
    sharpe = np.nan
    if port.std() > 0:
        sharpe = port.mean() / port.std() * np.sqrt(24 * 365)
    paper_n = len(events)
    if len(events) and "live" in events.columns:
        events = events[events["live"]]
    if len(trades) and "live" in trades.columns:
        trades = trades[trades["live"]]
    out = {
        "label": label,
        "events": len(events),
        "paper_events": paper_n,
        "legs": len(trades),
        "total_return": total,
        "cagr": cagr,
        "max_dd": dd,
        "sharpe": sharpe,
        "exposure": live.mean(),
    }
    if len(trades):
        w = trades["ret_net"]
        out["leg_winrate"] = (w > 0).mean()
        gains, losses = w[w > 0].sum(), -w[w <= 0].sum()
        out["leg_pf"] = gains / losses if losses > 0 else np.inf
    if len(events):
        b = events["levered_ret"]
        out["event_winrate"] = (b > 0).mean()
        g, l = b[b > 0].sum(), -b[b <= 0].sum()
        out["event_pf"] = g / l if l > 0 else np.inf
        out["avg_event"] = b.mean()
        out["worst_event"] = b.min()
        out["best_event"] = b.max()
    return out


def fmt(m):
    def pc(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:,.1f}%"
    return (f"{m['label']:<28} ev={m['events']:>4} totRet={pc(m['total_return']):>10} "
            f"cagr={pc(m['cagr']):>8} dd={pc(m['max_dd']):>7} shp={m.get('sharpe', float('nan')):>5.2f} "
            f"evWR={pc(m.get('event_winrate')):>6} evPF={m.get('event_pf', float('nan')):>5.2f} "
            f"legWR={pc(m.get('leg_winrate')):>6} legPF={m.get('leg_pf', float('nan')):>5.2f} "
            f"avgEv={pc(m.get('avg_event')):>6} worst={pc(m.get('worst_event')):>7}")
