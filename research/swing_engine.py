"""
Event-driven portfolio backtester for leveraged swing trading on 4h bars.

Honest conventions (NO look-ahead):
  * Entry signals are computed at the CLOSE of bar t-1 and filled at the OPEN
    of bar t (plus fee+slippage).
  * Stops are monitored INTRABAR with gap-aware fills: a long stop fills at
    min(open, stop) — you never get a better fill than the open — with extra
    stop slippage on top of the normal cost.
  * Trailing stops are updated at bar close and only take effect on the NEXT
    bar.
  * Sizing uses equity as of the PREVIOUS close (known at decision time).
  * Funding is charged at every 8h boundary on open notional: longs pay,
    shorts receive NOTHING (conservative — historically shorts collect, but
    that income is crowded/regime-dependent).
  * Liquidation: if equity falls below maint_frac * gross notional the account
    is wiped to ~0 and the simulation stops. Positions marked to close each
    bar; a same-bar wick that would have liquidated intrabar is approximated
    by checking equity at the worst intrabar mark as well.

The engine takes precomputed signal matrices (numpy, T x N) so parameter
sweeps only pay the (fast) portfolio loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

BPY_4H = 365 * 6                      # 4h bars per year


@dataclass
class Config:
    risk_frac: float = 0.005          # equity fraction risked per trade (1R)
    stop_atr: float = 2.5             # initial stop distance in ATRs
    trail_atr: float = 3.0            # chandelier trail distance in ATRs
    tmax_bars: int = 42               # time stop (4h bars); 42 = 7 days
    cooldown: int = 6                 # bars to wait after an exit before re-entry
    max_positions: int = 10
    heat_cap: float = 0.05            # max total open risk (sum of per-trade R at entry)
    gross_cap: float = 2.0            # max gross notional / equity
    pos_frac_cap: float = 0.30        # max single-position notional / equity
    cost_per_side: float = 0.0007     # taker 5bp + slippage 2bp (market orders)
    maker_fee: float = 0.0002         # resting limit fills (retest entries)
    stop_slip: float = 0.0005         # extra slippage on stop fills
    funding_8h: float = 1e-4          # longs pay 1bp per 8h (~11%/yr)
    maint_frac: float = 0.005         # maintenance margin for liquidation check
    allow_long: bool = True
    allow_short: bool = True
    # --- execution style ---
    entry_mode: str = "open"          # "open": market at next open; "retest": limit at broken level
    retest_bars: int = 12             # how long a retest limit order stays working
    # --- stop style ---
    stop_mode: str = "intrabar"       # "intrabar" | "close" (observe close, exit next open;
    disaster_atr: float = 2.0         #   intrabar disaster stop this many ATRs beyond close-stop)
    # --- portfolio drawdown throttle: scale new-trade risk by
    #     max(dd_floor, 1 - dd_k * current_drawdown). dd_k=0 disables. ---
    dd_k: float = 0.0
    dd_floor: float = 0.25
    max_entries_bar: int = 999        # cap on new positions opened in one bar


@dataclass
class Signals:
    """All matrices are T x N numpy arrays aligned to `index` / `symbols`."""
    index: pd.DatetimeIndex
    symbols: list
    O: np.ndarray; H: np.ndarray; L: np.ndarray; C: np.ndarray
    entry_long: np.ndarray            # bool, computed at close of that bar
    entry_short: np.ndarray
    regime_long: np.ndarray           # bool: long positions still valid at that close
    regime_short: np.ndarray
    conviction: np.ndarray            # float, for ranking simultaneous candidates
    atr: np.ndarray                   # 4h ATR at that close
    level_long: np.ndarray = field(default=None)   # broken level (retest limit px), longs
    level_short: np.ndarray = field(default=None)  # broken level, shorts
    last_bar: np.ndarray = field(default=None)  # per-symbol idx of last valid close


def run(sig: Signals, cfg: Config, start: int = 0, end: int | None = None) -> dict:
    O, H, L, C = sig.O, sig.H, sig.L, sig.C
    T, N = C.shape
    end = T if end is None else end
    C_ffill = pd.DataFrame(C).ffill().to_numpy()   # marking price for gap bars

    if sig.last_bar is None:
        valid = ~np.isnan(C)
        last_bar = np.where(valid.any(axis=0), T - 1 - np.argmax(valid[::-1], axis=0), -1)
    else:
        last_bar = sig.last_bar

    hour8 = (sig.index.hour % 8 == 0)              # funding boundary at bar OPEN
    close_stops = (cfg.stop_mode == "close")

    cash = 1.0
    # open positions: sym_idx ->
    #   [qty(+L/-S), entry_px, stop, extreme_close, bars_held, risk_amt, disaster_stop]
    # In "intrabar" mode `stop` is monitored intrabar and `disaster_stop` is unused.
    # In "close" mode `stop` is observed at the close (exit next open) and
    # `disaster_stop` (wider) is monitored intrabar.
    pos: dict[int, list] = {}
    pending: dict[int, list] = {}                  # j -> [level, atr_sig, is_long, conv, expire_t]
    cooldown_until = np.zeros(N, dtype=np.int64)
    force_exit = set()                             # positions flagged at prev close

    equity = np.full(T, np.nan)
    gross_hist = np.full(T, 0.0)
    npos_hist = np.zeros(T, dtype=np.int64)
    trades = []                                    # closed-trade log
    prev_eq = 1.0
    peak_eq = 1.0
    ruined = False

    def mark(t):
        v = cash
        for j, p in pos.items():
            v += p[0] * C_ffill[t, j]
        return v

    def close_pos(j, px, t, reason, slip_extra=0.0):
        nonlocal cash
        p = pos.pop(j)
        qty = p[0]
        fill = px * (1 - np.sign(qty) * slip_extra)
        cash += qty * fill
        cash -= abs(qty) * fill * cfg.cost_per_side
        pnl = qty * (fill - p[1]) - abs(qty) * (fill + p[1]) * cfg.cost_per_side
        trades.append({"sym": sig.symbols[j], "side": "L" if qty > 0 else "S",
                       "entry": p[1], "exit": fill, "bars": p[4], "t_exit": t,
                       "pnl_frac": pnl, "r_mult": pnl / max(p[5], 1e-12),
                       "reason": reason})
        cooldown_until[j] = t + cfg.cooldown

    def try_open(t, j, is_long, fill_px, atr_sig, eq_dec, heat, gross, fee, throttle=1.0):
        nonlocal cash
        rf = cfg.risk_frac * throttle
        if len(pos) >= cfg.max_positions or heat + rf > cfg.heat_cap + 1e-12:
            return heat, gross, False
        stop_dist = cfg.stop_atr * atr_sig
        if not (stop_dist > 0) or not (fill_px > 0):
            return heat, gross, False
        risk_amt = rf * eq_dec
        notional = min(risk_amt / stop_dist * fill_px, cfg.pos_frac_cap * eq_dec)
        if gross + notional > cfg.gross_cap * eq_dec + 1e-12:
            notional = cfg.gross_cap * eq_dec - gross
            if notional < 0.01 * eq_dec:
                return heat, gross, False
        qty = notional / fill_px * (1 if is_long else -1)
        stop = fill_px - np.sign(qty) * stop_dist
        disaster = fill_px - np.sign(qty) * stop_dist * (1 + cfg.disaster_atr / cfg.stop_atr)
        cash -= qty * fill_px
        cash -= abs(qty) * fill_px * fee
        pos[j] = [qty, fill_px, stop, fill_px, 0, abs(qty) * stop_dist, disaster]
        return heat + rf, gross + notional, True

    for t in range(max(start, 1), end):
        # ---- funding at bar open (longs pay on notional; shorts receive 0) ----
        if hour8[t] and pos:
            for j, p in pos.items():
                if p[0] > 0:
                    cash -= p[0] * C_ffill[t - 1, j] * cfg.funding_8h

        # ---- forced exits at the open (regime flip / time stop / close-stop / delist) ----
        for j in list(pos.keys()):
            if j in force_exit or t > last_bar[j]:
                px = O[t, j]
                if np.isnan(px):
                    px = C_ffill[t - 1, j]
                close_pos(j, px, t, "forced")
        force_exit.clear()

        # ---- intrabar stop check (gap-aware) ----
        for j in list(pos.keys()):
            p = pos[j]
            if np.isnan(L[t, j]) or np.isnan(H[t, j]):
                continue
            lvl = p[6] if close_stops else p[2]
            if p[0] > 0 and L[t, j] <= lvl:
                close_pos(j, min(O[t, j], lvl), t, "stop", cfg.stop_slip)
            elif p[0] < 0 and H[t, j] >= lvl:
                close_pos(j, max(O[t, j], lvl), t, "stop", cfg.stop_slip)

        # ---- entries (signals from close of t-1) ----
        eq_dec = prev_eq                            # equity known at decision time
        peak_eq = max(peak_eq, eq_dec)
        throttle = 1.0
        if cfg.dd_k > 0:
            dd = 1.0 - eq_dec / peak_eq
            throttle = max(cfg.dd_floor, 1.0 - cfg.dd_k * dd)
        heat = sum(p[5] for p in pos.values()) / max(eq_dec, 1e-12)
        gross = sum(abs(p[0]) * C_ffill[t - 1, j2] for j2, p in pos.items())

        if cfg.entry_mode == "retest":
            # cancel stale/invalid orders, then check fills on bar t
            for j in list(pending.keys()):
                o = pending[j]
                regime_ok = sig.regime_long[t - 1, j] if o[2] else sig.regime_short[t - 1, j]
                if t > o[4] or j in pos or not regime_ok or t > last_bar[j]:
                    pending.pop(j)
            fills = []
            for j, o in pending.items():
                if np.isnan(O[t, j]) or np.isnan(L[t, j]) or np.isnan(H[t, j]):
                    continue
                level, is_long = o[0], o[2]
                if is_long and (O[t, j] <= level or L[t, j] <= level):
                    fills.append((o[3], j, is_long, min(O[t, j], level), o[1]))
                elif (not is_long) and (O[t, j] >= level or H[t, j] >= level):
                    fills.append((o[3], j, is_long, max(O[t, j], level), o[1]))
            fills.sort(reverse=True)
            opened = 0
            for conv, j, is_long, px, atr_sig in fills:
                if opened >= cfg.max_entries_bar:
                    break
                heat, gross, ok = try_open(t, j, is_long, px, atr_sig, eq_dec,
                                           heat, gross, cfg.maker_fee, throttle)
                opened += int(ok)
                pending.pop(j, None)                # order done (filled or no capacity)
            # place new orders from signals at close t-1 (limit at the broken level)
            for j in np.flatnonzero(sig.entry_long[t - 1] | sig.entry_short[t - 1]):
                if j in pos or j in pending or t < cooldown_until[j] or t > last_bar[j]:
                    continue
                is_long = bool(sig.entry_long[t - 1, j])
                if (is_long and not cfg.allow_long) or (not is_long and not cfg.allow_short):
                    continue
                atr_sig = sig.atr[t - 1, j]
                lvl = sig.level_long[t - 1, j] if is_long else sig.level_short[t - 1, j]
                if np.isnan(atr_sig) or np.isnan(lvl):
                    continue
                pending[j] = [lvl, atr_sig, is_long, sig.conviction[t - 1, j],
                              t + cfg.retest_bars]
        else:
            cand = []
            for j in np.flatnonzero(sig.entry_long[t - 1] | sig.entry_short[t - 1]):
                if j in pos or t < cooldown_until[j] or t > last_bar[j]:
                    continue
                if np.isnan(O[t, j]) or np.isnan(sig.atr[t - 1, j]):
                    continue
                is_long = bool(sig.entry_long[t - 1, j])
                if (is_long and not cfg.allow_long) or (not is_long and not cfg.allow_short):
                    continue
                cand.append((sig.conviction[t - 1, j], j, is_long))
            cand.sort(reverse=True)
            opened = 0
            for conv, j, is_long in cand:
                if opened >= cfg.max_entries_bar:
                    break
                heat, gross, ok = try_open(t, j, is_long, O[t, j], sig.atr[t - 1, j],
                                           eq_dec, heat, gross, cfg.cost_per_side, throttle)
                opened += int(ok)

        # ---- close of bar t: trail stops, flag exits, mark equity ----
        for j, p in pos.items():
            c = C[t, j]
            if not np.isnan(c):
                a = sig.atr[t, j]
                if p[0] > 0:
                    p[3] = max(p[3], c)
                    if not np.isnan(a):
                        p[2] = max(p[2], p[3] - cfg.trail_atr * a)
                        p[6] = max(p[6], p[2] - cfg.disaster_atr * a)
                    if close_stops and c <= p[2]:
                        force_exit.add(j)
                    if not sig.regime_long[t, j]:
                        force_exit.add(j)
                else:
                    p[3] = min(p[3], c)
                    if not np.isnan(a):
                        p[2] = min(p[2], p[3] + cfg.trail_atr * a)
                        p[6] = min(p[6], p[2] + cfg.disaster_atr * a)
                    if close_stops and c >= p[2]:
                        force_exit.add(j)
                    if not sig.regime_short[t, j]:
                        force_exit.add(j)
            p[4] += 1
            if p[4] >= cfg.tmax_bars:
                force_exit.add(j)

        eq = mark(t)
        gross_notional = sum(abs(p[0]) * C_ffill[t, j] for j, p in pos.items())
        # intrabar worst-case mark for the liquidation check
        eq_low = cash + sum(p[0] * (L[t, j] if p[0] > 0 and not np.isnan(L[t, j]) else
                                    H[t, j] if p[0] < 0 and not np.isnan(H[t, j]) else
                                    C_ffill[t, j]) for j, p in pos.items())
        if eq <= 0 or eq_low <= cfg.maint_frac * gross_notional:
            equity[t:end] = 0.0
            ruined = True
            break
        equity[t] = eq
        gross_hist[t] = gross_notional / eq
        npos_hist[t] = len(pos)
        prev_eq = eq

    eq_ser = pd.Series(equity[start:end], index=sig.index[start:end]).dropna()
    net = eq_ser.pct_change().fillna(0.0)
    if len(eq_ser):
        net.iloc[0] = eq_ser.iloc[0] - 1.0
    return {
        "equity": eq_ser, "net": net, "ruined": ruined,
        "gross": pd.Series(gross_hist[start:end], index=sig.index[start:end]),
        "npos": pd.Series(npos_hist[start:end], index=sig.index[start:end]),
        "trades": pd.DataFrame(trades),
    }


# --------------------------- metrics ---------------------------
def metrics(net: pd.Series, name: str = "", bpy: float = BPY_4H) -> dict:
    net = net.dropna()
    if len(net) == 0 or (1.0 + net).le(0).any():
        return {"name": name, "n": len(net), "ruined": True}
    eq = (1.0 + net).cumprod()
    years = len(net) / bpy
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    mu, sd = net.mean(), net.std()
    sharpe = float(mu / sd * np.sqrt(bpy)) if sd > 0 else 0.0
    dn = net[net < 0].std()
    sortino = float(mu / dn * np.sqrt(bpy)) if dn and dn > 0 else 0.0
    peak = eq.cummax()
    mdd = float((eq / peak - 1).min())
    return {"name": name, "n": len(net), "years": round(years, 2),
            "CAGR": round(cagr, 4), "ann_vol": round(float(sd * np.sqrt(bpy)), 4),
            "Sharpe": round(sharpe, 3), "Sortino": round(sortino, 3),
            "maxDD": round(mdd, 4),
            "Calmar": round(cagr / abs(mdd), 3) if mdd < 0 else 0.0,
            "final_x": round(float(eq.iloc[-1]), 3)}


def fmt(m: dict) -> str:
    if m.get("ruined"):
        return f"{m.get('name','?'):>34}  RUINED"
    return (f"{m['name']:>34}  yrs={m['years']:>4}  CAGR={m['CAGR']*100:>7.1f}%  "
            f"Shrp={m['Sharpe']:>5.2f}  Sort={m['Sortino']:>5.2f}  "
            f"maxDD={m['maxDD']*100:>6.1f}%  Calmar={m['Calmar']:>5.2f}  eq={m['final_x']:>7.2f}x")


def trade_stats(tr: pd.DataFrame) -> dict:
    if tr is None or len(tr) == 0:
        return {}
    win = (tr["r_mult"] > 0)
    return {"n_trades": len(tr), "win_rate": round(float(win.mean()), 3),
            "avg_R": round(float(tr["r_mult"].mean()), 3),
            "avg_win_R": round(float(tr.loc[win, "r_mult"].mean()), 3) if win.any() else 0,
            "avg_loss_R": round(float(tr.loc[~win, "r_mult"].mean()), 3) if (~win).any() else 0,
            "med_bars": float(tr["bars"].median()),
            "by_reason": tr.groupby("reason")["r_mult"].agg(["count", "mean"]).round(3).to_dict("index")}


def monthly(net: pd.Series) -> pd.Series:
    eq = (1.0 + net.fillna(0.0)).cumprod()
    m = eq.resample("ME").last().pct_change()
    if len(m):
        m.iloc[0] = eq.resample("ME").last().iloc[0] - 1.0
    return m.dropna()
