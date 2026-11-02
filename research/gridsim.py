"""Geometric grid-trading simulator on hourly OHLC, vectorized across symbols.

Grid model (equal quote size Q per level, spacing g, n levels each side of the centre P_c):
    levels L_k = P_c * (1+g)^k,  k = -n..n
    State is the index r of the last filled level (start r = 0). Resting orders are buys at
    every level below r and sells at every level above it, so on a continuous path:
        price falls to p  -> r = max(min(r, ceil(x(p) + tau)), -n)     (buys fill)
        price rises to p  -> r = min(max(r, floor(x(p) - tau)), n)     (sells fill)
    with x(p) = log(p / P_c) / log(1+g) and tau = trade-through requirement in level units.
    A buy at L_k costs Q and buys Q/L_k units; those units are sold one level up for Q(1+g).
    Inventory is a function of r only:  inv(r) = U(r) - c,
        U(r) = (Q/P_c) * (a^r - a^n) / (1 - a),  a = 1/(1+g),
        c = 0 (long grid) | U(0) (neutral) | U(-n) (short grid).
Intrabar path: previous close -> O -> L -> H -> C on up bars, O -> H -> L -> C on down bars
("heuristic"; `path="reverse"` flips it). This is the minimum-oscillation path, so round trips
inside an hour are UNDER-counted for fine grids (conservative on grid income).

Costs: maker fee on grid fills, taker fee + slippage on market trades (start/reset/stop),
funding charged hourly on inventory value (longs pay when funding_annual > 0).
Equity is marked to the close every hour, in units of the starting capital (1.0).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


@dataclass
class GridParams:
    g: float = 0.01                 # level spacing (used when no per-bar g frame is given)
    n: int = 10                     # levels on each side of the centre
    bias: str = "neutral"           # long | neutral | short
    exposure: float = 1.0           # max |inventory notional| / capital  (= leverage)
    maker_fee: float = 0.0002       # Binance USD-M VIP0 maker
    taker_fee: float = 0.0005
    slippage: float = 0.0005        # market orders on alts
    through: float = 0.0            # limit fills require trading this fraction through the level
    funding_annual: float = 0.11    # default 0.01%/8h: longs pay, shorts receive
    on_exit: str = "hold"           # price beyond range by exit_buffer levels: hold | reset | stop
    exit_buffer: int = 1
    cooldown: int = 0               # hours flat after a stop before a new grid may start
    period: int = 0                 # forced re-centre every `period` hours (0 = never)
    compound: bool = True           # size each new grid from current equity (else capital 1.0)
    path: str = "heuristic"         # heuristic | reverse
    # A grid that loses its whole allocated capital K is liquidated (flattened, loss capped at K)
    # and stays flat until its next scheduled re-centre (period) or forever (period = 0).


def _u(r, Q, pc, a, n):
    return (Q / pc) * (a ** r - a ** n) / (1.0 - a)


def _cbias(bias, Q, pc, a, n):
    if bias == "long":
        return np.zeros_like(Q)
    if bias == "neutral":
        return _u(0, Q, pc, a, n)
    return _u(-n, Q, pc, a, n)


def _q_for(bias, capital, a, n, exposure):
    """Per-level quote size so the largest possible |inventory notional| = exposure*capital."""
    if bias == "long":
        k = (1 - a ** (2 * n)) / (1 - a)
    elif bias == "neutral":
        k = (a ** -n - 1) / (1 - a)
    else:
        k = (a ** (-2 * n) - 1) / (1 - a)
    return exposure * capital / k


def simulate(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, p: GridParams,
             g_frame: pd.DataFrame | None = None, allow: pd.DataFrame | None = None,
             bias_frame: pd.DataFrame | None = None) -> dict:
    """Run one grid per symbol (equal capital 1.0 each). Returns equity (T x S) and counters.

    g_frame    : optional spacing per bar/symbol, read when a grid (re)starts (NaN = can't start)
    allow      : optional bool frame; False at the close of bar t -> flatten and stay flat
    bias_frame : optional per-bar bias code (+1 long, 0 neutral, -1 short) read at (re)start
    """
    O, H, L, C = (x.to_numpy(np.float64) for x in (o, h, l, c))
    T, S = C.shape
    G = None if g_frame is None else g_frame.reindex_like(c).to_numpy(np.float64)
    A = None if allow is None else allow.reindex_like(c).fillna(False).to_numpy(bool)
    B = None if bias_frame is None else bias_frame.reindex_like(c).to_numpy(np.float64)
    n = p.n
    fee_m, cost_t = p.maker_fee, p.taker_fee + p.slippage
    f_h = p.funding_annual / HOURS_PER_YEAR

    active = np.zeros(S, bool)
    r = np.zeros(S)
    pc = np.ones(S)
    g = np.full(S, p.g)
    a = 1 / (1 + g)
    Q = np.zeros(S)
    cb = np.zeros(S)
    bias = np.zeros(S)          # +1 long, 0 neutral, -1 short (numeric for vector ops)
    cash = np.ones(S)
    last = np.full(S, np.nan)   # last processed price
    started = np.zeros(S, np.int64)
    e0 = np.ones(S)             # equity at the last (re)start
    K = np.ones(S)              # capital the current grid was sized from
    kap = np.full((T, S), np.nan)
    liqs = np.zeros(S)
    cool = np.zeros(S, np.int64)
    dead = np.zeros(S, bool)
    eq = np.full((T, S), np.nan)
    act = np.zeros((T, S), bool)
    fills = np.zeros(S)
    fees = np.zeros(S)
    funding = np.zeros(S)
    resets = np.zeros(S)
    stops = np.zeros(S)
    bias_code = {"long": 1.0, "neutral": 0.0, "short": -1.0}[p.bias]

    def inv_of(rr):
        out = np.zeros(S)
        m = active
        if m.any():
            out[m] = _u(rr[m], Q[m], pc[m], a[m], n) - cb[m]
        return out

    def cbias_vec(m):
        res = np.zeros(m.sum())
        bm = bias[m]
        for code, name in ((1.0, "long"), (0.0, "neutral"), (-1.0, "short")):
            k = bm == code
            if k.any():
                res[k] = _cbias(name, Q[m][k], pc[m][k], a[m][k], n)
        return res

    def qvec(m, capital):
        res = np.zeros(m.sum())
        bm = bias[m]
        for code, name in ((1.0, "long"), (0.0, "neutral"), (-1.0, "short")):
            k = bm == code
            if k.any():
                res[k] = _q_for(name, capital[k], a[m][k], n, p.exposure)
        return res

    def transition(pnew):
        nonlocal r
        m = active & np.isfinite(pnew) & np.isfinite(last)
        if not m.any():
            return
        x = np.log(pnew[m] / pc[m]) / np.log1p(g[m])
        tau = -np.log1p(-p.through) / np.log1p(g[m]) if p.through > 0 else 0.0
        rm = r[m]
        down = pnew[m] < last[m]
        up = pnew[m] > last[m]
        rd = np.maximum(np.minimum(rm, np.ceil(x + tau - 1e-9)), -n)
        ru = np.minimum(np.maximum(rm, np.floor(x - tau + 1e-9)), n)
        rn = np.where(down, rd, np.where(up, ru, rm))
        d = rn - rm
        buys = np.clip(-d, 0, None)
        sells = np.clip(d, 0, None)
        qm, gm = Q[m], g[m]
        cash[m] += -qm * buys * (1 + fee_m) + qm * (1 + gm) * sells * (1 - fee_m)
        fees[m] += qm * buys * fee_m + qm * (1 + gm) * sells * fee_m
        fills[m] += buys + sells
        r[m] = rn
        last[m] = pnew[m]

    def market_to(target_inv, price, m):
        """Trade inventory of symbols in mask m to target at price (taker + slippage)."""
        cur = inv_of(r)[m]
        dq = target_inv - cur
        cash[m] -= dq * price[m] + np.abs(dq) * price[m] * cost_t
        fees[m] += np.abs(dq) * price[m] * cost_t

    def start_grid(m, price, t):
        """(Re)centre grids for mask m at price, trading inventory to the new starting level."""
        nonlocal r
        if not m.any():
            return
        cur = inv_of(r)[m] if active[m].any() else np.zeros(m.sum())
        cur = np.where(active[m], cur, 0.0)
        equity = cash[m] + cur * price[m]
        gg = g[m] if G is None else G[t, m]
        ok = np.isfinite(gg) & (gg > 0) & ((equity > 0) | (not p.compound))
        idx = np.flatnonzero(m)[ok]
        if len(idx) == 0:
            return
        mm = np.zeros(S, bool); mm[idx] = True
        g[mm] = gg[ok]
        a[mm] = 1 / (1 + g[mm])
        if B is not None:
            bias[mm] = np.nan_to_num(B[t, mm], nan=bias_code)
        else:
            bias[mm] = bias_code
        pc[mm] = price[mm]
        capital = equity[ok] if p.compound else np.ones(ok.sum())
        e0[mm] = equity[ok]
        K[mm] = capital
        Q[mm] = qvec(mm, capital)
        cb[mm] = cbias_vec(mm)
        target = _u(0, Q[mm], pc[mm], a[mm], n) - cb[mm]
        curv = cur[ok]
        dq = target - curv
        cash[mm] -= dq * price[mm] + np.abs(dq) * price[mm] * cost_t
        fees[mm] += np.abs(dq) * price[mm] * cost_t
        r[mm] = 0.0
        active[mm] = True
        started[mm] = t
        last[mm] = price[mm]

    def flatten(m, price):
        if not m.any():
            return
        market_to(np.zeros(m.sum()), price, m)
        active[m] = False

    order_heur = (p.path == "heuristic")
    for t in range(T):
        ct = C[t]
        valid = np.isfinite(ct)
        if active.any():
            bull = C[t] >= O[t]
            first = np.where(bull == order_heur, L[t], H[t])
            second = np.where(bull == order_heur, H[t], L[t])
            for pt in (O[t], first, second, ct):
                transition(pt)
            m = active & valid
            if m.any() and f_h:
                fund = f_h * inv_of(r)[m] * ct[m]
                cash[m] -= fund
                funding[m] += fund
        # ---- events at the close of bar t (decisions use data <= t)
        inv = inv_of(r)
        equity_now = cash + np.where(active, inv * np.nan_to_num(ct), 0.0)
        blown = active & valid & (equity_now - e0 <= -K)
        if blown.any():                                   # allocation lost: liquidate, cap the loss
            liqs[blown] += 1
            active[blown] = False
            cash[blown] = e0[blown] - K[blown]
            if p.period:
                cool[blown] = started[blown] + p.period
            else:
                dead[blown] = True
            dead |= blown & (cash <= 0) & p.compound
        if A is not None:
            off = active & valid & ~A[t]
            flatten(off, ct)
        if p.on_exit != "hold":
            m = active & valid
            if m.any():
                x = np.full(S, 0.0)
                x[m] = np.log(ct[m] / pc[m]) / np.log1p(g[m])
                out = m & ((x < -n - p.exit_buffer) | (x > n + p.exit_buffer))
                if out.any():
                    if p.on_exit == "reset":
                        resets[out] += 1
                        start_grid(out, ct, t)
                    else:                                  # stop: flatten, cool down
                        stops[out] += 1
                        flatten(out, ct)
                        cool[out] = t + p.cooldown
        if p.period:
            per = active & valid & (t - started >= p.period)
            if per.any():
                resets[per] += 1
                start_grid(per, ct, t)
        can = ~active & ~dead & valid & (t >= cool)
        if A is not None:
            can &= A[t]
        if can.any():
            start_grid(can, ct, t)
        inv = inv_of(r)
        eq[t] = np.where(valid, cash + np.where(active, inv * np.nan_to_num(ct), 0.0), np.nan)
        act[t] = active
        kap[t] = K
    eqdf = pd.DataFrame(eq, index=c.index, columns=c.columns)
    first_valid = c.notna().cummax()
    eqdf = eqdf.where(first_valid).ffill().where(first_valid)
    return {"equity": eqdf, "active": pd.DataFrame(act, index=c.index, columns=c.columns),
            "capital": pd.DataFrame(kap, index=c.index, columns=c.columns), "liquidations": pd.Series(liqs, c.columns),
            "fills": pd.Series(fills, c.columns), "fees": pd.Series(fees, c.columns),
            "funding": pd.Series(funding, c.columns), "resets": pd.Series(resets, c.columns),
            "stops": pd.Series(stops, c.columns), "dead": pd.Series(dead, c.columns)}


# ----------------------------------------------------------------------------- reference
def reference(o, h, l, c, p: GridParams) -> np.ndarray:
    """Slow, explicit-order-book simulator for ONE symbol (1-D arrays). Used only to validate
    `simulate` (bias, fees, funding, resets, through). Supports on_exit hold/reset."""
    n = p.n
    cost_t = p.taker_fee + p.slippage
    f_h = p.funding_annual / HOURS_PER_YEAR
    eqs = np.full(len(c), np.nan)
    state = {"active": False}

    def start(price, cash, inv):
        g = p.g
        a = 1 / (1 + g)
        equity = cash + inv * price
        capital = equity if p.compound else 1.0
        Q = _q_for(p.bias, np.array([capital]), a, n, p.exposure)[0]
        levels = {k: price * (1 + g) ** k for k in range(-n, n + 1)}
        orders = {}
        for k in range(-n, 0):
            orders[k] = ("buy", Q / levels[k])
        for k in range(1, n + 1):
            orders[k] = ("sell", Q / levels[k - 1])
        u0 = sum(Q / levels[k] for k in range(0, n))
        target = {"long": u0, "neutral": 0.0, "short": u0 - sum(Q / levels[k] for k in range(-n, n))}[p.bias]
        dq = target - inv
        cash -= dq * price + abs(dq) * price * cost_t
        state.update(active=True, levels=levels, orders=orders, Q=Q, centre=price, r=0)
        return cash, target

    cash, inv, lastp = 1.0, 0.0, None
    for t in range(len(c)):
        if not np.isfinite(c[t]):
            continue
        if state["active"]:
            pts = [o[t]] + ([l[t], h[t]] if (c[t] >= o[t]) == (p.path == "heuristic") else [h[t], l[t]]) + [c[t]]
            for pt in pts:
                lv, od = state["levels"], state["orders"]
                if pt < lastp:
                    for k in sorted([k for k, v in od.items() if v[0] == "buy"], reverse=True):
                        if lv[k] * (1 - p.through) >= pt - 1e-12 * lv[k]:
                            qty = od.pop(k)[1]
                            cash -= qty * lv[k] * (1 + p.maker_fee); inv += qty
                            od[k + 1] = ("sell", qty)
                        else:
                            break
                elif pt > lastp:
                    for k in sorted([k for k, v in od.items() if v[0] == "sell"]):
                        if lv[k] / (1 - p.through) <= pt + 1e-12 * lv[k]:
                            qty = od.pop(k)[1]
                            cash += qty * lv[k] * (1 - p.maker_fee); inv -= qty
                            od[k - 1] = ("buy", state["Q"] / lv[k - 1])
                        else:
                            break
                lastp = pt
            cash -= f_h * inv * c[t]
        if state["active"] and cash + inv * c[t] <= 0:          # wiped out (same rule as simulate)
            eqs[t:] = 0.0
            return eqs
        if state["active"] and p.on_exit == "reset":
            x = np.log(c[t] / state["centre"]) / np.log1p(p.g)
            if x < -n - p.exit_buffer or x > n + p.exit_buffer:
                cash, inv = start(c[t], cash, inv)
        if not state["active"]:
            cash, inv = start(c[t], cash, inv)
        lastp = c[t]
        eqs[t] = cash + inv * c[t]
    return eqs
