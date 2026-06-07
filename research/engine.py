"""
Vectorized cross-sectional backtest engine — built to be honest.

Conventions (NO look-ahead):
  * Signals/target weights W[t] are computed from information available at the
    CLOSE of bar t.
  * They are applied to the return of the NEXT bar: realized over bar (t+1).
    Implemented as holding W[t-1] over bar t  =>  gross_ret[t] = sum_i W[t-1,i]*R[t,i].
  * Costs are drift-aware: between rebalances, holdings drift with price; turnover
    is |target - drifted_holdings|, charged at the rebalance and felt next bar.
  * Funding is charged on NET exposure (longs pay, shorts receive) — realistic for
    perpetual futures and ~0 for market-neutral books.

Leverage is studied separately (leverage_curve): the strategy emits weights with a
chosen gross; we then scale the net per-bar return by k and check for liquidation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---- cost defaults (per side, as fraction of notional traded) ----
TAKER_FEE = 0.0005      # Binance USDT-M futures taker (VIP0) = 5 bps
SLIPPAGE  = 0.0002      # extra slip/spread for liquid majors = 2 bps
COST_PER_SIDE = TAKER_FEE + SLIPPAGE          # 7 bps per side, 14 bps round trip
FUNDING_ANNUAL = 0.11   # ~0.01%/8h * 3 * 365 ≈ 11%/yr drag on net-long exposure


def bars_per_year(interval: str) -> float:
    return 365.0 if interval == "1d" else 24 * 365.0


def to_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns. NaN where price missing."""
    return close.astype("float64").pct_change()


def simulate(weights: pd.DataFrame, ret: pd.DataFrame, interval: str,
             cost_per_side: float = COST_PER_SIDE,
             funding_annual: float = FUNDING_ANNUAL,
             funding_matrix: pd.DataFrame | None = None) -> dict:
    """
    weights : target weights at close of each bar (cols=symbols). sum|w| = gross.
    ret     : simple returns aligned to same index/cols.
    funding_matrix : optional per-asset realized funding rate per bar (longs pay when >0).
                     If given, charges accurate per-coin funding (income on the side that
                     receives) instead of the generic net-exposure drag.
    Returns dict with net per-bar return series and diagnostics.
    """
    W = weights.reindex_like(ret).fillna(0.0)
    R = ret.fillna(0.0)

    # gross portfolio return over bar t from holding previous targets W[t-1]
    Wp = W.shift(1).fillna(0.0)
    gross = (Wp * R).sum(axis=1)

    # drift-aware turnover: weights drift over bar t, then we rebalance to W[t]
    drifted = (Wp * (1.0 + R)).div(1.0 + gross, axis=0).fillna(0.0)
    turnover = (W - drifted).abs().sum(axis=1)               # one-sided, at close[t]
    cost = (turnover * cost_per_side).shift(1).fillna(0.0)    # felt next bar

    if funding_matrix is not None:
        Fm = funding_matrix.reindex_like(ret).fillna(0.0)
        funding = (Wp * Fm).sum(axis=1)                      # long pays +funding; short receives
    else:
        f_bar = funding_annual / bars_per_year(interval)
        funding = (Wp.sum(axis=1).clip(lower=None) * f_bar)  # net exposure * per-bar funding
    funding = funding.fillna(0.0)

    net = gross - cost - funding
    return {
        "net": net, "gross": gross, "cost": cost, "funding": funding,
        "turnover": turnover, "gross_exposure": W.abs().sum(axis=1),
        "net_exposure": W.sum(axis=1),
    }


def vol_target(weights: pd.DataFrame, ret: pd.DataFrame, interval: str,
               target_vol: float = 0.40, vol_lookback: int | None = None,
               max_leverage: float = 3.0, cost_per_side: float = COST_PER_SIDE,
               funding_annual: float = FUNDING_ANNUAL,
               funding_matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scale a (gross~1) weight matrix so trailing realized portfolio vol ~= target_vol
    (annualized). Scale is computed from the strategy's own past returns and lagged one
    bar => no look-ahead. Capped at max_leverage to avoid blow-ups when vol is tiny."""
    bpy = bars_per_year(interval)
    if vol_lookback is None:
        vol_lookback = int(round(bpy / 12))      # ~1 month trailing window
    base = simulate(weights, ret, interval, cost_per_side, funding_annual, funding_matrix)["net"]
    rv = base.rolling(vol_lookback, min_periods=vol_lookback // 2).std() * np.sqrt(bpy)
    scale = (target_vol / rv).shift(1).clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


# --------------------------- metrics ---------------------------
def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def metrics(net: pd.Series, interval: str, name: str = "") -> dict:
    net = net.dropna()
    if len(net) == 0:
        return {"name": name, "n": 0}
    bpy = bars_per_year(interval)
    eq = (1.0 + net).cumprod()
    n = len(net)
    years = n / bpy
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else -1.0
    mu, sd = net.mean(), net.std()
    sharpe = float(mu / sd * np.sqrt(bpy)) if sd > 0 else 0.0
    downside = net[net < 0].std()
    sortino = float(mu / downside * np.sqrt(bpy)) if downside and downside > 0 else 0.0
    mdd = _max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    wins = net[net > 0]; losses = net[net < 0]
    win_rate = float((net > 0).mean())
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf
    return {
        "name": name, "n": n, "years": round(years, 2),
        "total_return": round(total, 4), "CAGR": round(cagr, 4),
        "ann_vol": round(float(sd * np.sqrt(bpy)), 4),
        "Sharpe": round(sharpe, 3), "Sortino": round(sortino, 3),
        "maxDD": round(mdd, 4), "Calmar": round(calmar, 3),
        "win_rate": round(win_rate, 4), "profit_factor": round(pf, 3),
        "final_equity_x": round(float(eq.iloc[-1]), 3),
    }


def monthly_returns(net: pd.Series) -> pd.Series:
    eq = (1.0 + net.fillna(0.0)).cumprod()
    m = eq.resample("ME").last().pct_change()
    if len(eq) and len(m):                       # seed first month from inception
        first = eq.resample("ME").last().iloc[0] - 1.0
        m.iloc[0] = first
    return m.dropna()


def monthly_summary(net: pd.Series) -> dict:
    m = monthly_returns(net)
    if len(m) == 0:
        return {}
    return {
        "n_months": int(len(m)),
        "mean_monthly": round(float(m.mean()), 4),
        "median_monthly": round(float(m.median()), 4),
        "pct_positive": round(float((m > 0).mean()), 4),
        "best": round(float(m.max()), 4),
        "worst": round(float(m.min()), 4),
        "std_monthly": round(float(m.std()), 4),
    }


# --------------------- leverage / risk of ruin ---------------------
def leverage_curve(net: pd.Series, k: float, maint_margin: float = 0.005) -> dict:
    """
    Scale the (gross=1) net per-bar return by leverage k and compound, modelling
    liquidation: if a single bar loss wipes equity below maintenance, account is
    ruined (equity -> ~0). Returns equity curve stats + ruin flag.
    Note: per-bar (close-to-close); true intrabar liquidation can be worse, so this
    is optimistic on the ruin timing — treat ruin probability as a lower bound.
    """
    r = net.fillna(0.0).to_numpy()
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    ruined = False
    ruin_at = None
    curve = np.empty(len(r))
    liq_threshold = -(1.0 - maint_margin) / k   # per-bar return that wipes the account
    for i, x in enumerate(r):
        if x <= liq_threshold:
            eq = 0.0
            ruined = True
            ruin_at = i
            curve[i:] = 0.0
            break
        eq *= (1.0 + k * x)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
        curve[i] = eq
    return {"k": k, "final_x": eq, "maxDD": mdd, "ruined": ruined, "ruin_bar": ruin_at,
            "curve": pd.Series(curve, index=net.index)}


def fmt_metrics(m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return f"{m.get('name','?'):>22}  (no data)"
    return (f"{m['name']:>24}  yrs={m['years']:>4}  CAGR={m['CAGR']*100:>7.1f}%  "
            f"Shrp={m['Sharpe']:>5.2f}  Sort={m['Sortino']:>5.2f}  "
            f"maxDD={m['maxDD']*100:>6.1f}%  Calmar={m['Calmar']:>5.2f}  "
            f"win={m['win_rate']*100:>4.1f}%  PF={m['profit_factor']:>4.2f}  "
            f"eq={m['final_equity_x']:>7.2f}x")
