"""
Live signal computation for the ECHO Engine + RIFT combined book.

Fetches daily klines for the tradeable USDT-perp universe from Binance, builds aligned
close/volume matrices, and runs the SAME validated research code (strategies.py, engine.py,
echo.py, rift.py) used in the backtests — so the bot trades exactly what was tested.
Returns the current target weights (fraction of equity per symbol, sign = long/short).

The book = blend of two strategies (corr ~ +0.4..0.5 -- they share alt-momentum
exposure, so diversification is modest):
  ECHO v2 (40% trend + 40% residual continuation + 20% crowd-positioning fade)
  RIFT    (weekly cross-sectional momentum L/S, validated on the ...-dxl3r5 branch)
Each is vol-targeted to cfg.target_vol on its own trailing net returns, then combined
cfg.blend_echo / (1 - cfg.blend_echo). Backtest of this exact construction
(seismo/combined.py): ECHO-weight 0.8-1.0 is a flat Sharpe plateau (~1.98 full-period);
default 0.80 keeps RIFT as a 20% regime-insurance sleeve at zero measured cost.

Only fully CLOSED daily bars are used (the in-progress bar is dropped), so there is no
look-ahead and the live signal matches the backtest convention.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd

# non-crypto names (tokenized stocks / metals / FX) excluded from the crypto universe
EXCLUDE = {"NVDA", "TSLA", "MSTR", "SPY", "QQQ", "INTC", "MU", "MRVL", "AMD", "SNDK",
           "SKHYNIX", "NOK", "EWY", "SOXL", "CRCL", "SPCX", "XAG", "XAU", "XAUT", "PAXG",
           "CL", "BZ", "SLX", "BILL", "COIN", "GENIUS", "CHIP", "BTW", "ZEST"}


def _import_research(research_dir):
    if research_dir not in sys.path:
        sys.path.insert(0, research_dir)
    import strategies, engine, echo, rift  # noqa: E402
    return strategies, engine, echo, rift


def fetch_universe_klines(client, cfg, log=print):
    """Return (close_df, vol_df, base_to_symbol) of CLOSED daily bars across the universe."""
    filt = client.load_filters()                      # symbol -> filters (PERPETUAL USDT)
    base_to_symbol = {}
    for sym, f in filt.items():
        base = f["base"]
        if base in EXCLUDE:
            continue
        base_to_symbol[base] = sym                    # e.g. BTC -> BTCUSDT
    now_ms = int(time.time() * 1000)
    closes, vols = {}, {}
    syms = sorted(base_to_symbol.items())
    for i, (base, sym) in enumerate(syms):
        try:
            kl = client.klines(sym, interval="1d", limit=cfg.klines_limit)
        except Exception as e:
            log(f"  klines fail {sym}: {e}")
            continue
        rows = [(int(k[0]), float(k[4]), float(k[5])) for k in kl if int(k[6]) < now_ms]
        if len(rows) < cfg.min_history_days:
            continue
        idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
        closes[base] = pd.Series([r[1] for r in rows], index=idx)
        vols[base] = pd.Series([r[2] for r in rows], index=idx)
        if (i + 1) % 40 == 0:
            log(f"  fetched {i+1}/{len(syms)} symbols")
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).reindex_like(close)
    return close, vol, base_to_symbol


def _fetch_positioning(client, base_to_symbol, close, log=print):
    """Build a (day x base) top-trader long/short POSITION ratio matrix aligned to `close`.
    Returns None if the endpoint yields nothing (e.g. blocked/unavailable)."""
    cols, n_ok = {}, 0
    for base, sym in base_to_symbol.items():
        if base not in close.columns:
            continue
        rows = client.top_long_short_position_ratio(sym, period="1d", limit=30)
        if rows:
            idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True).floor("D")
            cols[base] = pd.Series([r[1] for r in rows], index=idx)
            n_ok += 1
    if n_ok == 0:
        return None
    ls = pd.DataFrame(cols).sort_index()
    log(f"  positioning: {n_ok} symbols")
    return ls.reindex(close.index).reindex(columns=close.columns)


def compute_targets(client, cfg, log=print):
    """Compute current target weights. Returns (targets, info).
    targets : {binance_symbol: weight}  (weight = signed fraction of equity)
    info    : diagnostics dict
    """
    st, engine, echo, rift = _import_research(cfg.research_dir)
    close, vol, base_to_symbol = fetch_universe_klines(client, cfg, log)
    if close.shape[1] < 5 or close.shape[0] < cfg.min_history_days:
        raise RuntimeError(f"insufficient data: close shape {close.shape}")

    ret = engine.to_returns(close)
    elig = st.eligibility(close, vol, cfg.min_history_days, cfg.min_dollar_vol, liq_lookback=30)

    # SLEEVE A — trend (flagship), SLEEVE B — residual continuation
    sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0)
    wA = st.concentrate(st.ema_ensemble(st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
    wB = echo.residual_continuation(close, ret, elig, beta_window=60, formations=(3, 5, 8), cap=0.10)

    # SLEEVE C — crowd-positioning reversal (validated alt-data edge), if live data available
    ls = _fetch_positioning(client, base_to_symbol, close, log)
    wC, used_C = None, False
    if ls is not None and ls.notna().sum().sum() > 200:
        wC = echo.positioning_sleeve(close, ret, elig, ls, fade=True, lag=1, beta_window=60, cap=0.10)

    if wC is not None:                       # 3-sleeve blend (40/40/20)
        blend = 0.40 * wA + 0.40 * wB + 0.20 * wC
        used_C = True
    else:                                    # fall back to 2-sleeve (e.g. positioning unavailable)
        blend = cfg.blend_trend * wA + (1 - cfg.blend_trend) * wB

    # STRATEGY 2 — RIFT: weekly cross-sectional momentum L/S (uncorrelated book)
    wR = rift.rift_weights(close, ret, elig)

    # equal-risk combination: vol-target each book on its own trailing net
    # returns, then blend 50/50 (validated in seismo/combined.py)
    w_echo = engine.vol_target(blend, ret, "1d", target_vol=cfg.target_vol, max_leverage=3.0)
    w_rift = engine.vol_target(wR, ret, "1d", target_vol=cfg.target_vol, max_leverage=3.0)
    wv = cfg.blend_echo * w_echo + (1.0 - cfg.blend_echo) * w_rift

    last = wv.iloc[-1].dropna()
    last = last[last.abs() > 1e-6] * cfg.leverage      # apply strategy leverage multiplier

    # keep only the strongest N positions (runnable on a small account)
    if len(last) > cfg.max_positions:
        last = last.reindex(last.abs().sort_values(ascending=False).index[:cfg.max_positions])

    gross = float(last.abs().sum())
    if gross > cfg.max_gross:                          # safety cap on total exposure
        last = last * (cfg.max_gross / gross)
        gross = cfg.max_gross

    targets = {base_to_symbol[b]: float(w) for b, w in last.items() if b in base_to_symbol}
    info = {
        "asof": str(close.index[-1].date()),
        "universe": int(close.shape[1]),
        "eligible": int(elig.iloc[-1].sum()),
        "n_targets": len(targets),
        "gross": round(gross, 3),
        "net": round(float(last.sum()), 3),
        "n_long": int((last > 0).sum()),
        "n_short": int((last < 0).sum()),
        "sleeves": ("A+B+C" if used_C else "A+B") + "+RIFT",
    }
    return targets, info
