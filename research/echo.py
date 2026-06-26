"""
The ECHO ENGINE — beta-decomposed dual-horizon overreaction harvesting.

Thesis (the orthogonal alpha was inside the price data all along):
  Each coin's return R_i = alpha_i + beta_i * M + eps_i, where M is the crypto
  "market" (cross-sectional mean return). The two components are PHYSICALLY DIFFERENT
  processes and demand OPPOSITE models:
    * beta_i * M (the systematic/BTC-driven part)  ->  TRENDS  (herding, slow diffusion)
    * eps_i      (the idiosyncratic residual)       ->  OVERREACTS & MEAN-REVERTS
                                                        (retail overshoot on coin-specific
                                                         noise in a 24/7 reflexive market)
  Classic trend on the RAW return contaminates the trend with the mean-reverting residual.
  So we split the book:
    SLEEVE A (existing flagship): trend the return.
    SLEEVE B (this module, the novel orthogonal engine): FADE THE RESIDUAL.
        -> long the coins whose idiosyncratic move has over-extended DOWN (oversold echo)
        -> short the coins whose idiosyncratic move has over-extended UP   (overbought echo)
        market-neutral, beta-neutral, inverse-residual-vol sized.

This is "short-term residual reversal" (Blitz/Hanauer/Vidojevic) reframed for crypto:
fading raw short-term returns is weak (it fights the trending market component); fading the
BETA-NEUTRALIZED residual isolates pure overreaction and is far cleaner. Crypto is the ideal
habitat: retail-dominated, 24/7, reflexive, huge idiosyncratic vol.

No look-ahead: every rolling window ends at bar t (known at close t); the engine lags
execution by one bar.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def market_return(ret: pd.DataFrame, elig: pd.DataFrame) -> pd.Series:
    """The crypto 'market' factor: equal-weight mean return of eligible coins each bar.
    (Equal-weight cross-sectional mean is the cleanest market proxy for residualization;
    it is not dominated by a single mega-cap and is what the residual is defined against.)"""
    R = ret.where(elig)
    return R.mean(axis=1)


def rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int) -> pd.DataFrame:
    """Rolling market beta per coin over a trailing `window` ending at bar t.
    beta = Cov(R_i, M)/Var(M).  All windows end at t -> no look-ahead."""
    M = mkt.reindex(ret.index)
    Mb = pd.DataFrame(np.repeat(M.to_numpy()[:, None], ret.shape[1], axis=1),
                      index=ret.index, columns=ret.columns).where(ret.notna())
    mp = max(window // 2, 10)
    mean_R = ret.rolling(window, min_periods=mp).mean()
    mean_M = Mb.rolling(window, min_periods=mp).mean()
    mean_RM = (ret * Mb).rolling(window, min_periods=mp).mean()
    cov = mean_RM - mean_R * mean_M
    var_M = (Mb * Mb).rolling(window, min_periods=mp).mean() - mean_M * mean_M
    return cov / var_M.replace(0.0, np.nan)


def residuals(ret: pd.DataFrame, elig: pd.DataFrame, *, beta_window: int = 60) -> pd.DataFrame:
    """Idiosyncratic residual eps_i,t = R_i,t - beta_i,t * M_t  (alpha absorbed by the
    cross-sectional demeaning done later). beta from a trailing window ending at t."""
    mkt = market_return(ret, elig)
    beta = rolling_beta(ret, mkt, beta_window)
    M = mkt.reindex(ret.index)
    eps = ret.sub(beta.mul(M, axis=0))
    return eps.where(elig)


def _ema_ensemble(weights: pd.DataFrame, spans=(3, 5, 8)) -> pd.DataFrame:
    """Smooth target weights with an ensemble of EMA spans (cuts turnover/whipsaw)."""
    return sum(weights.ewm(span=s).mean() for s in spans) / len(spans)


def residual_continuation(close: pd.DataFrame, ret: pd.DataFrame, elig: pd.DataFrame, *,
                          beta_window: int = 60, formations=(3, 5, 8), skip: int = 0,
                          smooth_spans=(3, 5, 8), cap: float = 0.10,
                          gross: float = 1.0) -> pd.DataFrame:
    """SLEEVE B — beta-neutral idiosyncratic-momentum ("ride the residual's wake").

    The data verdict (see ECHO_ENGINE.md): at the daily horizon the idiosyncratic
    residual does NOT mean-revert in tradeable dollar terms — its large moves CONTINUE
    (real coin-specific breakouts/news), so we RIDE the residual instead of fading it.

      signal_i,t = mean_F tanh( z[ cumulative residual eps over last F bars ] )
                   (long coins whose market-stripped move is extending up; short those
                    extending down) -> cross-sectionally demeaned (dollar-neutral),
                    EMA-ensemble smoothed, and per-name CAPPED.

    The tanh-bounded, capped sizing is essential: a linearly-sized residual book is
    dominated by a few violent outliers and behaves erratically; bounding harvests the
    broad cross-section (breadth) while the cap stops any single name from dominating.
    Near-orthogonal to the trend sleeve (corr ~ +0.13) -> genuine diversification.
    """
    eps = residuals(ret, elig, beta_window=beta_window)
    sig = 0.0
    for F in formations:
        cum = eps.shift(skip).rolling(F, min_periods=F).sum().where(elig)
        z = cum.sub(cum.mean(axis=1), axis=0).div(cum.std(axis=1).replace(0.0, np.nan), axis=0)
        sig = sig + np.tanh(z)
    sig = (sig / len(formations)).where(elig)
    sig = sig.sub(sig.mean(axis=1), axis=0).where(elig)           # dollar-neutral
    w = sig.div(sig.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    w = _ema_ensemble(w, smooth_spans).clip(-cap, cap)
    # renormalize to target gross after capping
    s = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(s, axis=0).mul(gross).fillna(0.0)
