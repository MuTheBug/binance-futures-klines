"""
The honest frontier within reach: mine the OHLC *candle anatomy* — a latent proxy for
liquidation / absorption / order-flow that close-to-close returns throw away.

Signals (all from bar t's OHLC, known at close t; engine lags execution 1 bar -> no look-ahead):
  * CLV   close-location-value: where the close sits in the day's range
  * WICK  lower-minus-upper wick (intraday rejection / absorption asymmetry)
  * GAP   overnight gap vs prior close (open relative to prev close)
  * IVOLR intraday range vs |close-to-close| (how much path was traded vs net move)

Each is built market-neutral, beta-neutralized, inverse-vol sized, EMA-smoothed — the SAME
construction as the residual sleeve — then judged IS-first and for orthogonality to A/B.
Accept ONLY if it raises the blend's in-sample Sharpe.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import data, lab, engine, echo, strategies as st

IS_END = lab.IS_END
close, vol, ret, elig = lab.load_data("1d")
opn = data.load("open", "1d").reindex_like(close)
high = data.load("high", "1d").reindex_like(close)
low = data.load("low", "1d").reindex_like(close)

rng = (high - low).replace(0.0, np.nan)
body_hi = np.maximum(opn, close)
body_lo = np.minimum(opn, close)
feat = {
    "CLV  close-loc":   ((close - low) - (high - close)) / rng,        # +1 closed at high
    "WICK absorption":  ((body_lo - low) - (high - body_hi)) / rng,    # +1 long lower wick
    "GAP  overnight":   (opn / close.shift(1) - 1.0),                  # gap vs prev close
    "IVOLR range/move": (rng / (close - opn).abs().replace(0, np.nan)),# path vs net move
}


def micro_sleeve(signal, beta_window=60, smooth=(3, 5, 8), cap=0.10, fade=False):
    s = signal.where(elig)
    z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
    sig = (-np.tanh(z) if fade else np.tanh(z)).where(elig)
    sig = sig.sub(sig.mean(axis=1), axis=0).where(elig)
    # beta-neutralize against the market (so it's not a hidden directional bet)
    beta = echo.rolling_beta(ret, echo.market_return(ret, elig), beta_window).where(elig, 0.0)
    w = (sig * (1.0 / ret.rolling(30, min_periods=15).std().replace(0, np.nan))).where(elig, 0.0)
    num = (w * beta).sum(axis=1); den = (beta * beta).sum(axis=1).replace(0.0, np.nan)
    w = w.sub(beta.mul(num / den, axis=0), fill_value=0.0).where(elig, 0.0)
    w = st.ema_ensemble(w, smooth).clip(-cap, cap)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def net_of(w):
    return engine.simulate(engine.vol_target(w, ret, "1d", target_vol=0.40, max_leverage=3.0), ret, "1d")["net"]


def shp(net):
    i, o = lab.split(net); return engine.metrics(i, "1d")["Sharpe"], engine.metrics(o, "1d")["Sharpe"]


# existing sleeves for orthogonality + blend baseline
sigT = st.strength_signal(close, lookbacks=(15, 30, 60, 90), strength_vol=30, k=2.0)
A = st.concentrate(st.ema_ensemble(st.signal_to_weights(sigT, ret, elig, 15), (5, 10, 15)), 10)
B = echo.residual_continuation(close, ret, elig, beta_window=60, formations=(3, 5, 8), cap=0.10)
nA, nB = net_of(A), net_of(B)
base = net_of(0.5 * A + 0.5 * B); base_is = engine.metrics(lab.split(base)[0], "1d")["Sharpe"]

print("=" * 90)
print(f"baseline blend (A+B 50/50): IS Sharpe {base_is:.2f}")
print("=" * 90)
print("CANDLE-ANATOMY SLEEVES — pick best direction by IS, then orthogonality + blend test\n")
print(f"{'signal':20} {'dir':5} {'IS':>5} {'OOS':>6} | {'corrA':>6} {'corrB':>6} | "
      f"{'blend+C IS':>10} {'OOS':>6} {'verdict':>8}")
keep = {}
for name, f in feat.items():
    best = None
    for fade in (False, True):
        w = micro_sleeve(f, fade=fade)
        nn = net_of(w); i, o = shp(nn)
        if best is None or i > best[0]:
            best = (i, o, fade, w, nn)
    i, o, fade, w, nn = best
    cdf = pd.concat([lab.split(nA)[0], lab.split(nB)[0], lab.split(nn)[0]], axis=1).dropna()
    cA = cdf.iloc[:, 0].corr(cdf.iloc[:, 2]); cB = cdf.iloc[:, 1].corr(cdf.iloc[:, 2])
    three = net_of((A + B + w) / 3.0)
    ti, to = shp(three)
    verdict = "ACCEPT" if ti > base_is + 0.02 else "reject"
    if verdict == "ACCEPT":
        keep[name] = w
    print(f"{name:20} {'fade' if fade else 'ride':5} {i:5.2f} {o:6.2f} | {cA:6.2f} {cB:6.2f} | "
          f"{ti:10.2f} {to:6.2f} {verdict:>8}")

# if any accepted, test the full multi-sleeve blend
if keep:
    print("\n" + "=" * 90)
    sleeves = [A, B] + list(keep.values())
    combo = sum(sleeves) / len(sleeves)
    ci, co = shp(net_of(combo))
    print(f"FULL blend A+B+{'+'.join(k.split()[0] for k in keep)} (equal): IS {ci:.2f} | OOS {co:.2f}  "
          f"(baseline {base_is:.2f})")
else:
    print("\nNo candle-anatomy sleeve raised in-sample Sharpe -> honestly reject (no free lunch).")
