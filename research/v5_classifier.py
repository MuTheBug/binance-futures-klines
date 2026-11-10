"""Calibrate the weekend-activity threshold that separates crypto perps from tokenized TradFi
(stocks/ETFs/commodities) in the design window. Uses name labels only; no strategy returns.

Labeled non-crypto: vision_panel.NON_CRYPTO (minus index/stablecoin products).
Labeled crypto: every instrument first listed before 2025 (TradFi perps did not exist then) that
is not in NON_CRYPTO, plus all of them later. Ambiguous names are reported, not used.
"""
import numpy as np
import pandas as pd
import vision_panel as vp

c = vp.load("close")
qv = vp.load("quote_volume")
r = c.pct_change(fill_method=None)
wr = vp.weekend_ratio(r)
first = c.apply(lambda s: s.first_valid_index())
base = pd.Series([x.split(".")[0] for x in c.columns], index=c.columns)
tradfi = base.isin(set(vp._TRADFI.split()))
amb = base.isin(vp.AMBIGUOUS)
crypto_old = (first < pd.Timestamp("2025-01-01", tz="UTC")) & ~base.isin(vp.NON_CRYPTO)
crypto_new = (first >= pd.Timestamp("2025-01-01", tz="UTC")) & ~base.isin(vp.NON_CRYPTO) & ~amb

liq = qv.rolling(30, min_periods=15).median() >= 5e6           # only liquid days matter for the universe
obs = wr.where(liq)
print(f"instruments with weekend-ratio observations on liquid days: TradFi-labeled {int(obs.loc[:, tradfi].notna().any().sum())}, "
      f"crypto(pre-2025) {int(obs.loc[:, crypto_old].notna().any().sum())}, crypto(2025+) {int(obs.loc[:, crypto_new].notna().any().sum())}, "
      f"ambiguous {int(obs.loc[:, amb].notna().any().sum())}")
for name, m in (("TradFi-labeled", tradfi), ("crypto pre-2025", crypto_old), ("crypto 2025+", crypto_new)):
    v = obs.loc[:, m].stack()
    print(f"{name:<16} symbol-days {len(v):>7}  quantiles 1% {v.quantile(.01):.2f}  5% {v.quantile(.05):.2f}  25% {v.quantile(.25):.2f}  "
          f"50% {v.quantile(.5):.2f}  75% {v.quantile(.75):.2f}  95% {v.quantile(.95):.2f}  99% {v.quantile(.99):.2f}")
print("\nthreshold  TradFi symbol-days kept(wrong)  crypto symbol-days dropped(wrong)")
tv, cv = obs.loc[:, tradfi].stack(), obs.loc[:, crypto_old | crypto_new].stack()
for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
    print(f"  {thr:.2f}      {float((tv >= thr).mean()):6.1%}                       {float((cv < thr).mean()):6.2%}")
med = obs.median()
print("\nTradFi-labeled, median ratio per symbol (liquid days):")
print(med[tradfi].dropna().sort_values().round(2).to_string())
print("\nambiguous names, median ratio per symbol (liquid days):")
print(med[amb].dropna().sort_values().round(2).to_string())
print("\ncrypto-labeled symbols with the LOWEST median ratio (possible mislabels):")
print(med[crypto_old | crypto_new].dropna().sort_values().head(15).round(2).to_string())
