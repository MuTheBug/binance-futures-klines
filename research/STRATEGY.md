# Crypto Trend Strategy — Research Log & Continuation Doc

**Goals as stated:** (1) $60 → +40%/month; (2) push Sharpe to 2.0.
**Honest verdict:**
- 40%/month is mathematically impossible (§5).
- Sharpe: raised **1.19 → 1.40 → ~1.4 (real funding costs)** robustly. **A robust 2.0 is NOT reachable**
  on this price+funding data — we tested the prime candidate (funding carry) with real data and it
  fails out-of-sample (crowded out). 2.0 needs genuinely orthogonal data (§6).
- Realistic target: **~5–7%/month at 1.5–2x leverage** ($60 → ~$120–140/yr median), ~33% drawdowns.

Last updated 2026-06-07 (v3 + real funding; carry fetched, tested, rejected).

---

## 1. The recommended strategy (v3, runnable on $60)

**Daily, multi-lookback RISK-ADJUSTED trend strength, long & short, inverse-vol sized,
EMA-smoothed weights, vol-targeted; select from a broad CRYPTO universe, hold the top-10.**

Rules (computed on daily close; trade next bar):
1. **Universe (point-in-time):** USDT perps, ≥60d history, trailing-30d median $vol ≥ $5M.
   Exclude tokenized stocks/metals (NVDA/TSLA/XAU/CL/PAXG…; they only exist 2025-26 & inflate OOS).
2. **Trend strength/coin:** `tanh(k·r_L/(σ·√L))` averaged over L=15,30,60,90d, k≈2, σ=30d stdev.
3. **Size** × inverse vol (15d); normalize Σ|w|=1; **select top-6** by conviction.
4. **Smooth** weights with an **EMA-span ensemble (5,10,15)** (cuts turnover ~60% + whipsaw),
   then **hold the top-10** (runnable).
5. **Vol-target** ~40%/yr (lagged, capped 3x); model **real per-coin funding**; **leverage** 1.5–2x.

Goes short → profits in bear markets (positive every year incl. 2022 +33%).

---

## 2. Performance (realistic costs 14bps RT + REAL per-coin funding), crypto-only, ~1x

| Window | Years | CAGR | Sharpe | maxDD | Calmar |
|---|---|---|---|---|---|
| FULL (2020-05→2026-06) | 6.0 | **73%** | **1.39** | −33% | 2.25 |
| In-sample (→2024-12) | 4.6 | 73% | **1.38** | −33% | 2.24 |
| **Out-of-sample (2025-01→2026-06)** | 1.4 | 74% | **1.43** | −24% | 3.11 |

IS 1.38 ≈ OOS 1.43 — tight agreement = trustworthy. Yearly: `2020:+22 2021:+132 2022:+33
2023:+48 2024:+122 2025:+61 2026:+38%`. Monthly: mean +5.5%, median +1.6%, 57% positive,
best +44%, worst −25%. Turnover ~0.17/day, ~10 positions.

### The Sharpe journey (each step IS+OOS-validated)
| Version | Change | FULL Sharpe |
|---|---|---|
| v1 | naive sign-trend | 1.19 |
| v2 | risk-adjusted trend strength (tanh) | 1.40 |
| **v3** | + EMA-ensemble smoothing, top-10 | 1.47 (generic funding) / **1.39 (real funding)** |
| + carry | rejected — OOS-negative (see §4, §6) | — |

---

## 3. Anti-overfit evidence (IS-first; OOS confirmed once; crypto-only)

k-plateau (1–3 → 1.50–1.56), lookback-set & concentration & EMA-span robust, cost-robust to
25–40bps/side, **IS≈OOS**, positive every year, smoothing halves turnover (real cost reduction).

---

## 4. Tested and REJECTED (discipline over wishful thinking)

| Idea | Result | Verdict |
|---|---|---|
| Renko bricks (2/3/5%) | OOS −0.97/−0.73/+0.25 | overfit |
| Dollar/volume bars (López de Prado) | OOS −0.47/+0.06 | overfit |
| Volume profile (POC rev / VA breakout) | OOS −0.78 / regime-luck | reject |
| Multi-speed sleeves; reversal; crash overlay; gk-vol+EMA | OOS↓ / negative / negligible | reject |
| **Funding CARRY** (real data, w/ harvest income) | **IS Sharpe up to 1.18 (t-stat −3.8) but OOS −0.56** | reject |
| — carry + trend (static blend) | FULL↑1.53 but **OOS↓** (1.16–1.32) | reject |
| — carry + factor-momentum timing | binary: OOS↓; continuous: ≈trend-alone | reject |
| Tokenized stocks in universe | OOS 1.43→1.85 but recent-only | exclude |

Two overfit fingerprints, both seen & distrusted: *great-IS/bad-OOS* (renko, dollar bars, carry)
and *bad-IS/good-OOS* (value-area breakout). Code kept: `altbars.py`, `boost*.py`, `carry.py`,
`factor_timing.py`, `funding.py`, `fetch_funding.py`.

---

## 5. Why 40%/month is impossible & leverage→$60 (block-bootstrap, 1yr, w/ liquidation)

| Lev | IS-base med/mo | $60→1yr | OOS-base med/mo | $60→1yr | P(ruin) |
|---|---|---|---|---|---|
| 1.0x | +4.5% | $101 | +3.8% | $94 | 0% |
| **1.5x** | +6.0% | $121 | +5.2% | $110 | 0% |
| **2.0x** | +7.1% | $137 | +6.2% | $123 | 0% |
| 3.0x | +7.8% | $148 | +6.8% | $132 | 0% |
| ≥5x | vol drag → negative, ruin rises to 100% by ~20x |||||

Growth-optimal ≈ 2–3x (median 7–8%/mo). 40%/mo (+5,570%/yr) unreachable at any leverage before
ruin dominates. Recommended **1.5–2x**; keep ≤2x + stops (intraday liquidation not fully modeled).

---

## 6. Can we reach Sharpe 2.0? — the honest answer: not on this data

Robust ceiling ≈ **1.4–1.5** for price-trend. We tested the strongest known diversifier, **funding
carry**, with REAL Binance funding data (fetched for 154 symbols, harvest income modeled):
- It's a genuine factor historically (IS Sharpe to 1.18; funding→fwd-return rank-IC t-stat −3.8).
- But it's **crowded out OOS** (2025-26 basis/yield-product boom): OOS Sharpe −0.56, and every blend
  (static or factor-timed) either hurts OOS or collapses to trend-alone. Rejected.

Remaining honest routes to 2.0 (all = adding *orthogonal* alpha; `IR=IC·√breadth`):
1. **Carry's crowding unwinds** (monitor: trend70/carry30 had FULL 1.53 historically; redeploy if
   carry's trailing Sharpe turns durably positive — hooks exist in `factor_timing.py`).
2. **Orthogonal data we don't have:** on-chain flows, options-implied vol/skew, perp basis term
   structure, sentiment/liquidations.
3. **Intraday execution** for far more independent bets (needs infra/tick data).

---

## 7. Operating procedure ($60) & how to run

- Binance USDT-M perps, crypto only. Rebalance once daily (fixed UTC). ~10 positions; at 2x,
  $120 gross/10 = ~$12 each (mind min-notional; drop names below it). Leverage ≤2x + stops.
- Expect live Sharpe below backtest. Paper-trade 1–2 months first.
- Today (`results/today_target_weights.csv`, 2026-06-07, net short): SHORT LTC/1000SHIB/ETC/DOT/
  CRV/AAVE/WLFI/TRUMP/ADA, LONG TRX. Recompute on fresh data before trading.

```bash
cd research && python3 data.py            # cache klines (once)
python3 fetch_funding.py                   # fetch real funding (once; needs network)
python3 final_v3.py                        # >>> DEFINITIVE strategy + artifacts
python3 validate_v3.py                     # robustness gauntlet + leverage
python3 carry.py factor_timing.py          # carry investigation (rejected, documented)
python3 boost.py boost2.py boost3.py boost4.py run_altbars.py improve*.py  # full search history
```
Code: `data.py engine.py strategies.py lab.py altbars.py funding.py`. `engine.simulate(...,
funding_matrix=F)` uses real per-coin funding. Outputs in `results/`.
**Discipline: tune only on data ≤ 2024-12-31 (`lab.IS_END`); add nothing that doesn't beat OOS.**

---

## 8. v4 (2026-10-28): + pump-follow sleeve — Sharpe ~2.0 in backtest, with real caveats

**Thesis:** a coin that beats BTC by >25% in one day keeps drifting up over the next ~5 days,
because attention, short squeezes and late chasers spread out over days. We trade it by going long
the pumped coin and short BTC (same notional) on the daily close, holding 5 days. This is
short-horizon, coin-specific momentum, nearly uncorrelated with v3 (ρ = +0.02).

**Rules:** daily; universe as v3 (crypto only, ≥60d history, ≥$5M 30d median $vol). Trigger
when the coin's daily return minus BTC's is above +25%. Hold 5 days, size by inverse 20d volatility,
BTC-hedged, vol-targeted to 40% (cap 3x). Longs in pumped coins pay a **stressed 50%/yr funding**.
**Blend: 70% v3 + 30% sleeve.**

| (generic funding, 14bps RT) | FULL Sharpe | IS Sharpe | OOS Sharpe | maxDD | CAGR |
|---|---|---|---|---|---|
| v3 trend | 1.47 | 1.45 | 1.52 | −33% | 80% |
| pump-follow sleeve | 1.38 | 1.36 | 1.47 | −39% | 114% |
| **v4 blend 70/30** | **1.96** | **1.92** | **2.11** | **−28%** | **103%** |

Yearly v4: `2020 +29 · 2021 +185 · 2022 +50 · 2023 +79 · 2024 +156 · 2025 +80 · 2026(5mo) +56%`.
Monthly: median +4.9%, 61% positive, worst −23%.

**Search log (pre-committed pass bar: IS ≥ 0.8, OOS ≥ 0.5, ρ(v3) < 0.3; `explore_v4.py`):**
| Idea | IS / OOS Sharpe | Verdict |
|---|---|---|
| H1 short new listings (d2–30/60/90, hedged or not) | −0.5…−1.4 / −0.4…−1.3 | reject |
| H2 fade big pumps | −1.2…−1.9 / −0.5…−1.9 | reject (pumps *continue*) |
| H3 betting-against-beta | 0.85–1.18 / −0.03…−0.50 | reject (overfit fingerprint) |
| H2′ follow big pumps (sign flip of H2) | 1.36 / 1.47 | **kept, see caveats** |

**Robustness (`robust_v4.py`):** positive IS & OOS on 29/30 grid cells (thr .15–.35 × hold 2–15d),
with a plateau at thr .25–.30 × hold 3–10. Survives 30bps/side costs (1.20/1.28) and entering one
day late (1.28/1.89).

**Caveats, weakest link first:**
1. **Survivorship bias, which favors this sleeve.** The dataset holds only perps still listed in 2026.
   Coins that pumped and were later delisted are missing, along with their crashes. Fix before
   trusting it: backtest on point-in-time data that includes delisted symbols.
2. **Contaminated OOS.** H2′ is the sign-flip of H2, whose OOS was seen first. The 2.11 OOS is not a
   clean holdout. Only paper/live trading gives a clean test.
3. **Lottery profile.** The sleeve wins on 14–29% of days. Dropping its 20 best days takes IS Sharpe
   to ~0. A dry spell of many months is normal, not proof that it broke.
4. **Execution.** Entry is at the close of a +25% day, when spreads are widest and funding spikes
   (stressed at 50%/yr here, but real spikes can run to hundreds of %/yr). Worst sleeve day −23%.
5. v3 numbers here use generic funding. With real funding v3 was ~0.08 lower (§2).

**Kill switch:** pause the sleeve if its trailing 6-month Sharpe is < −0.5, or if live fills average
more than 50bps worse than the close. Pause everything at −35% drawdown (above the backtest max).

**Leverage (1yr block bootstrap, IS base):** 1x +5.7%/mo median · 1.5x +8.2% · 2x +10.4% (P lose
half 1%) · 3x +14% (3%). Bootstrap ruin understates intraday liquidation, so **stay ≤2x**.

```bash
python3 explore_v4.py      # H1/H2/H3 search (all rejected)
python3 robust_v4.py       # pump-follow gauntlet
python3 final_v4.py        # v4 blend tearsheet + leverage table
```
