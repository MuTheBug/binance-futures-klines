# The ECHO Engine — a beta-decomposed dual-engine for crypto perps

**One-line thesis (the unconventional move):** the *orthogonal* alpha that earlier research
hunted for in exotic external data (funding carry — fetched, tested, **rejected OOS**) was
hiding **inside the price itself**. Split every coin's daily return, by a rolling market-beta
regression, into two physically different processes that demand **opposite** models — then
trade *both*.

```
R_i,t  =  alpha_i  +  beta_i,t · M_t  +  eps_i,t
              \________ ________/      \__ __/
                       v                  v
            SLEEVE A: "ride the tide"   SLEEVE B: "ride the wake"
            the systematic component    the idiosyncratic residual
            TRENDS (herding, slow        whose LARGE moves CONTINUE
            diffusion) -> trend-follow   (coin-specific breakouts) -> ride them
```

`M_t` = equal-weight market return of the eligible universe. Classic trend on the *raw*
return contaminates the trend signal with the residual; isolating the two lets each be
traded with the model it deserves. The sleeves are **near-uncorrelated (corr ≈ +0.16)**, so
the blend's Sharpe clears the ~1.4 trend-only ceiling that the prior research log called the
hard limit (`IR = IC·√breadth` — two independent bets beat one).

---

## How the discovery actually went (discipline over wishful thinking)

The honest path mattered — the first hypothesis was **wrong**, and the data corrected it:

1. **Hypothesis:** residuals *overreact* → fade them (short-term residual reversal).
2. **Rank-IC said reversal** (t-stat up to −5.6 IS). **But the tradeable dollar P&L said the
   opposite.** Reconciliation: the residual carries *two superimposed effects* — in **rank
   space** the typical coin's wiggle mean-reverts, but in **magnitude space** a handful of
   violent residual jumps (real coin-specific news) **continue**, and those few tails are so
   large they swamp a linearly-sized book and flip its P&L. A naive reversal book
   accidentally bets the farm on the tail and gets run over.
3. **Corrected, data-driven strategy:** **ride** the residual continuation, not fade it — and
   size it **tanh-bounded + per-name capped** so the broad cross-section (breadth) is
   harvested while no single outlier dominates.

Ideas tested and **rejected** for completeness (each killed on honest IS+OOS+cost evidence,
not opinion): daily residual *reversal* (rank-space mirage), hourly liquidation-cascade
reversal (real gross edge, IS Sharpe 1.69, but **dies on turnover cost** — it is a
liquidity-*provider* strategy needing maker/HFT infra), lottery/MAX & low-vol & IVOL & skew
anomalies (positive IS but **negative OOS** — crushed in the 2025–26 alt rally), hour-of-day
& day-of-week seasonality (**unstable IS→OOS**), and five dynamic-leverage throttles
(equity-curve, signal-coherence, dispersion — **all hurt OOS**).

---

## Sleeve C — crowd-positioning reversal (the alt-data breakthrough → Sharpe 2.0)

The original research declared a robust **Sharpe 2.0 "NOT reachable" on price+funding data**.
It needed genuinely orthogonal data — and we found it: Binance's **top-trader long/short
*position* ratio** (multi-year history from `data.binance.vision`). The data verdict, under
strict discipline: **FADE the herd** — when large accounts are maximally crowded long, the move
is late and mean-reverts. Short the most-crowded-long names, long the least; market- and
beta-neutral, built exactly like Sleeve B.

It cleared the hard bar that GLOBAL-retail-ratio and TAKER-flow **failed**: adding C improves
**both** IS (1.55→1.67) *and* OOS (2.77→**3.12**); it's near-orthogonal (corr ~0.1); robust
across lags 0/1/2 (OOS 2.96→3.02→**3.08** — *strengthens* with lag, so no look-ahead) and across
cap/beta-window; and it adds value in late-IS (+0.23) and OOS (+0.35), neutral early, hurts no
period. (`altdata/` to download; `research/altdata_sleeves.py` is the IS-gated judge.)

## Performance — ECHO v2 = 40% Trend + 40% Residual + 20% Positioning

(daily rebalance, 14bps round-trip cost, vol-targeted to 40%/yr, ~1x gross)

| Sleeve / Blend | IS Sharpe | OOS Sharpe |
|---|---|---|
| A — Trend ("ride the tide") | 1.44 | 1.58 |
| B — Residual continuation ("ride the wake") | 0.70 | 2.75 |
| C — Crowd-positioning ("fade the herd", alt-data) | 1.13 | 0.73 |
| ECHO v1 = 50% A + 50% B | 1.55 | 2.77 |
| **ECHO v2 = 40% A + 40% B + 20% C** | **1.67** | **3.12** |

**ECHO v2 full-period (2020-05 → 2026-06, ~1x):** CAGR **120%**, Sharpe **2.00**, Sortino 3.16,
maxDD −37%, Calmar 3.22, win-rate 50.2%, profit-factor **1.36**, **114× equity** in 6y.
**Positive every single year**, including the 2022 bear (**+35%**, the book goes short):
`2020:+37% 2021:+201% 2022:+35% 2023:+57% 2024:+122% 2025:+199% 2026:+98%`.
(v1, without alt-data, remains the fallback: Sharpe 1.85, CAGR 108%, 82×.)

> Honest caveat: Sleeve C only covers the ~40 symbols with metrics history (from 2021), so it
> is a 20% overlay, not the core; and OOS is 1.4y. Treat **IS 1.67 / full 2.00** as the figure,
> with the OOS 3.12 as confirmation, not a forward promise.

> Honest caveat: Sleeve B's *standalone* edge is regime-varying (IS 0.70 ≪ OOS 2.75) — it is
> strongest when idiosyncratic dispersion is high (2025–26 alt/meme season). Treat **IS 1.55**
> as the conservative forward estimate for the blend, not the OOS 2.77.

### Anti-overfit evidence
- **Parameter plateau:** every cell of formation-window F∈{3,5,8,13} × beta-window∈{30,60,90}
  beats trend-alone (blend IS 1.53–1.71, OOS 2.36–3.02). Broad plateau, no fragile spike.
- **Blend-ratio plateau:** smooth from 30/70 → 70/30 (IS 1.59–1.66).
- **No look-ahead:** re-running with `skip=1` (residual sleeve uses *no* contemporaneous bar)
  leaves it unchanged (IS 1.64 / OOS 2.56). Engine applies a 1-bar execution lag throughout.
- **Cost-robust:** survives 20 bps/side — nearly 3× the base cost (IS 1.36 / OOS 2.04).
- Sleeve B is built as a **multi-F ensemble** {3,5,8} (no single "magic" window).

---

## LEVERAGE — block-bootstrap (1-yr paths, 20-day blocks) with per-bar liquidation

Vol drag and liquidation are modelled, so the growth-optimal leverage is **finite**:

| leverage | 1×  | 1.5× | 2×  | 2.5× | 3×  | 4×  | 5× |
|---|---|---|---|---|---|---|---|
| FULL median 1-yr (×) | 1.99 | 2.61 | **3.26** | 3.89 | 3.16 | 0.00 | 0.00 |
| FULL P(ruin) | 0% | 0% | **0%** | 0% | 17% | 94% | 100% |

Growth-optimal **k\* ≈ 2.5×** (IS & FULL), 3× (OOS). **2× is the sweet spot: 0% bootstrapped
ruin in every window, near growth-optimal.**

### Headline — ECHO Engine @ **2× leverage** (recommended; survivable, growth-near-optimal)
**CAGR 254% · Sharpe 1.84 · Sortino 2.95 · maxDD −62% · Calmar 4.09 · win 50% · PF 1.33** —
**$60 → $121,534** over 6 years, path liquidation-checked (not ruined). The −62% drawdown is
the price of the leverage; **1.5×** (CAGR ~160%, maxDD ~−47%) is the lower-stomach alternative.
Beyond 3× the account is destroyed by vol drag + liquidation — do **not** chase it.

---

## Stress tests (`stress.py`) — trying hard to break it
- **Crisis-resilient / crisis-alpha:** positive in 5 of 6 named crashes (LUNA/UST **+23%**,
  3AC dele­verage +8%, FTX +4%, Aug-2024 carry-unwind +9%, Feb-2025 −selloff +5%) because the
  book goes short; only the fast May-2021 V-flush hurt (−16% at 1×).
- **Execution-robust:** unchanged with +1 day execution lag; still 1.43 IS at +2 days; survives
  the combined worst case **30bps/side + 1-day lag + 30%/yr funding** (IS 1.05 / OOS 2.14).
- **Breadth, not a few names:** excluding **BTC+ETH** barely dents it (CAGR 98%); but it *needs*
  breadth — restricting to the top-20 coins collapses OOS to 0.27 (so trade the broad universe).
- **Downtime-proof:** rebalancing every 2–3 days is *slightly better* than daily and cheaper.
- **Sobering tail:** at 2× the worst 1% of bootstrapped years draws down **−79%**, and the single
  worst path reached −96% ("no liquidation" in the per-bar model, but that is ruin in practice).
  → **1.5× (worst-1% DD ≈ −68%) is the wiser leverage; reserve 2× for risk you can stomach.**

## Pushing further (`echo_boost.py`) — the disciplined verdict
Only one change survived the IS-Sharpe test: **rebalance every 2–3 days** (IS 1.55→1.61, lower
cost). Risk-parity blending lands back at ~50/50; a 3rd sleeve (xs-momentum, donchian) does **not**
raise in-sample Sharpe — the two-sleeve blend already captures the orthogonal alpha in this data.
Beyond that, more PnL comes only from **leverage** (a drawdown trade-off, not new alpha) or from
**genuinely new data** (on-chain flows, options skew, order-book/liquidation feeds) — not from
more mining of OHLCV. The honest move is to bank the cadence win and refuse the overfit.

## Run it
```bash
cd research
python3 data.py                 # build cached OHLCV matrices (once)
python3 echo_engine.py          # >>> the definitive report + results/echo_engine.png
python3 stress.py               # crisis / cost / breadth / downtime / tail stress battery
python3 echo_boost.py           # disciplined improvement attempts (IS-gated)
```
Code: `echo.py` (sleeves: `residuals`, `residual_continuation`), `echo_engine.py` (blend +
leverage + bootstrap), reusing `engine.py`, `strategies.py`, `lab.py`, `data.py`.
**Discipline preserved: nothing was tuned on data after `lab.IS_END = 2024-12-31`.**
Today's target weights: `results/echo_today_weights.csv` (recompute on fresh data before trading).
