# v5 protocol: fresh data, complete universe, pre-registered tests

Written and committed on 2026-11-10, **before any result from the new data was looked at**. The
git history is the proof of order: this file lands before any analysis script output.

## Why this exists

Everything in `STRATEGY.md` §1–12 was designed and tested on one dataset. That dataset has four
problems:
- **It ends 2026-06-07.** Every later "out-of-sample" window had been looked at before.
- **It holds 156 symbols picked in June 2026.** Binance's archive has 941 USDT-perp histories.
  The repo's set is the survivors, and it was also selected by June-2026 status.
- **It has no real funding rates** in this checkout. Funding was assumed: 11%/yr, or a 50%/yr stress.
- **It has no positioning data**: open interest, long/short ratios, taker flow.

About 350 strategy variants have been tried on it (`STRATEGY.md` §11). Searching it further only
selects luck. The only new evidence available is new data. Since 2026-11-10 this environment can
reach `data.binance.vision`, so we now have:
- every USDT perpetual ever listed, including delisted ones;
- real funding;
- 5-minute positioning metrics since 2020-09;
- about five months no strategy here has seen: 2026-06-08 → 2026-11-09.

## The three periods

| Name | Dates | Use |
|---|---|---|
| IS | start (2020-06 for prices, 2020-10 for positioning) → 2024-12-31 | design / selection |
| VAL | 2025-01-01 → 2026-06-07 | confirmation (already used for v3/v4 on the old universe) |
| **FRESH** | **2026-06-08 → last complete day (≈ 2026-11-09)** | **final exam, run once** |

**Blindness rule.** Every research loader cuts the data at 2026-06-07 (`DESIGN_END`). FRESH is read
by exactly one script (`fresh_test.py`), run exactly once, after the candidates below are frozen
and committed. No statistic computed on FRESH data may be printed before that.

**What FRESH can and cannot show.** Five months is 0.42 years. The standard error of a Sharpe
ratio measured over that span is about 1/√0.42 ≈ 1.5. FRESH can expose a disaster. It cannot prove
an edge. It will be reported that way, whatever comes out.

## Common machinery (unchanged from v3/v4)

- **Bars and timing:** daily bars, UTC. Signals at the close of day t are held over day t+1.
- **Universe (point-in-time):** USDT-margined **crypto** perpetuals with ≥ 60 daily bars and a
  trailing-30-day median daily quote volume ≥ $5M.
- **Universe hygiene**, fixed here before any result:
  - **By name:** the old `EXCLUDE` list, plus tokenized stocks, ETFs, commodities, FX, pre-IPO
    products and index products (BTCDOM, DEFI, BLUEBIRD, FOOTBALL), plus stablecoin pairs (USDC).
    The list is in `vision_panel.py::NON_CRYPTO`, written from symbol names only.
  - **By data, as a safety net:** a symbol is dropped while its weekend activity ratio is below a
    threshold (it trades like a market that closes on weekends), or while its trailing-30-day daily
    volatility is below 0.5% (stablecoin-like).
    - The weekend ratio is the median |return| on Saturday and Sunday bars divided by the median
      |return| on Monday–Friday bars, over the trailing 60 days. Medians are used so that one
      large pump day cannot flag a real coin.
    - The threshold is calibrated on design-window data only, to separate names labeled non-crypto
      from names known to be crypto. It is frozen before any strategy result is computed and
      recorded under *Deviations*. Strategy returns play no part in the calibration.
  - The live bot uses the exchange's own `underlyingType == "COIN"` flag. That API is geo-blocked
    from this environment, so the two rules above are the offline stand-in.
- **Contracts delisted and relisted under the same name** (`*SETTLED` histories) are separate
  instruments. A position in a contract that stops trading is closed at its last price.
- **Costs:** 7 bps per side on turnover (20 bps for coins under $5M median volume in B6 only).
- **Funding:** the real funding events of each holding day, i.e. events with timestamp in
  (day 00:00, next day 00:00]. Longs pay positive rates and shorts receive them.
- **Risk scaling:** each sleeve is scaled to 40%/yr with its own trailing-30-day realized
  volatility, lagged one day, cap 3×.
- **Sharpe:** daily net returns × √365.

## Part A: bias audit of v3/v4 (no pass/fail, just measurement)

The exact v3 and v4 rules, run on:
1. the old 156-symbol universe with generic funding (should reproduce the published numbers);
2. the old universe with real funding;
3. the full universe (including delisted) with real funding.

Reported for IS and VAL. Pre-committed consequence: if v4 on (3) has IS or VAL Sharpe below 1.0,
the earlier numbers are declared inflated, and the user-facing expectation is reset to (3).

## Part B: eight new hypotheses, each with one specification

A cross-sectional sleeve `xs(signal)` is built as follows:
1. Take the signal on eligible coins and rank it cross-sectionally to [−0.5, +0.5].
2. Multiply by 1/σ(15d).
3. Normalize the long side to +0.5 gross and the short side to −0.5 gross (dollar-neutral).
4. Smooth weights with a 3-day EMA.
5. Vol-target to 40%.

The sleeve is flat when fewer than 10 coins have a signal.

Notation:
- `z30(x)` = (x − mean of the last 30 daily values) / their standard deviation.
- LSA = global long/short **account** ratio, daily mean.
- TTP = top-trader long/short **position** ratio, daily mean.
- OI = open interest in contracts, last value of the day.

| # | Name | Signal (higher = more long) | Why it might work | Prior |
|---|---|---|---|---|
| B1 | Retail crowding, contrarian | −z30(log LSA) | When an unusual share of accounts is long, late retail buyers are in; documented as a contrarian indicator in retail FX positioning | medium |
| B2 | Top traders, follow | +z30(log TTP) | Large accounts are better informed | low–medium |
| B3 | Smart minus retail | z30(log TTP) − z30(log LSA) | Composite of B1 and B2 | medium |
| B4 | OI-confirmed move | tanh(2·r7/(σ30·√7)) × (2·rank(Δ7 log OI) − 1) | A move with rising open interest is new positioning (continue); with falling open interest it is liquidation or covering (fade) | medium |
| B5 | Funding-crowding veto on v3 | v3, but zero a long signal when trailing-3-day funding is above +50%/yr annualized, and a short signal when below −50%/yr | Don't pay >50%/yr to hold a trend; extreme funding marks a crowded side | medium |
| B6 | Small-cap extension of v4 | v4 with the liquidity floor at $1M instead of $5M; 20 bps/side for coins under $5M | A $100 account has no market impact; anomalies are usually stronger in small names | medium |
| B7 | New-listing short (re-test) | Short each coin from day 2 to day 30 after its first bar; inverse-vol (10d), 25% name cap; long BTC for equal notional; needs 3-day median volume ≥ $5M | Supply overhang after listing. Rejected in §8 on survivors-only data with stressed funding; that bias ran against the short side | low–medium |
| B8 | Taker-flow persistence | +z over 60d of the 5-day mean taker-buy share of quote volume | Persistent aggressive buying continues | low |

**Pass bars, fixed now:**
- **Standalone sleeves (B1–B4, B7, B8):** IS Sharpe ≥ 1.0 **and** VAL Sharpe ≥ 0.7 **and**
  |correlation with v4 daily returns in IS| < 0.3.
- **Modifications (B5 vs v3, B6 vs v4):** Sharpe better than its baseline by ≥ 0.15 on **both**
  IS and VAL, and VAL max drawdown not worse by more than 10 points.
- **No rescue by re-tuning.** No other parameter values are tried to rescue a fail. Robustness
  sweeps may be run afterwards for diagnosis only and cannot change a verdict.
- **Chance of a false pass:** under the null of no edge, a standalone sleeve passes both bars
  about 1% of the time. Across six sleeves that is roughly a 5% chance of at least one false pass.

## Building v5 (rule fixed now)

- **Base:** v4 (70% trend / 30% pump-follow), with B5 and/or B6 applied if they pass.
- **Added sleeves:** each passing standalone sleeve gets 15% of the risk budget, at most two
  sleeves (highest average of IS and VAL Sharpe first). The base gets the rest.
- **If nothing passes,** v5 = v4, and only v4 goes to the exam.
- **$100 check:** v5 must still be tradable at $100. Through the real bot pipeline on the design
  window, at least 60% of target size must be placed.

## The exam (run once)

**Candidates:**
- **C0:** v4 with the shipped rules (`bot/config.100usd.toml`, $100, 1.5×), through the real bot
  pipeline, on the full live-style universe with the hygiene rules above.
- **C1:** v5, only if it differs from v4.

**Benchmarks:** BTC buy-and-hold, and an equal-weight basket of eligible coins.

**Reported:** total return, monthly returns, Sharpe, max drawdown.

**Pre-committed reading of the result for C0 at 1.5×:**

| FRESH result | Verdict |
|---|---|
| Total return < −20%, or Sharpe < −1 | **Red flag**: recommendation becomes "do not trade live" |
| Between | **Inconclusive**: consistent with a weaker edge than the backtest; paper trade |
| Sharpe > 1 and total return > 0 | **Consistent with the backtest**: paper trade, then small live |

**Ship rule:** the bot's preset changes only if v5 ≠ v4 and v5's FRESH result is not a red flag.

## Deviations

Any deviation from this file, such as a bug found or data unavailable, is appended below with the
date and reason **before** the affected result is used.
