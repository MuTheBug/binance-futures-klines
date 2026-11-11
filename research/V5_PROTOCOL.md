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

### D1 (2026-11-10, before any strategy result): weekend filter calibrated and narrowed

`v5_classifier.py`, design-window data, name labels only:
- **Name-labeled TradFi perps** have a median weekend ratio of 0.04–0.32.
- **Crypto** has a 1% quantile of 0.31 and a median of 0.8.
- **Three names from the old exclude list look like crypto** on this measure: BILL, CHIP and
  GENIUS at 0.7–1.25. They stay excluded by name, as pre-registered.
- **Threshold frozen at 0.35.** On liquid symbol-days it wrongly keeps 12% of TradFi days (mostly
  gold and oil products, which are excluded by name anyway) and wrongly drops 1.6% of crypto days.
- **Narrowing:** the weekend rule is applied only to instruments first listed on or after
  2025-01-01.
  - Reason: applied to everything, it would drop BTC on 362 liquid days, ETH on 204, and SOL, XRP,
    ADA and BNB on about 100 each. The majors have quiet weekends too.
  - No tokenized TradFi perp existed before 2025.
- **Stablecoin rule:** it flags only USDC, which is already excluded by name.

---

## Results before unblinding (recorded 2026-11-11, FRESH still untouched)

### Data obtained

- **Instruments:** 941 USDT-perp histories. 755 have bars in the design window, and 149 of those
  stopped trading inside it. The old repo set was 156 files, 124 after exclusions.
- **Funding:** 2.85M funding events. Real funding covers 99.7% of eligible coin-days.
- **Positioning:** 487k symbol-days of 5-minute metrics, covering 89% of eligible coin-days since
  2020-10.
- **Eligible coins per day:** median 162, against about 40 in the old universe.

### Part A: bias audit (`v5_audit.py`)

Sharpe, IS / VAL:

| Setup | v3 trend | Pump-follow | v4 blend |
|---|---|---|---|
| 1. Old universe, assumed funding (published) | 1.45 / 1.52 | 1.36 / 1.47 | 1.92 / 2.11 |
| 2. Old universe, real funding | 1.38 / 1.43 | 1.49 / 1.85 | 1.94 / 2.24 |
| 3. Full universe, assumed funding | 1.78 / 2.27 | 0.23 / −0.25 | 1.56 / 2.05 |
| **4. Full universe, real funding** | **1.67 / 2.13** | **0.60 / 1.09** | **1.68 / 2.46** |

- **Setup 1 reproduces the published numbers exactly.**
- **The pump-follow sleeve was mostly a survivorship and selection artifact.**
  - On the complete universe its IS Sharpe is 0.60 (not 1.36) and its max drawdown is −72%.
  - Pump triggers: 1710, against 385 in the old data.
  - What remains leans on funding. Pumped coins usually have *negative* funding (median +11%/yr
    but mean −113%/yr), so longs are paid.
- **The trend book is better on the complete universe.** It has more candidates, and the delisted
  coins were good shorts.
  - 30 bps/side costs: 1.32 / 1.73.
  - Entering 1 or 2 days late: 1.54 / 2.03 and 1.41 / 2.16.
  - 40 most liquid coins only: 1.55 / 1.38. 100 most liquid: 1.90 / 1.79.
- **Trend book weaknesses:**
  - Without its 20 best days it is 1.03 / 1.19.
  - Its best day (+28%, 2025-10-10) was a liquidation cascade. Exchanges auto-deleveraged
    profitable shorts that day, so a live account may have captured less.
  - 52% of P&L comes from 10 coins.
  - IS profit came from longs. VAL profit came almost entirely from shorts.
- **v4 by year (setup 4):** 2020 2.75 · 2021 3.03 · 2022 1.22 · **2023 0.33** · 2024 1.85 ·
  2025 2.16 · 2026 3.07.
- **Pre-committed consequence:** v4's IS and VAL Sharpe on setup 4 are both above 1.0, so v4 is
  not declared inflated as a whole. Its pump component is.

### Part B: eight hypotheses (`v5_hypotheses.py`): **none passed**

| # | Hypothesis | IS | VAL | Verdict |
|---|---|---|---|---|
| B1 | Retail crowding, contrarian | −0.40 | −1.12 | fail |
| B2 | Top traders, follow | −1.01 | −2.53 | fail |
| B3 | Smart minus retail | −0.49 | −2.31 | fail |
| B4 | OI-confirmed move | −2.33 | −0.49 | fail |
| B5 | Funding veto on v3 (base 1.67 / 2.13) | 1.69 | 2.11 | fail (no gain) |
| B6 | Small-cap extension of v4 (base 1.68 / 2.46) | 1.63 | 1.62 | fail (worse) |
| B7 | New-listing short | 0.88 | −0.01 | fail |
| B8 | Taker-flow persistence | 0.80 | 0.65 | fail (both just under the bars) |

Diagnostic for B1–B4 (cannot change a verdict): the reversed signs also lose after costs in at
least one period. These sleeves turn over 0.5–1.0 of the book per day, and at 7 bps a side that
costs more than the signal carries. Zero-cost Sharpes are between −1.6 and +1.0 with no consistent
sign. Daily positioning data, used this way, has no tradable edge here.

### v5 = v4 (by the rule)

Nothing passed, so only C0 goes to the exam.

### Harness changes made before unblinding

- **`bot/replay.py`:** real funding per holding day; a daily tradable-universe callback (what the
  exchange lists that day); settlement of positions in contracts that stop trading.
- **Bot universe hygiene:** stablecoin guard (`min_daily_vol` 0.5%), and the exclude list extended
  to the non-crypto names. Without it the live bot would have treated USDCUSDT as a coin.
  Old-universe results are unchanged (parity test passes).
- **Harness check on VAL** (design data): C0 at $100, 1.5× through the real pipeline:
  $100 → $471, Sharpe 2.40, max DD −19.4%, 76% of target size held, 2 delisting settlements.
  Idealized 1×: Sharpe 2.46.

### What the exam will run (frozen)

- **C0 at $100, 1.5×** (the verdict).
- **Declared diagnostics:** C0 at 1.0×; C0 at $1,000; BTC and equal-weight benchmarks; the
  idealized trend and pump parts.
- Funding events for the current month are not yet in the archive. Days without funding data
  assume zero.

---

## Exam, first run (2026-11-11): FRESH = 2026-06-08 → 2026-11-09, 155 days

Raw output is saved in `V5_FRESH_RESULT_first_run.json`.

| Candidate | Result |
|---|---|
| **C0: v4 as shipped, $100, 1.5×** | **$100 → $89.66 (−10.3%), Sharpe −0.25, max DD −26.9%** |
| C0 at 1.0× | $100 → $94.90 (−5.1%), max DD −18.3% |
| C0 with $1,000, 1.5× | −14.6%, max DD −28.5% |
| BTC buy-and-hold | +23.4% |
| Equal-weight eligible coins | +15.2% |
| v4 idealized 1× | −9.4% (trend part −20.4%, pump part +12.2%) |

C0 at 1.5× by month: Jun +1.3% · Jul +7.5% · **Aug −22.8%** · Sep +2.0% · Oct +4.3% · Nov +0.4%.

**Pre-committed reading: "Inconclusive: consistent with a weaker edge than the backtest."**
The result is not below −20% and its Sharpe is not below −1. It is negative.

### D2 (2026-11-11, found after the first run): the archive has no bar for 2026-10-28

**The defect.** `data.binance.vision` has no kline for 2026-10-28, for any symbol, at any interval.
The monthly October file has 30 rows and the daily files return 404. The exchange did trade:
BTC's close on the 27th and open on the 29th are continuous with the mark-price bar for the 28th.
It is the only market-wide hole in the whole panel, so the design-window results (Parts A and B)
are unaffected.

**Effect on the first run:**
- The harness saw "no price" for every position on the 28th and settled the whole book at the
  27th's close. Those are the "18 settled" positions.
- The strategy's rule of 60 consecutive daily bars then made every coin ineligible, so the
  account sat flat from 10-28 to 11-09.
- A live bot would not have seen this, because the exchange API has the bar.

**Repair, fixed here before re-running:**
- Binance's own daily **mark-price** kline for 2026-10-28 (which the archive does have) supplies
  open, high, low and close for each symbol.
- Volume fields are the mean of the two adjacent days. They only feed a 30-day median.
- Nothing about the strategy, the config or the harness logic changes.

**Reporting.** Both runs are reported. The repaired run is the better measure of what the bot
would have done. The first run stays on record above.
