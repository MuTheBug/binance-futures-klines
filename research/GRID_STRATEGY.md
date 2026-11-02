# Grid trading on Binance USDT-M futures: study and verdict

**Question:** can a grid bot be "highly profitable" starting from $100?
**Short answer:** see the Verdict at the end. The evidence is below, in the order it was produced.
Code: `gridsim.py` (simulator), `grid_meanrev.py`, `grid_baseline.py`, `grid_search.py`,
`grid_final.py`, `grid_account100.py`. The bot lives in `gridbot/`.

---

## 1. What a grid actually is (theory and literature)

A grid places buy limits below the price and sell limits above it, at fixed spacing `g`. Every
down-then-up crossing of a level pair books `g` (minus fees). The catch is the inventory: the grid
holds more coins the further the price falls, and fewer the further it rises.

- **Expected value is zero on a random walk.** Chen, Chen & Jang ("Dynamic Grid Trading Strategy:
  From Zero Expectation to Market Outperformance", arXiv 2506.11921, 2025) prove that a classic
  grid has zero expected value under a symmetric random walk with no fees.
- **It is a short-volatility position**, like a short straddle without the option premium. It wins
  in ranges and loses in trends (Quantpedia, "What's the relation between grid trading and delta
  hedging?").
- **The "grid profit" exchanges display excludes the inventory loss.** The P&L that matters is grid
  profit plus the unrealized loss on the bag.
- **The only real sources of edge** are drift (a long grid on a rising coin, which is just market
  exposure) and **mean reversion at the grid's scale**: more back-and-forth level crossings than a
  random walk produces.
- **The "dynamic grid" paper (DGT) mostly measures something else.** Its reported 60–70%/yr IRR
  on BTC/ETH for 2021–24 comes from rules that keep the coins bought on every break below the
  range and **inject new outside capital** to fund the next grid. That is dollar-cost averaging
  into a falling coin, and its IRR formula treats later deposits as if invested on day one. In a
  sample that ends in a bull market, that flatters the result. The published code also does not run
  as written: two calls pass a `max_price` argument the functions do not accept.
- **Other evidence is thin and mixed.** An EAI 2023 BTC backtest found parameter-dependent results.
  A Stevens Institute study found ML-tuned grids no better than plain ones. A 2026 walk-forward
  study over 17 coins found markets were trending almost all the time, starving grids.

## 2. Is there mean reversion to harvest? (`grid_meanrev.py`)

Lo–MacKinlay variance ratios with the heteroskedasticity-robust z\*, per coin per calendar year,
on hourly log returns. VR < 1 means reversal.

| Median VR across coins | 2h | 4h | 8h | 24h | 72h | 1 week |
|---|---|---|---|---|---|---|
| 2020 | 0.94 | 0.88 | 0.84 | 0.88 | 0.83 | 0.83 |
| 2022 | 1.01 | 0.97 | 0.95 | 0.96 | 0.95 | 0.91 |
| 2023 | 0.97 | 0.95 | 0.91 | 0.88 | 0.85 | 0.85 |
| 2024 | 0.99 | 0.96 | 0.94 | 0.91 | 0.93 | 0.95 |
| 2025 | 1.00 | 0.99 | 1.00 | 0.96 | 0.88 | 0.78 |

- **Mild reversal at 4h to 1 week, every year.** 9–15% of coin-years show significant reversal at
  4–8h. Almost none show significant trending at these short horizons. Idiosyncratic returns (coin
  minus beta × BTC) show the same.
- **It flips from year to year.** BTC's 1-week VR was 1.29 (trending) in 2023 and 0.72 (reverting)
  in 2025.
- **So the raw material exists, but it is weak and unstable.** The open question is whether it
  survives fees, funding and the occasional trend.

## 3. Simulator and its validation (`gridsim.py`)

- **Model:** geometric grid, equal USDT per level, inventory as a closed-form function of the
  last-filled level. All 128 coins run vectorized on hourly bars, with the intrabar path
  open → low → high → close on up bars (open → high → low → close on down bars).
- **Costs:** maker 0.02% on grid fills, taker 0.05% + 0.05% slippage on market trades, funding
  11%/yr (longs pay), and liquidation when a grid loses its allocation.
- **Order-book check:** the vectorized simulator matches an explicit order-book reference to
  **2e-13** on 24 configurations (long/neutral/short × hold/reset × trade-through × path order).
- **Production check:** the bot's engine plus its exchange simulator matches the research
  simulator to **< 0.02% of equity**, including a real −18% SOL wick (`gridbot/tests/test_engine.py`).
- **Theory check:** 60 synthetic paths × 60 days at minute resolution.
  - Mean-reverting prices make money; trending prices blow up; a martingale gives ≤ 0.
  - **The simulation is conservative.** Sampling the path misses back-and-forth crossings that
    would have earned money, so fine grids are biased down.
  - Hourly bars catch 98% of the minute-path fills at `g` = 1 hourly sigma, but only ~50% at 0.25
    sigma. **Study grids therefore use `g` ≥ 1.5 × hourly volatility.**
- **Limitation:** no 1-minute or tick data. Binance's data host is blocked in this environment, so
  intrabar ordering is modelled, not observed.

## 4. Baseline: classic grids as people deploy them (`grid_baseline.py`)

Monthly re-centred, static inside the month (hold when price leaves the range), 1x exposure, 128
liquid crypto perps, IS = 2022-06..2024-12, OOS = 2025-01..2026-06. Each grid's P&L is measured on
its allocated capital.

| | DEV-IS Sharpe | HOLDOUT-IS Sharpe | OOS Sharpe | OOS CAGR |
|---|---|---|---|---|
| Equal-weight buy & hold | 1.00 (IS) | | −0.30 | −42% |
| Neutral, ±10–25% range, g 0.5–4% | −1.79 … 0.00 | −2.27 … −0.44 | −2.98 … −0.44 | −72% … −17% |
| Long, ±10–25% range, g 0.5–4% | −0.26 … 0.83 | −0.58 … 0.61 | −1.81 … −0.66 | −65% … −39% |

**Classic grids lose to buy-and-hold.**
- Neutral grids lose everywhere, with ~0.3 liquidations per grid-year: a coin that doubles in a
  month wipes the short side.
- Long grids are a diluted buy-and-hold: up in the 2022–24 alt bull market (still below
  buy-and-hold), down more than buy-and-hold in the 2025–26 alt crash.
- Coarser grids always beat finer ones.

## 5. Improvement search: 120 pre-registered designs (`grid_search.py`)

The grid of designs:
- Spacing: {1.5, 3, 6} × 7-day hourly volatility, or fixed {2%, 4%}.
- Levels: {5, 10} per side.
- Exit: {hold + monthly re-centre, reset when the range breaks, stop + 24h cool-down}.
- Bias: {neutral, **neutral only while the daily trend is weak** (\|v3 strength\| < 0.3), long, follow the trend (long / short / neutral)}.

Selection used **DEV coins ≤ 2024 only**. Held-out coins and 2025–26 were looked at for the top 3 only.

What the design-period ranking shows:
- **The trend filter is what makes neutral grids work.** It takes 12 of the top 13 places. Unfiltered neutral grids score −1.50 … +0.65.
- **Wider spacing wins:** 6σ and 4% beat 3σ, which beats 1.5σ and 2%. That matches the stronger weekly reversal in §2, and wide grids are also least exposed to fees and simulation bias.
- **Long grids score 0.3–1.1,** but in 2022–24 that is alt-bull beta (buy-and-hold: 1.00).
- **Trend-following grids are mixed** (−0.55 … 1.32).

| Top 3 (by DEV-IS) | DEV-IS | HOLDOUT-IS | **OOS 2025–26** |
|---|---|---|---|
| 6σ, 10 levels, reset, neutral + trend filter | 1.56 (+7.7%/yr, DD −3.6%) | 1.51 (+10.3%) | **0.05** (+0.1%, DD −8.0%) |
| 6σ, 10 levels, stop 24h, neutral + trend filter | 1.53 (+7.6%, DD −3.6%) | 1.62 (+10.8%) | **0.07** (+0.3%, DD −7.9%) |
| 4%, 5 levels, stop 24h, neutral + trend filter | 1.44 (+21.1%, DD −11.2%) | 0.66 (+11.3%) | **−0.00** (−2.8%, DD −26.2%) |

**Pre-committed bar (IS ≥ 1.0, OOS ≥ 0.7, HOLDOUT ≥ 0.7, DD ≤ 35%): failed on OOS.**
- The leading design held up on coins never used in design, so it isn't fitted to particular coins.
- Its edge then disappeared in 2025–26. It did not lose money (−8% worst drawdown while the alt basket fell 42%/yr), but it did not make any either.

## 6. Robustness of the leading design (`grid_final.py`)

| 6σ / 10 levels / stop 24h / neutral + trend filter | DEV-IS | HOLDOUT-IS | OOS |
|---|---|---|---|
| Base (conservative intrabar path) | 1.53 | 1.62 | 0.07 |
| Limit fills need a 0.05% trade-through (queue) | 1.39 | 1.47 | −0.09 |
| 2× fees | 1.36 | 1.49 | −0.07 |
| Funding 0% / 30% | 1.55 / 1.49 | 1.63 / 1.60 | 0.09 / 0.05 |
| **Reversed intrabar path** | **3.20** | **2.97** | **1.59** |

- **Sharpe by year:** 2022 0.57 · 2023 1.62 · 2024 2.41 · **2025 0.14 · 2026 −0.16**. The edge has decayed.
- **Market exposure:** daily correlation with equal-weight buy-and-hold is +0.15 (IS) and +0.05 (OOS). The design is close to market-neutral.
- **The biggest uncertainty is what happens inside each hour.** The reversed path adds ~25% more round trips and doubles the Sharpe.
  - The base path is the shortest route through each bar's high and low, so it is conservative.
  - The reversed path adds an extra excursion through every bar, so it is probably optimistic.
  - The truth lies between them, or above if prices wiggle within the hour. Settling it needs 1-minute data, which is blocked in this environment, or the bot's paper mode on live prices.

## 7. Also tested and rejected: BTC-hedged grid on coin/BTC (`grid_ratio.py`)

Motivation: idiosyncratic returns mean-revert too (§2), and hedging out BTC removes market trends.
- **Result:** all 18 variants lose, even at 6σ spacing with ~0.3 fills a day.
  - Best DEV-IS: −1.86.
  - Same design on HOLDOUT: −2.31.
  - Same design on OOS: −1.87.
- **Why:** alts drift persistently against BTC, so the hedged grid keeps accumulating the losers.
- **Costs assumed:** two market orders per level, 0.2% per fill.
- **Caveat:** the close-only simulation is harsh on fine spacings, but not enough to explain −2 Sharpe at the widest spacing.

## 8. What a $100 account would have experienced (`grid_account100.py`)

Leading design, one coin per $100 account (compounding, liquidation modelled), every liquid coin:

| Window | Exposure | Median end | p10 – p90 | Ended up | Median / worst DD | Buy-and-hold median |
|---|---|---|---|---|---|---|
| 2023–24 (2 yrs) | 1x | $102 | $90 – $110 | 62% | −8% / −30% | **$252** |
| | 3x | $104 | $66 – $131 | 56% | −23% / −73% | |
| 2025-01 → 2026-06 | 1x | $101 | $89 – $107 | 57% | −5% / −37% | **$23** |
| | 3x | $102 | $67 – $120 | 55% | −16% / −85% | |

- **One coin:** the trend filter keeps the grid off ~76% of the time, so a single-coin account mostly waits. The bot's rotation mode (`max_active`) moves the slot to whichever coins currently qualify. That is how the portfolio rows in §5 deploy capital: +8–11%/yr at 1x in 2022–24, ~0% in 2025–26.
- **Leverage widens the spread of outcomes, not the median.**
- **Minimum order size:** at ≤ 10 levels a side, $100 (or $50 per slot with 2 slots) stays above Binance's ~5 USDT per order on liquid alts. The bot shrinks the level count and refuses coins whose lot size would inflate exposure.

## 9. Verdict

1. **No grid design passed the pre-registered test.**
   - The best one, a neutral grid that runs only when the daily trend is weak with wide (6σ) spacing, was real across coins in 2022–24: Sharpe ~1.5, +8–11%/yr at 1x, drawdowns ~4%, almost no market exposure.
   - It went flat in 2025–26.
   - Its level depends on intrabar price order that this data cannot resolve.
2. **"Highly profitable from $100" is not supported.**
   - The realistic range for a $100 grid account was about −10% … +10% a year at 1x, i.e. ± $10.
   - Leverage adds variance, not edge.
   - A grid's best case is a low-correlation, capital-preserving sleeve: in the 2025–26 alt crash it kept $101 of $100 where holding the same coins kept $23. It is not a profit engine.
3. **Classic grids as sold by exchanges are worse than useless here.**
   - Neutral grids lose and get liquidated.
   - Long grids are a diluted buy-and-hold.
   - The "dynamic grid" headline returns come from adding money after losses.
4. **What could change this verdict:**
   - 1-minute or tick data to settle the intrabar question (the reversed-path bound says the design could be good).
   - Months of the bot's own paper-trading record on live prices.
   - Until then: **paper trade only**.
