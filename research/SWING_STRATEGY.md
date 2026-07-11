# CBS v2 — Channel-Breakout Swing (4h, leveraged, long-only) + sleeve blend

**Request:** invent a highly profitable leveraged swing trading strategy at institutional level.
**Honest verdict up front:** "highly profitable" cannot be promised by anyone; what CAN be delivered
is institutional *process*: signal alpha proven at the event level, execution engineered to capture
it, parameters frozen on in-sample data only, one-shot out-of-sample evaluation, leverage sized
against ruin, and full disclosure of the losing year. Numbers below are what the data actually says.

- Standalone CBS: **IS Sharpe 1.12 (CAGR 43%)**, **OOS Sharpe ~0** (2025 −22%, 2026 +53%) — the
  breakout factor itself had a losing 2025 (proven below), not a config artifact.
- **Recommended deployment: 70/30 blend with the v3 daily L/S trend strategy** (correlation 0.3):
  IS Sharpe **1.44**, OOS Sharpe **1.43**, OOS CAGR +54%, maxDD −18% — IS≈OOS, the trustworthy kind.
- Leverage: growth-optimal ≈ 2x on CBS standalone; ruin risk explodes ≥3x. Recommended ≤1.5–2x.

Last updated 2026-07-11. Data: Binance USDT-M perps, 1h→4h bars 2022-05→2026-06 (126 crypto
symbols after excluding tokenized stocks/metals), daily bars from 2020 for regime/universe.
Discipline: **tuned only on data ≤ 2024-12-31**; OOS (2025-01→2026-06) evaluated once, after freezing.

---

## 1. The frozen strategy (exact rules)

**Universe (point-in-time, daily, 1-day info lag):** USDT perps, crypto only, ≥60d history,
trailing-30d median dollar volume ≥ $5M. Tokenized stocks/metals/FX excluded.

**Daily regime (1-day info lag):** long-eligible when `close > EMA50` **and** trend strength
`mean_L∈{20,60} tanh(1.5·r_L/(σ20·√L)) > 0.10`. Conviction for ranking = |trend strength|.

**Entry signal (4h close):** close breaks above the **prior 42-bar (7-day) high**.
No volatility-compression filter, no volume filter (both tested, both HURT — §3).
**Long-only** (every short variant tested has negative expectancy — §3).

**Execution (the part that mattered as much as the signal):**
- **Retest limit order at the broken level**, working 18 bars (3 days), maker fill (2bp).
  No chasing: if price never retests, the trade is skipped.
- Initial stop **3.5×ATR(14, 4h)** below fill; **chandelier trail 5×ATR** below highest close
  since entry. Both **observed at 4h close, exited at next open** (taker 7bp) — no intrabar
  wick-outs. A hard **intrabar disaster stop 3×ATR beyond** the close-stop (fills with 5bp extra slip).
- No time stop. 6-bar re-entry cooldown per symbol. Exit at open after regime flips off.

**Portfolio & risk:** risk **0.375% of equity per trade** (1R), max **20 positions**, total open
risk (heat) ≤ 7.5%, gross ≤ 2x, single-position notional ≤ 30%. Funding modeled at 1bp/8h on long
notional. Optional overlay (default off): drawdown throttle `dd_k=2` (new-trade risk scales by
`1−2·DD`) — cuts maxDD 32→26% for −0.17 Sharpe; a risk-preference knob, not alpha.

Average gross exposure is only ~0.33x (regime-gated) — headline returns already include being
mostly in cash; leverage multiplies this (§5).

---

## 2. The evidence chain (how this was built — replicate before trusting)

1. **Event study first** (`diag`-style, IS only): side-adjusted forward returns after signals.
   Long breakouts in daily uptrends: **+4.7% mean over 7d vs +1.5% unconditional, t=12.6**, edge
   still growing at 15d, fat right tail (median ≈ 0) → made for trail-stop swing trading.
2. **Execution engineering** (IS): naive market-at-open entries + intrabar stops captured almost
   none of that edge (Sharpe 0.21). Two standard institutional fixes, each worth ~+0.45 Sharpe:
   retest limit entries (median breakout initially retraces −0.6%) and close-based stops
   (4h wicks routinely spike 2–3 ATR and recover). Together: **0.21 → 0.69 → 1.12** with breadth.
3. **Breadth beats concentration** (IS): 20 positions × 0.375% risk ≫ 10 × 0.75% (Sharpe
   0.97 vs 0.69, lower DD). The edge is broad; diversify it.
4. **Plateau, not argmax:** every one-step neighbor of the frozen config lands at IS Sharpe
   0.91–1.21 (see `validate_swing.py` output) — the config sits on a plateau, not a spike.
5. **One-shot OOS** after freezing: §4.

## 3. Tested and REJECTED (all on IS; discipline over wishful thinking)

| Idea | Result (IS) | Verdict |
|---|---|---|
| Volatility-compression (squeeze) pre-filter | kills the long edge: event +4.7%→+1.0% @7d; Sharpe → −0.35 | reject |
| Volume-confirmation on breakout bar | Sharpe 1.12 → 0.46 | reject |
| Short breakdown continuation | event mean −2.1% @7d (t=−8.5): breakdowns get BOUGHT | reject |
| Short the bounce in downtrends | event mean −1.7% @7d: bounces keep going | reject |
| Any short sleeve at half risk | drags portfolio (0.28 L/S vs 1.12 long-only) | reject |
| BTC market gate on new longs | 2022 −18%→−16%, hurts 2024; per-coin regime already covers it | reject |
| Buy-the-dip (breakdown in uptrend) | event ≈ 0 (t<3 at all horizons) | reject |
| Tight time stop (7d) | cuts winners exactly where edge keeps growing | reject (no time stop) |

## 4. Results (frozen config, realistic costs incl. funding)

| Window | Years | CAGR | Sharpe | maxDD | trades | win | avgR |
|---|---|---|---|---|---|---|---|
| IS (2022-08→2024-12) | 2.42 | **+42.8%** | **1.12** | −32% | 663 | 32% | +0.53 |
| **OOS (2025-01→2026-06)** | 1.43 | **−4.4%** | **−0.02** | −26% | 494 | 29% | −0.04 |
| FULL | 3.85 | +23.0% | 0.77 | −38% | 1159 | 30% | +0.29 |

Yearly: 2022(Aug–) −18%, 2023 +62% (Shrp 1.49), 2024 +78% (1.52), **2025 −22% (−0.76)**,
2026(→Jun) +53% (1.72). Monthly profile is right-skewed trend-following: median month −1.4%,
38% positive, best +84%, worst −13%. You are paid rarely and big; expect long flat-to-down stretches.

**Why OOS failed, honestly diagnosed:** the SIGNAL itself inverted in 2025 — event-level forward
returns after breakouts: IS +4.7% @7d (t=+12.6) → **2025 −2.3% (t=−5.9)** → 2026 +3.4% (t=+3.5).
Every neighbor config fails OOS the same way (validate output) → not config overfit; 2025 was a
factor-level losing year (alt chop, false breakouts) for long-only breakout trend, while
cross-sectional L/S trend (v3) made +61%. This is regime risk inherent to the style.

Cost-robust: 2x all fees/slippage costs only 0.03 Sharpe (maker entries, ~1.2 trades/day,
median hold ~6 days). Funding 2x costs ~0.09 Sharpe.

## 5. Leverage (the honest version, two independent methods)

Engine re-run with risk×k (adaptive sizing, real compounding) and 3000-path block bootstrap
(static lever, per-bar liquidation, FULL-period 4h returns):

| k | avg gross | IS CAGR | FULL CAGR | FULL maxDD | bootstrap median 1yr | P(ruin) |
|---|---|---|---|---|---|---|
| 0.5 | 0.17x | +23% | +13% | −23% | — | 0% |
| 1.0 | 0.33x | +43% | +23% | −38% | x1.20 | 0% |
| 1.5 | 0.49x | +58% | +29% | −50% | x1.26 | 0% |
| 2.0 | 0.63x | +68% | **+32%** | −60% | x1.28 | 0% |
| 3.0 | 0.92x | +76% | +28% | −75% | x0.77 | **24%** |
| 4.0 | 1.19x | +70% | +16% | −86% | ruin | **75%** |

Sharpe is leverage-invariant; only the drawdown/ruin trade-off changes. Growth-optimal ≈ 2x;
**recommended ≤1.5x standalone** (a −50% drawdown is already beyond most tolerances). The
bootstrap is harsher than the engine because it doesn't shrink positions after losses — reality
sits between the two. Both agree: **≥3x is where accounts die.**

## 6. Recommended deployment: sleeve portfolio (institutions don't run one strategy)

CBS daily-return correlation to v3 (daily cross-sectional L/S trend, `final_v3.py`): **+0.30 IS,
+0.12 OOS** — genuinely diversifying (convex long-trend sleeve vs market-neutral-ish L/S sleeve).

| Allocation | IS Sharpe | IS CAGR | IS maxDD | OOS Sharpe | OOS CAGR | OOS maxDD |
|---|---|---|---|---|---|---|
| v3 100% | 1.29 | +67% | −32% | 1.52 | +82% | −22% |
| **v3 70 / CBS 30** | **1.44** | +64% | **−25%** | **1.43** | +54% | **−18%** |
| v3 50 / CBS 50 | 1.49 | +59% | −26% | 1.25 | +36% | −18% |
| CBS 100% | 1.16 | +43% | −32% | −0.04 | −4% | −26% |

The 70/30 blend improves IS Sharpe AND drawdown over v3 alone, and OOS confirms (1.43≈IS 1.44).
That is the institutional-grade product here: **70% v3 + 30% CBS at ≤1.5–2x total leverage**
(historically ≈ 4–6%/mo median at 1.5–2x; NOT 40%/mo — see v3 STRATEGY.md §5 for why that's
impossible). CBS earns its seat by convexity in trending years (2023/24/26), v3 carries chop years.

## 7. Operating procedure & how to run

- Rebalance point: 4h closes (UTC 00/04/08/12/16/20). Place/refresh retest limits after each close;
  manage stops at closes; disaster stops resting on-exchange (stop-market).
- Small accounts: 20 positions × ~10% notional needs gross ~2x headroom; respect min-notionals.
  Expect live < backtest. Paper-trade first. The strategy is mostly flat/small — that is by design.
- Kill-switch discipline: if realized 12-month Sharpe < −0.5 or event-study edge (re-run yearly)
  stays negative two years running, retire the sleeve.

```bash
cd research
python3 swing_data.py        # build 1h/1d caches + 4h matrices (once)
python3 run_swing.py --oos   # DEFINITIVE strategy: IS + one-shot OOS + artifacts
python3 validate_swing.py    # plateau, stress, leverage ladder, bootstrap, sleeve blend
python3 sweep_swing.py       # IS-only search history (kept for the record)
```
Code: `swing_data.py` (4h data layer), `swing_signals.py` (regime/breakout/eligibility),
`swing_engine.py` (event-driven portfolio engine: retest limits, close-stops + disaster stops,
funding, heat/gross caps, liquidation), `run_swing.py` (frozen config), `validate_swing.py`.
Artifacts in `results/swing_*`.

**Limitations & candor:** 3.85y of 4h data, one full bear tail, one full chop year; costs modeled,
not measured; funding generic (real per-coin funding files not in repo for 4h grid); liquidation
modeled at bar granularity. The OOS year was negative standalone — deploy this as a sleeve, not a
whole book, and only with the blend + leverage limits above.
