# SEISMO — trading liquidation cascades like earthquakes

An intentionally unconventional strategy: treat market-wide liquidation
cascades on Binance USDT-perps as **seismic events**. The main shock is
forced, indiscriminate deleveraging — *flow, not information*. SEISMO never
fades the first quake; it waits for an **aftershock** (a second distinct
shock while the market is still shaken), buys the coins whose crash is
*least explained by their own beta* (the forced-flow overshooters), holds
them ~36 hours with vol-targeted leverage, and is flat >90% of the time.

On top of that sits the most unconventional layer: the strategy paper-trades
itself continuously and only deploys real capital while its own trailing
paper track is profitable — an **anomaly heartbeat monitor**. Anomalies are
born and die; the meta-layer's job is to detect death and pull capital.
It did exactly that here (see the honest OOS section below — this document
reports a decayed anomaly faithfully instead of torturing the data until it
confesses).

---

## 1. Signal architecture

**L1 — the aftershock event engine (directional edge)**

| step | rule |
|---|---|
| universe(t) | perps with ≥3 weeks of history and 30-day median hourly dollar volume ≥ $1M (point-in-time, shifted 1 bar; no survivorship) |
| market factor | cross-sectional **median** hourly return of the universe |
| shock trigger | z-score of the market's 4h return ≤ −2.5 (vs its own trailing 30d) **and** ≥85% of the universe down over 4h **and** ≥15 names in universe |
| **aftershock gate** | trade only if the previous trigger is **24–120 h old**. A shock out of calm is a *main shock* (follow-through risk — skip). A shock 1–5 days after previous activity is an *aftershock*: sellers exhausted, bounce odds highest. |
| selection | rank universe by 4h **residual** = own return − β·market (β = rolling 30d); buy the 5 most negative (deepest forced-flow overshoot) |
| execution | signal at bar close → fill at next bar open; exit at open 36h later; one basket at a time |

**L2 — vol-targeted leverage (size, don't stop).** Per event, notional =
min(3×, 10% risk budget ÷ predicted basket move) of equity. Violent
cascades (LUNA, FTX, Oct-2025) automatically get small size; garden-variety
flushes get full size. Realized average leverage 1.26×, max 2.27×. Stops
were tested and rejected: cascades whipsaw, and an 8% stop reliably sold
local bottoms (IS event PF fell from 2.9 to 1.2).

**L3 — anomaly heartbeat.** Every event is always evaluated on paper. Real
capital deploys only while the mean levered return of the last 20 paper
events is positive. No lookahead: the gate reads only past events.

Costs: 10 bps per side (taker fee + slippage) on every leg.

## 2. Anti-overfitting protocol

- **Hard fence**: data before 2024-07-01 is in-sample (IS); after is
  out-of-sample (OOS). All exploration, diagnostics and tuning used IS only.
- **Design constants, never tuned**: 4h trigger horizon, 30d lookbacks,
  $1M liquidity floor, breadth 0.85, cost assumption, 3× cap.
- **Coarse grid, plateau required**: 324 combos over 6 parameters.
  **100% of the 228 qualifying combos were profitable with event PF > 1 on
  IS** — the edge is a plateau, not a lucky spike. Grid-edge extension
  (hold 48–60h, window 144–168h) decays smoothly, confirming an interior
  optimum.
- **Ablations (IS)**: remove the aftershock gate → Sharpe halves (1.56→0.77)
  and max DD doubles; mirror trade (short the resilient) → destroyed
  (PF 0.36), so the effect is asymmetric, not an artifact; random picks
  instead of residual ranking → base beats 85% of 40 seeds (most alpha is
  the *event timing*, selection adds a modest layer).
- **Walk-forward**: OOS traded in four 6-month folds, parameters re-tuned
  each fold on an expanding window ending before the fold. Chosen params
  were stable across folds (z=2.5, window≈120h, hold=36h every time).
- **Full disclosure of every OOS look**: the OOS window was evaluated
  exactly twice — once for the frozen directional config, once for the
  frozen beta-hedged config. Both are reported below, including the
  failure. The heartbeat gate (L3) was added *after* observing the OOS
  decay; its window was set to the natural default (20) and sensitivity to
  10/30 is reported rather than optimized. Treat L3's OOS numbers with that
  caveat in mind.

## 3. Results

Hourly mark-to-market, compounded, net of 10 bps/side. "Event WR / PF" are
per-basket; a leg is one coin in a basket.

### Headline: the full gated system (L1+L2+L3)

| window | total PnL | CAGR | max DD | Sharpe | event WR | event PF | leg WR | leg PF | events (live/paper) | exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| in-sample (2021→2024-06) | **+454%** | 45.9% | −34.0% | 1.31 | 68.4% | 2.71 | 64.5% | 2.56 | 76 / 97 | 7.1% |
| out-of-sample (2024-07→2026-06) | −28.2% | −15.8% | −46.1% | −0.28 | 43.3% | 0.84 | 44.7% | 0.87 | 30 / 44 | 6.5% |
| **full history** | **+298%** | **23.8%** | **−46.1%** | 0.79 | **61.3%** | **1.78** | 58.9% | 1.79 | 106 / 141 | 6.9% |

$10,000 compounds to ~$39,800 over the full period while being in the market
only ~7% of the time, at an average deployed leverage of 1.26× (max 2.27×,
cap 3×).

![equity](out/equity_full.png)

### The honest part: the anomaly died mid-2024

The ungated L1 edge was outstanding in-sample and **positive in every
calendar year 2021–2024H1** — then inverted:

| configuration | IS | OOS |
|---|---|---|
| L1 directional (frozen) | +1,188%, Sharpe 1.56, PF 2.94, WR 69% | **−39%, Sharpe −0.38, PF 0.81, WR 39%** |
| L1 beta-hedged vs BTC (frozen) | +195%, Sharpe 1.33, PF 3.62, WR 63% | **−56%, Sharpe −1.25, PF 0.53, WR 33%** |
| walk-forward (param-blind) | — | −36%, PF 0.62 |

The hedged failure is the decisive diagnostic: it isn't just "alts kept
falling" — the **residual itself now trends instead of reverting**.
Post-2024 cascade overshooters keep underperforming their beta afterwards
(plausible structural causes: a flood of low-float/high-FDV listings whose
dumps are unlock-driven rather than liquidation-driven, and the
disappearance of the retail snap-back bid in alts).

![histogram](out/event_hist.png)

The heartbeat monitor read the decay in real time and withdrew capital
(gray bands: warm-up in 2021, a pause mid-2025, and full retirement from
late 2025 — using only trailing information):

![heartbeat](out/heartbeat.png)

![oos](out/gated_vs_ungated_oos.png)

### Sensitivity (no cherry-picking)

Gate window (full period / OOS total return): w=10 → +389% / −43%;
**w=20 → +298% / −28%**; w=30 → +307% / −12%. All three preserve the bulk
of IS performance and retire the strategy in OOS; none rescues OOS into
profit, as expected — a trailing gate caps decay losses, it cannot create
alpha.

Costs per side (gated, full period): 5 bps → +421%; **10 bps → +298%**;
15 bps → +223%; 20 bps → +179%. The edge survives a doubling of assumed
frictions.

Leverage cap (gated, full period): 1× → +370% (DD −28%, Sharpe 1.02);
2× → +396% (DD −42%); 3× → +298% (DD −46%); 5× ≡ 3× (the 10% risk budget
rarely asks for more than ~2.3×). The vol-targeting, not the cap, is the
real risk engine; running the cap at 1–2× is the saner deployment.

## 4. Limitations

- Funding payments are not modeled (positions are held ≤36h across at most
  five 8h funding marks; during post-crash hours funding is typically
  negative, i.e. longs get *paid*, so the omission is likely conservative).
- 10 bps/side may understate slippage in the worst cascade minutes for the
  least liquid picks; the 20 bps column above bounds this.
- Entry at next-bar-open is implementable (signal needs only bar-close
  data), but a live system would face occasional halted/delisting symbols;
  the backtest exits those at the last printed open.
- The heartbeat layer was specified after seeing the OOS decay (window not
  tuned, sensitivity disclosed) — its OOS benefit is a structural claim
  (bounded decay losses), not a validated alpha claim.

## 5. Verdict

The aftershock-overshoot anomaly was real, large, and robust for three
years, and it is dead as of this data's end — the system is **correctly
flat today**, and says so itself. That is the actual deliverable: not a
backtest that looks perfect because its failures were quietly tuned away,
but a machine that knows *when it has an edge and when it doesn't*. If the
heartbeat turns positive again (20-event trailing mean > 0), it redeploys
automatically at 1–2× vol-targeted leverage.

## 6. Reproduce

```bash
pip install pandas numpy matplotlib
python3 -m seismo.panel      # build & cache the aligned hourly panel
python3 -m seismo.tune       # in-sample grid (writes out/grid_is.csv)
python3 -m seismo.final      # IS/OOS/walk-forward/ablations/sensitivity
python3 -m seismo.plots      # charts
```

Files: `panel.py` (point-in-time panel), `engine.py` (signals + event
backtester), `diagnose*.py` (IS-only exploration that discovered the
aftershock effect), `tune.py`, `final.py`, `plots.py`; all tables/charts in
`seismo/out/`.
