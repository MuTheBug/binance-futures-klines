# v4 trading bot: Binance USDT-M futures, paper and live

This bot trades the **v4 strategy** from `research/STRATEGY.md` §8: 70% risk-adjusted trend book
(v3) plus 30% BTC-hedged pump-follow sleeve. It trades once a day, a few minutes after the
00:00 UTC daily close.

> **Read this first.** On the only data this strategy had never seen (June–November 2026) it made
> nothing: **$100 → $100 at 1.5× leverage, with a −27% drawdown on the way, while BTC rose 19%.**
> Its backtest on Binance's complete history is good (Sharpe 1.7–2.5), so the true edge is
> uncertain and probably well below the backtest (`research/V5_PROTOCOL.md`). Paper trade. Do not
> go live on the strength of the backtest. Nothing here is financial advice, and you can lose
> everything you put in.

## Quick start

```bash
pip install -r bot/requirements.txt          # numpy, pandas, requests (Python 3.11+)
cp bot/config.example.toml bot/config.toml   # defaults: paper mode on live Binance data

python -m bot -c bot/config.toml targets     # today's target portfolio, trades nothing
python -m bot -c bot/config.toml once        # one paper cycle now
python -m bot -c bot/config.toml run         # scheduler: one cycle per day at 00:05 UTC
python -m bot -c bot/config.toml status
```

Paper mode with `data_source = "binance"` needs **no API keys**: it uses public market data and
simulates fills at the mark price, with fees, slippage and the funding actually paid.

### Testnet, then live

1. Create keys on the Binance Futures **testnet**, set `[exchange] testnet = true` and `mode = "live"`.
   If Binance moves the testnet, set `base_url` to the new URL.
2. `export BINANCE_API_KEY=... BINANCE_API_SECRET=...`
3. `python -m bot -c bot/config.toml --confirm-live once --dry-run` shows the planned orders.
4. `python -m bot -c bot/config.toml --confirm-live once` sends one real cycle. Check fills in the
   Binance UI and in `bot_state/journal.jsonl`.
5. For mainnet: `testnet = false`, keys from a **sub-account** dedicated to the bot, with
   **trading enabled and withdrawals disabled**, IP-restricted. The account must be in
   **One-way** position mode.
6. Run it as a service: `bot/v4bot.service` (systemd). Put keys in `bot/.env` (git-ignored, `chmod 600`).

Every command that can trade live requires `--confirm-live`.

## Running with $100

Use the preset `bot/config.100usd.toml`. It is v4 unchanged, with a $100 paper account at **1.0×**
leverage (lowered from 1.5× after the fresh-data test).

```bash
python -m bot -c bot/config.100usd.toml targets
```
```bash
python -m bot -c bot/config.100usd.toml run
```

**What the evidence says** (`research/V5_PROTOCOL.md`). Every row is the real bot pipeline starting
from $100, on Binance's complete USDT-perp history (every contract ever listed, including delisted
ones), with real funding and Binance minimum order sizes:

| Test | Result |
|---|---|
| **Fresh data, never used in design: 2026-06-08 → 2026-11-09, 1.5×** | **$100 → $100.16**, worst drawdown −27% (August −23%) |
| Same, 1.0× | $100 → $101.07, worst drawdown −18% |
| BTC buy-and-hold over the same months | +19% |
| Design-period check, 2025-01 → 2026-06-07, 1.5× (data the strategy was built on) | $100 → $471, worst drawdown −19% |

- **Backtest Sharpe on the complete market:** 1.68 (2020–24) and 2.46 (2025 – mid-2026).
  **On the fresh five months: 0.27.**
- **How unusual the fresh result is:** it sits in the worst 7% of the backtest's own five-month
  stretches. It was either a bad stretch for trend-following (the book was short when the market
  reversed in August) or the real edge is smaller than the backtest. Five months cannot tell which.
- **The pump-follow part was mostly an artifact.** On complete data its standalone Sharpe is 0.60,
  not 1.36. The old data was missing the coins that pumped and later died.
- **Withdrawn numbers:** earlier versions of this section quoted "$143 median after six months".
  That came from the incomplete 124-coin data.
- **Skipped orders:** at $100, about 25–40% of the target size is skipped because orders fall
  below Binance's minimum.

## Commands

| Command | What it does |
|---|---|
| `run` | Scheduler loop. Trades once per day after the close, retries failures, stops cleanly on SIGTERM |
| `once [--dry-run] [--force]` | One cycle now. `--dry-run` plans without trading; `--force` re-runs a day already traded |
| `targets` | Today's target weights and planned orders (no trading) |
| `status` | Equity, peak, halt state, positions |
| `flatten --yes` | Close every position and halt |
| `resume` | Clear a halt after you've reviewed it (resets the drawdown peak) |
| `replay --start YYYY-MM-DD [--end]` | Offline: run the full pipeline day by day over the repo CSVs |
| `once/targets --as-of YYYY-MM-DD` | Offline (CSV paper mode only): act as if that day just closed |

## What one cycle does

1. **Guards:** a `bot_state/STOP` file pauses the bot. A risk halt blocks trading until `resume`.
   A day already traded is skipped, so restarts never double-trade.
2. **Data:** universe = tradable USDT perpetuals, crypto only (tokenized stocks/metals excluded),
   24h volume prefilter. Fetches 400 **closed** daily bars per symbol. Refuses to trade on stale
   data, e.g. the latest bar missing or the BTC hedge leg missing.
3. **Strategy:** `bot/strategy.py`, a line-for-line port of the research code.
   `tests/test_strategy_parity.py` checks it reproduces the research weights (max diff ~1e-7).
4. **Risk:** equity vs peak (`max_drawdown`) and vs the last cycle (`max_cycle_loss`). A breach
   **flattens everything and halts**.
5. **Orders:** notional = weight × equity × leverage, then the per-symbol cap and gross cap. Sizes
   round toward zero to the lot step, sub-min-notional targets drop to 0, and changes inside the
   rebalance band are skipped. Reduce-only orders go first, and sign flips are split into
   close + open.
6. **Execution (live):** MARKET orders with deterministic `clientOrderId`s. After a timeout or 5xx,
   the bot looks the order up by that id before resending, so it never double-fills. Each symbol's
   margin type and leverage are set once. Optional exchange-side catastrophe stops.
7. **Records:** `state.json` (atomic writes), `journal.jsonl` (every target, order and fill),
   `equity.csv`, `bot.log`. Optional webhook or Telegram notifications.

## Risk controls

| Setting | Default | Notes |
|---|---|---|
| `risk.leverage` | 1.0 | Hard limit 5. Lowered from 1.5 after the fresh-data test (−27% drawdown at 1.5×, −18% at 1×) |
| `risk.max_gross` | 3.0 | Total notional / equity cap |
| `risk.max_position` | 0.50 | One symbol. Inverse-vol sizing sometimes wants more (e.g. TRX); this clips it |
| `risk.max_drawdown` | 0.35 | Flatten + halt. Backtest max DD was ~28% |
| `risk.max_cycle_loss` | 0.20 | Flatten + halt on a 20% one-day loss |
| `risk.stop_loss_pct` | 0 (off) | Exchange-side STOP_MARKET per position. **Not part of the backtest** |

Other safeguards:
- Refuses to start in Hedge Mode.
- Refuses to trade if the account holds positions it didn't open (use a dedicated sub-account).
- Refuses to run a second instance on the same state dir.
- Refuses to mix paper state with live state.

## Verification done

```bash
python3 -m unittest discover -s bot/tests -t .     # 28 tests
python -m bot replay --start 2025-01-01             # full pipeline vs idealized backtest
```

- **Unit tests:** config validation, order planning (rounding, caps, bands, flips, min notional),
  paper accounting (avg entry, flips, fees, funding), client signing/retries/time resync/rate limits,
  exchange-filter parsing, risk checks.
- **Strategy parity** with the research code on 7 dates (max weight diff < 1e-5).
- **End-to-end** against an in-process fake Binance (signed requests verified, orders filled,
  positions tracked): paper and live cycles, restarts, funding, stale-data refusal, STOP file,
  drawdown halt + flatten, the scheduler, idempotent recovery from a 503 after an order filled,
  foreign-position and hedge-mode refusal, single-instance lock.
- **Replay** on the repo's CSVs, 2025-01 → 2026-06 at 1.5x: bot Sharpe 2.04 vs idealized 2.19,
  daily-return correlation 0.993. The gap is from the rebalance band, position caps and lot
  rounding.
- **Full-market replay** (`research/v5_exam.py`): on Binance's complete history with real funding,
  2025-01 → 2026-06-07, the bot pipeline scored Sharpe 2.40 against 2.46 idealized. The fresh-data
  result is in *Running with $100*.

**Not verified here:** connectivity to the real Binance trading API. Binance geo-blocks it from
this build environment (HTTP 451), so the client was tested against the fake exchange built from
Binance's documented formats. Run steps 1–4 of *Testnet, then live* before risking money.

## Known limitations

- **Small accounts:** Binance's minimum order is ~5 USDT (more for BTC/ETH). Below it, positions
  are skipped. At $100 the bot holds about 60–75% of its target size on the full market (see
  *Running with $100*). Every skipped order is tracking error.
- **Missing daily bars:** a coin needs 60 consecutive daily bars to be eligible, so one missing
  bar makes it ineligible for 60 days. The live API has no such holes, but Binance's public archive
  does (2026-10-28), and the offline research repairs it.
- **Daily cadence:** intraday moves are only guarded by the optional exchange stops and by
  liquidation. Cross margin with a 5x exchange setting gives headroom, but a market-wide crash can
  still hit the account before the next cycle.
- **Conditional orders:** Binance has been changing how stop orders are submitted. If stop
  placement errors on testnet, leave `stop_loss_pct = 0` (that matches the backtest).
- **Universe filtering** relies on `underlyingType == "COIN"` plus an exclude list. Check the
  `targets` output for unexpected symbols when Binance lists new non-crypto perps.
