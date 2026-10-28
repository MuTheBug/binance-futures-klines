# v4 trading bot: Binance USDT-M futures, paper and live

This bot trades the **v4 strategy** from `research/STRATEGY.md` §8: 70% risk-adjusted trend book
(v3) plus 30% BTC-hedged pump-follow sleeve. It trades once a day, a few minutes after the
00:00 UTC daily close.

> **Read this first.** The backtest Sharpe (~2.0) is inflated by survivorship bias and a
> contaminated out-of-sample window (§8 caveats). Expect live results to be worse. Paper trade for
> 1–2 months, then start live small on a **dedicated sub-account** at ≤ 2× leverage. Nothing here
> is financial advice, and you can lose everything you put in.

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
| `risk.leverage` | 1.5 | Hard limit 5. Research: growth-optimal ~2–4x *if* the backtest edge is real; ruin risk beyond ~6x |
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
python3 -m unittest discover -s bot/tests -t .     # 24 tests
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
- **Replay** 2025-01 → 2026-06 at 1.5x: bot Sharpe 2.04 vs idealized 2.19, daily-return correlation
  0.993. The gap is from the rebalance band, position caps and lot rounding.

**Not verified here:** connectivity to the real Binance API. This build environment cannot reach
Binance, so the client was tested against the fake exchange built from Binance's documented
formats. Run steps 1–4 of *Testnet, then live* before risking money.

## Known limitations

- **Small accounts:** Binance's minimum order is ~5 USDT (100 for BTC). With $60 at 1.5x, many
  positions fall below it and get skipped, so the account won't track the strategy. Use ≥ $1–2k.
- **Daily cadence:** intraday moves are only guarded by the optional exchange stops and by
  liquidation. Cross margin with a 5x exchange setting gives headroom, but a market-wide crash can
  still hit the account before the next cycle.
- **Conditional orders:** Binance has been changing how stop orders are submitted. If stop
  placement errors on testnet, leave `stop_loss_pct = 0` (that matches the backtest).
- **Universe filtering** relies on `underlyingType == "COIN"` plus an exclude list. Check the
  `targets` output for unexpected symbols when Binance lists new non-crypto perps.
