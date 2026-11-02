# Grid bot: Binance USDT-M futures, paper and live

A grid bot for Binance USDT-margined perpetual futures, built around the research in
`research/GRID_STRATEGY.md`. It runs a geometric grid of post-only limit orders on one or a few
coins and re-centres or stops when the price leaves the range. It is designed to run safely on a
small ($100) account.

> **Read `research/GRID_STRATEGY.md` before risking money.** In this repo's data, classic grids
> lost money or merely tracked the market, and no grid design passed the pre-registered test.
> A grid is a short-volatility bet: it earns small amounts in sideways markets and gives them back,
> and more, in trends. Paper trade first. Nothing here is financial advice.

## Default strategy and what to expect

The defaults (`config.example.toml`) are the design that ranked first among 120 tested:
- **Grid:** neutral, spacing = 6 × the coin's hourly volatility over the last 7 days (≈ a 36-hour
  move), 10 levels a side, 1x exposure.
- **Trend filter:** it runs only while the coin's daily trend is weak (v3 trend strength within
  ±0.3).
- **Exit:** if the price leaves the range, it closes and waits 24 hours.
- **Rotation:** watches 8 liquid alts and runs at most 2 grids at a time, each with half the account.

| Backtest (1x, after fees and funding) | Sharpe | Return/yr | Max DD |
|---|---|---|---|
| 2022-06 → 2024-12, design coins | 1.53 | +7.6% | −3.6% |
| 2022-06 → 2024-12, coins held out of design | 1.62 | +10.8% | −4.3% |
| **2025-01 → 2026-06, out of sample** | **0.07** | **+0.3%** | −7.9% |

**For a $100 account that means roughly ± $10 a year.** The edge faded to nothing in 2025–26,
although the design did avoid the alt crash (single-coin $100 accounts ended at a median of $101,
where holding the same coins left $23). The numbers also depend on intrabar price order that hourly
data cannot resolve. Paper trading will show which way it goes. **Leverage (`exposure` > 1) widens
the outcomes but did not raise the median.** See `research/GRID_STRATEGY.md` §5–9.

## Quick start (paper, no API keys needed)

```bash
pip install -r bot/requirements.txt
cp gridbot/config.example.toml gridbot/config.toml
python -m gridbot -c gridbot/config.toml plan     # the grid it would place right now
python -m gridbot -c gridbot/config.toml run      # paper-trade on live Binance prices
python -m gridbot -c gridbot/config.toml status
```

Paper mode keeps a simulated futures account in `gridbot_state/state.json`. It fills orders from
live 1-minute prices, including the wicks between polls, charges maker/taker fees and slippage,
and applies the real funding rate every 8 hours.

## Live (testnet first)

1. Use a **dedicated sub-account** with futures enabled, **One-way** position mode, and API keys
   with trading on and withdrawals off, IP-restricted.
2. `export BINANCE_API_KEY=... BINANCE_API_SECRET=...`, then in the config set `mode = "live"`
   and `[exchange] testnet = true`.
3. `python -m gridbot -c gridbot/config.toml plan` shows the levels, per-order size and range.
4. `python -m gridbot -c gridbot/config.toml --confirm-live once` runs one cycle. Check the orders in
   the Binance UI.
5. `python -m gridbot -c gridbot/config.toml --confirm-live run` runs it, ideally as a service
   (see `bot/v4bot.service` for a systemd template). Set `testnet = false` only after testnet looks
   right.

## How it works

Every `poll_seconds` (default 10), for each symbol:

1. **Prices:** the last traded price (live), or the latest 1-minute path (paper).
2. **Risk:** account drawdown from peak and loss since 00:00 UTC. A breach cancels every grid
   order, closes every position and halts until `resume`. A `STOP` file in the state dir pauses.
3. **Rotation:** with `max_active` > 0, a coin without a grid may only start one while fewer than
   `max_active` grids are running. Each grid is sized at `equity / max_active`.
4. **Grid creation:** when there is no grid, the strategy picks the spacing (`k * hourly
   volatility` or fixed), the bias (neutral, long, short, or by daily trend) and whether to run at
   all (trend filter). The grid is centred on the current price and sized so its largest possible
   inventory is `exposure * capital`. If the account is too small for Binance's minimum order
   (~5 USDT), it uses fewer levels. It refuses a coin whose lot size would inflate the planned
   exposure by more than 25%.
5. **Range exit:** when the price is beyond the range by `exit_buffer` levels, the bot re-centres
   (`reset`), closes and cools down (`stop`), or keeps the grid (`hold`). More than
   `max_resets_per_day` exits on a symbol triggers a cool-down (whipsaw guard).
6. **Reconcile:** the bot maps the **real position** to the nearest grid level. If the price
   passed levels whose orders never filled (gaps, downtime, rejected post-only orders), it catches
   up with a market order. Then it makes the resting orders exactly right: post-only (GTX) buys
   below, sells above.

Because the bot's state is derived from the exchange's own position and orders each cycle,
restarts, crashes and missed fills are self-correcting. A restart re-places nothing that is
already correct.

## Commands

| Command | What it does |
|---|---|
| `plan` | Shows what grid would be placed now: spacing, bias, levels, USDT per order, range |
| `once` | Runs one cycle |
| `run` | Loops every `poll_seconds`; stops cleanly on Ctrl-C/SIGTERM |
| `status` | Shows equity, peak, halt state, positions and grids |
| `flatten --yes` | Cancels grid orders, closes positions and halts |
| `resume` | Clears a halt and resets the drawdown peak |
| `replay --symbol S --start D [--end D] [--balance B]` | Runs the real engine offline over the repo's hourly CSVs |

Live trading needs `mode = "live"`, keys in the environment, and `--confirm-live` on every command
that trades.

## Tests

```bash
python3 -m unittest discover -s gridbot/tests -t .
```

What the tests cover:
- **Parity:** the production engine plus the exchange simulator reproduces the research simulator
  (`research/gridsim.py`) to within 0.02% of equity. This covers long, neutral and short grids,
  the reset, hold and stop exit rules, a random walk, and real SOL data including an 18% wick. It
  also covers the default design itself (trend filter + stop + 24h cool-down) on 6 months of real
  DOGE data.
- **Live path:** the real HTTP client runs against a fake Binance server, covering signing,
  post-only orders, fills, opposite orders and an idempotent restart.
- **Robustness:** catch-up after downtime, post-only rejections, stop and cool-down, small-capital
  sizing (including coarse lot sizes), paper-state persistence, and rotation slots.
- **Risk:** drawdown and daily-loss halts that flatten everything, the STOP file and the whipsaw
  guard.
- **Unit:** grid geometry, the exchange simulator's rules, and config validation.

**Not verified here:** the real Binance API, because this build environment cannot reach it. The
fake server follows Binance's documented formats. Use testnet before mainnet.
