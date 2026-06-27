# ECHO Engine — autonomous Binance Futures bot + Telegram control

Fully autonomous bot that trades the **ECHO Engine** strategy (see
[`../research/ECHO_ENGINE.md`](../research/ECHO_ENGINE.md)) on **Binance USDT-M perpetual
futures**, rebalancing once daily, with a **Telegram** command/alert interface.

It computes the live signal through the *exact same* validated research code
(`strategies.py`, `engine.py`, `echo.py`), so what it trades is what was backtested:
**50% trend (beta) + 50% beta-neutral residual-continuation, vol-targeted, leveraged.**

```
run.py ── bot.py (loop: poll Telegram + daily rebalance + circuit breaker)
            ├─ signals.py   fetch klines  → build matrices → ECHO target weights
            ├─ trader.py    target weights → minimal orders (band/dust/reduce-only)
            ├─ exchange.py  signed Binance USDT-M REST (dry-run + testnet aware)
            ├─ notifier.py  Telegram (raw HTTP long-poll)
            └─ state.py     equity history, peak, flags (state.json)
```

## ⚠️ Read this first
- **Leveraged futures can lose 100% of your capital.** The 2× research config showed a
  **−62% max drawdown**. Position the bot accordingly and never trade money you can't lose.
- The bot ships **DRY-RUN + TESTNET ON**. It will not touch real money until you set
  `DRY_RUN=false` **and** `BINANCE_TESTNET=false`. **Paper-trade on testnet for weeks first.**
- API keys: enable **Futures only**, **disable withdrawals**, and IP-whitelist them.
- Backtests are optimistic. Expect live Sharpe below backtest (slippage, funding, downtime).

## Setup
```bash
cd bot
pip install -r requirements.txt
cp .env.example .env          # then edit .env (keys, telegram, leverage…)

python3 run.py --selftest     # offline logic tests (no network/keys)
python3 run.py --signal       # print today's target weights once (needs Binance reachable)
python3 run.py                # start the autonomous bot
```

### Telegram in 3 steps
1. Create a bot with **@BotFather**, copy the token into `TG_TOKEN`.
2. Leave `TG_CHAT_IDS` empty, start the bot, then send it any message — it **auto-learns**
   your chat id (confirm with `/id`, then paste it into `TG_CHAT_IDS` to lock it down).
3. Send `/help`.

## Commands
| command | action |
|---|---|
| `/status` | mode (dry/live, testnet/main), equity, DD, leverage, next rebalance |
| `/balance` | equity, peak, drawdown |
| `/positions` | open positions + unrealized PnL |
| `/pnl` | PnL / equity summary since start |
| `/report` | live performance (total, ann. pace, DD) vs backtest reference |
| `/signal` | recompute target weights now (no trading) |
| `/weights` | last computed targets |
| `/rebalance` | force a rebalance immediately |
| `/flat` | **panic** — close ALL positions (reduce-only market) |
| `/pause` `/resume` | stop / start auto-rebalancing (resume also clears the breaker) |
| `/leverage <x>` | set the leverage multiplier (capped at `MAX_GROSS`) |
| `/dryrun <on\|off>` | toggle paper mode at runtime |
| `/risk` `/log` `/id` | risk settings · recent log · chat id |

## How it trades
- **Daily** at `REBALANCE_UTC_HOUR:MINUTE` (default 00:05 UTC, just after the daily close),
  it fetches fully-closed daily klines for the liquid USDT-perp universe, computes the ECHO
  target weights, and reconciles the book with the **minimum set of orders** (a no-churn band
  suppresses tiny drifts; closes/trims use `reduceOnly`; side-flips don't, so they fill).
- **Sizing:** `target_notional_i = weight_i × equity`. Weights already include the vol-target
  and the `LEVERAGE` multiplier; total gross is hard-capped at `MAX_GROSS`.
- **Circuit breaker:** before each rebalance it records equity; if drawdown from peak exceeds
  `MAX_DRAWDOWN_STOP`, it **flattens, pauses, and alerts** (latched until `/resume`).

## Run it 24/7 (systemd)
```ini
# /etc/systemd/system/echobot.service
[Unit]
Description=ECHO Engine bot
After=network-online.target
[Service]
WorkingDirectory=/path/to/repo/bot
ExecStart=/usr/bin/python3 run.py
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now echobot && journalctl -u echobot -f
```
State (`state.json`) and `.env` persist across restarts; Telegram offset is saved so commands
aren't reprocessed. `state.json` and `.env` are git-ignored.

## Notes & limits
- **One-way position mode required** (Binance default). On startup the bot checks your
  account; if it's in **Hedge mode** it **blocks live orders** and tells you to switch to
  One-way (Binance app → Settings → Position Mode). It also syncs to Binance server time on
  startup and before each rebalance (avoids timestamp errors), and sends a **daily check-in**
  alert at `SUMMARY_UTC_HOUR`.
- Market orders only (the strategy is daily; majors are liquid). For thin alts, consider
  lowering `MAX_POSITIONS` / raising `MIN_DOLLAR_VOL`.
- Funding costs are real on perps; the backtest models them but live funding varies.
- This is research software provided as-is, **not financial advice.** You are responsible for
  every order it places. Start on testnet, in dry-run, small.
