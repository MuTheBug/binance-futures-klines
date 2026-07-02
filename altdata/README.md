# Alt-data — the orthogonal feeds that can break the OHLCV ceiling

The ECHO Engine is at the honest ceiling of what *price* data can give. The only real way
higher is **orthogonal information**. Binance publishes it with full multi-year history (no API
key): **open interest, top-trader vs retail long/short positioning, taker buy/sell flow**
(from `data.binance.vision`) and **funding rate** (from the public REST API). These download
scripts pull it, resample to daily, and write tidy CSVs you can commit. Then one command tests —
with the same IS-first discipline — whether any of it actually adds alpha.

## 1. Download (run on your machine, where Binance is reachable)
```bash
cd altdata
pip install requests pandas              # if not already installed

# open interest + long/short + taker flow (multi-year, from data.binance.vision)
python3 fetch_metrics.py                  # default: top-40 liquid crypto bases, full history
#   options:  --all                       (every crypto base in the repo — large, hours)
#             --symbols BTC ETH SOL ...    (explicit list)
#             --top 60                     (top-N by liquidity)
#             --start 2022-01-01

# funding-rate history (public REST)
python3 fetch_funding.py                  # same scoping flags
```
- **Resumable:** re-running only fetches days after the last saved row, so these double as
  daily collectors (put them in cron to grow history over time).
- **Expect gaps:** metrics history begins ~2021 and not every symbol has it; the scripts skip
  what's missing. `fetch_metrics.py --top 40` is a few thousand small files — minutes, not hours.
- If `data.binance.vision` or `fapi.binance.com` is blocked in your region, run behind a VPN.

Output (one row per UTC day):
```
altdata/<BASE>_USDT_metrics_1d.csv   date, oi, oi_value, toptrader_acc_ls,
                                     toptrader_pos_ls, global_acc_ls, taker_ls
altdata/<BASE>_USDT_funding_1d.csv   date, funding_sum, funding_mean, n
```

## 2. Push to the repo
```bash
cd ..                                     # repo root
git add altdata/*.csv
git commit -m "Add futures alt-data (open interest, long/short, taker, funding)"
git push origin claude/unconventional-trading-strategy-bim2hm
```
(That's the working branch for this project. The `.py` scripts are already committed; this
just adds the downloaded CSVs.)

## 3. Test whether it adds alpha (after pushing, or locally)
```bash
cd research
python3 altdata_sleeves.py
```
This builds candidate sleeves — ΔOpen-Interest, top-trader L/S (follow "smart money"), global
account L/S (fade the retail crowd), taker flow, funding carry — each market-neutral and
beta-neutral, and **accepts one only if it raises the ECHO blend's in-sample Sharpe** (OOS is
confirmation, never the target). If something passes, it's a genuine new edge and I'll wire it
into `echo_engine.py` and the live bot as Sleeve C. If nothing passes, that's an honest result
too (likely needs more history — keep the collectors running and re-test later).

> Honesty up front: more data ≠ guaranteed more PnL. We hold these to the exact same anti-overfit
> bar that rejected funding-carry, hourly reversal, lottery/low-vol, seasonality, and the candle
> -anatomy signals. The win, if it comes, will be earned — not curve-fit.
