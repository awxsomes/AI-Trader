# First-hour chart-pattern trader

Hard-codes the patterns from the Fidelity / Kirkpatrick deck *Identifying
Chart Patterns with Technical Analysis* as exact rules on daily bars, then
trades their breakouts in the first hour of each session (9:30 to 10:30 ET)
on a dedicated Alpaca **paper** account.

| File | What it does |
|---|---|
| `detect.py` | The 31 patterns as rules: swing-point patterns (double/triple tops and bottoms, head and shoulders, rectangles, triangles, wedges, cup and handle), candlesticks (doji, harami, hammer, hanging man, inverted hammer, shooting star, engulfing, piercing line, dark cloud) and short-term patterns (flags/pennants, explosion gaps, pipe bottoms/tops, inside bars, NR4). Each gives a breakout level, stop and target. No look-ahead: a swing high only counts once 3 lower highs follow it. |
| `pattern_trade.py` | The live program. Scans yesterday's charts, watches prices for the first hour, enters breakouts with bracket orders (entry + target + stop). |
| `backtest_patterns.py` | Tests each pattern on 25 years of local daily data, net of costs, against random breakouts with the same geometry. |
| `run_patterns.bat` | For Windows Task Scheduler. |

## How a trade works

1. **Before the open**: every armed setup is listed in
   `patterns\logs\<date>_setups.csv` (symbol, pattern, side, trigger, stop,
   target, date the pattern completed).
2. **9:30 to 10:30**: prices are polled every 20 s. When a stock crosses its
   trigger, the program sends a bracket order: a marketable limit entry, a
   take-profit at the pattern's target and a stop-loss. Two-sided patterns
   (rectangle, triangles, doji, inside bar, NR4, harami) trade whichever side
   breaks first. If a stock opens past its trigger by more than half the
   distance to the target, it is skipped rather than chased.
3. **10:30**: unfilled entries are cancelled. By default positions are kept
   (`--exit hold`): Alpaca keeps working the stop and target (GTC) after the
   program stops, and anything still open after 10 sessions is closed at a
   later open. `--exit flatten` closes everything the program opened at
   10:30 instead (note the backtest measures the hold version).

Sizing: each trade risks 0.5% of equity at its stop, no position over 10% of
equity, at most 20 positions, total exposure at most 1x equity, and at most
2 new trades per pattern per day (so the common patterns, like NR4 and doji,
can't crowd out the rare ones during the test).

## Setup (switching the old benchmark account over)

1. On the Alpaca dashboard, open the paper account you are retiring,
   **reset it** (or close its positions), and generate new API keys for it.
2. Remove that account from whatever runs `live\paper_trade.py` for it, and
   move its `live\state\<account number>.json` file out of `live\state\`
   (the pattern program refuses to trade an account that has one).
3. Save the keys under the pattern program's own names (then open a new
   PowerShell window):
   ```
   setx PATTERN_ALPACA_API_KEY "the-key-id"
   setx PATTERN_ALPACA_SECRET_KEY "the-secret"
   ```
4. Dry runs:
   ```
   python patterns\pattern_trade.py            # today's setups only
   python patterns\pattern_trade.py --watch    # during 9:30-10:30: shows the orders it would send
   ```
5. Go live (paper): either leave it running,
   `python patterns\pattern_trade.py --execute --loop`,
   or schedule `patterns\run_patterns.bat` in Task Scheduler for weekdays at
   9:20 AM (the computer must be awake).

## Backtest first

```
python patterns\backtest_patterns.py
```
Uses `TRADING_DATA_DIR` like the rest of the repo. For each pattern it
reports trades, win rate, mean net return, the same for random breakouts
(same stock, direction, trigger distance, stop/target distances and holding
limit, on random days) and the difference (`edge_bp`) with its t-statistic,
separately before 2015 and from 2015 on. It ends by listing the patterns
with t > 2 in **both** periods. Trade only those:
```
python patterns\pattern_trade.py --execute --patterns double_bottom,cup_handle
```
Checked on simulated random-walk prices (where no pattern can work): edges
average zero and no pattern passes both periods, so a pattern that passes on
real data is unlikely to be an artifact of the test itself.

Limits: daily bars can't show whether a breakout happened in the first hour,
so the backtest measures breakouts at any time of day; the universe is
today's S&P 500 (survivorship bias in absolute returns, not in the
pattern-vs-random difference); and the parameters in `detect.P` were set
from the deck, not tuned. If you change them, look only at the before-2015
results while doing it.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--exit hold/flatten` | hold | keep positions with stop/target, or close at the window's end |
| `--window-minutes` | 60 | trading window after the open |
| `--patterns` | all | comma list of names from `detect.ALL_PATTERNS` |
| `--risk` | 0.005 | equity risked per trade |
| `--max-positions` | 20 | open positions at most |
| `--max-per-pattern` | 2 | new trades per pattern per day |
| `--max-hold-days` | 10 | close held positions after this many sessions |
| `--feed iex/sip` | iex | live price feed (free accounts: iex) |
| `--source alpaca/yahoo/local` | alpaca | daily bars for the scan (falls back to Yahoo) |
