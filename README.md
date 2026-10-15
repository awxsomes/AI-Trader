# Paper implementations: overnight returns and deep learning stat arb

This project implements two finance papers on your daily S&P 500 stock data,
backtests them, and lets you run the second one live on an Alpaca **paper**
(simulated money) account.

1. **Knuteson (2020), "Strikingly Suspicious Overnight and Intraday Returns"**
   (arXiv:2010.01727). Splits each day's stock return into the part earned
   overnight and the part earned during trading hours.
2. **Guijarro-Ordonez, Pelger & Zanotti (2022), "Deep Learning Statistical
   Arbitrage"** (arXiv:2106.04028). Builds market-hedged "residual"
   portfolios, uses a CNN + Transformer to spot short-term patterns in them,
   and trades on those patterns.

---

## 1. What's in the folder

```
papers_impl/
  README.md               this file
  RESULTS.md              numbers from the backtest runs
  common_data.py          loads the *_1d.parquet stock files
  overnight_intraday.py   Paper 1
  tickers.txt             the 496-stock universe
  download_data.py        re-downloads the stock data from Yahoo (only if needed)
  run_paper_trade.bat     one-click daily paper rebalance (for Task Scheduler)
  statarb/                Paper 2 research code
    residuals.py            step 1: PCA factor model -> residual returns
    models.py               step 2: OU, Fourier+FFN, CNN+Transformer models
    backtest.py             step 3: rolling out-of-sample backtest
  live/                   Paper 2 live paper trading
    live_common.py          shared live logic (data, signals, weights)
    train_live.py           trains the model on recent data and saves it
    paper_trade.py          computes today's targets and trades on Alpaca
    model_cnn_trans.pt      the saved model (created by train_live.py)
    logs/                   daily targets, order plans, errors
  output/                 figures and CSVs written by the scripts
```

## 2. Setup (once per computer)

```powershell
pip install pandas pyarrow numpy matplotlib torch yfinance alpaca-py
```

The research scripts read your stock files from the folder given by the
`TRADING_DATA_DIR` environment variable. On this laptop:

```powershell
$env:TRADING_DATA_DIR = "C:\Users\alche\Documents\trading2\data"
```

That line only lasts for the current PowerShell window. To make it
permanent, run `setx TRADING_DATA_DIR "C:\Users\alche\Documents\trading2\data"`
once and open a new window.

Run everything from inside the `papers_impl` folder.

---

## 3. How each part works

### 3.1 Paper 1: overnight vs intraday returns

```powershell
python overnight_intraday.py        # about 1-2 minutes
```

**Idea.** A stock's daily return can be split into two pieces:

- overnight: from yesterday's close to today's open, `Open(t) / Close(t-1) - 1`
- intraday: from today's open to today's close, `Close(t) / Open(t) - 1`

Knuteson shows that, for most world stock indices, almost all of the
long-run gain happened overnight while the intraday part was flat or
negative, and argues this is hard to explain innocently.

**What the script does.** For every stock it compounds the two pieces
separately over the whole history, then does the same for an
equal-weighted average of all stocks.

**Important data fix.** Old Yahoo data has fake opening prices. Before
about 2000, most days have the open set equal to the previous close, which
quietly moves the whole day's return into the intraday piece.
`common_data.first_valid_open_date()` starts each stock only from the date
its opening prices look real. Without this the results are meaningless.

**Outputs.** `output/fig_overnight_intraday_grid.png` (21 well-known
stocks), `output/fig_overnight_intraday_market.png` (all-stock average),
`output/overnight_intraday_summary.csv` (one row per stock).

**What we found.** Individual stocks show the paper's dramatic split
(AAPL: overnight +84,109% vs intraday +39%). Averaged across all stocks,
overnight only dominates in 2020-2026, which matches Knuteson's own note
that the US pattern faded after 2008. Details in `RESULTS.md`.

### 3.2 Paper 2: deep learning statistical arbitrage

```powershell
python statarb\residuals.py         # step 1, about 5-10 minutes
python statarb\backtest.py          # steps 2-3, about 30-90 minutes
```

Statistical arbitrage bets that similar stocks which drift apart will move
back together. The paper breaks this into three steps.

**Step 1: residual portfolios** (`statarb/residuals.py`). Most of a stock's
daily move is explained by a few market-wide forces (the market itself,
sectors, and so on). Every day we estimate those forces with PCA:

- take the last 252 days of returns for every stock with complete history,
- find the 5 strongest common patterns (principal components of the
  correlation matrix); each becomes a "factor" portfolio of stocks,
- measure each stock's exposure to the 5 factors using the last 60 days,
- the stock's **residual** for the next day is its return minus what the
  factors explain.

A residual is the return of a tradeable portfolio: long the stock, short
a mix of stocks that copies its factor exposures. It is hedged against
the market and sector swings, so it should wander around zero and revert.
Everything is computed using only information available the day before
(no look-ahead). Check: residuals have an average pairwise correlation of
0.04, versus 0.40 for raw returns.

**Step 2: signals** (`statarb/models.py`). For each stock, take its last 30
residuals and add them up into a little "price path". Three models turn
that path into a trading weight (positive = buy the residual, negative =
sell it):

| Model | How it reads the 30-day path |
|---|---|
| OU + Threshold | Fits a mean-reversion model; buys when the path is unusually low, sells when unusually high. The classic textbook rule. |
| Fourier + FFN | Breaks the path into waves of different lengths (FFT) and feeds them to a small neural network. |
| CNN + Transformer | The paper's model. A small convolutional network spots local shapes (up-ticks, reversals); a transformer relates those shapes across the 30 days; a final layer outputs the weight. |

**Step 3: training and trading** (`statarb/backtest.py`). The neural
networks are not trained to predict returns. They are trained to directly
maximize the **Sharpe ratio** of the trading strategy they produce. Each
day the weights are scaled so the absolute weights add up to 1 (a leverage
limit). The models are trained on the previous 1000 days, used for the
next 250 days, then retrained on a moving window, so every reported day is
genuinely out of sample (2003-2026).

**Outputs.** `output/statarb_performance.csv`,
`output/statarb_oos_returns.csv`, `output/fig_statarb_cumret.png`,
`output/fig_statarb_example.png`.

**Your results** (out of sample 2003-2026, no trading costs):

| Model | Sharpe | Annual return | Annual volatility |
|---|---|---|---|
| CNN + Transformer | 2.60 | 5.8% | 2.2% |
| Fourier + FFN | 1.53 | 2.9% | 1.9% |
| OU + Threshold | 0.55 | 2.0% | 3.7% |
| Equal-weight residuals | 1.19 | 1.1% | 0.9% |

The paper reports 3.36 / 1.98 / 0.73 for the same three models. The
ranking and the size of the gaps match.

**Where we simplified the paper** (to run on a laptop): retraining every
250 days instead of 125, fewer training epochs, mini-batch training, and
today's S&P 500 members instead of a historical member list. That last
point (survivorship bias) makes all the backtest returns look better than
they would have been in real time.

---

## 4. Live paper trading on Alpaca

### 4.1 How it works each day

`live/paper_trade.py` runs the same three steps on today's data:

1. downloads the last ~13 months of daily prices for the 496 stocks from
   Yahoo Finance (during market hours, the latest bar is today's price so
   far),
2. computes the last 30 days of residuals and today's factor model,
3. runs the saved CNN + Transformer to get a weight per residual portfolio,
4. converts residual weights into **stock** weights. Holding a residual
   means holding the stock and hedging its factor exposure with other
   stocks; these hedges are netted across all residuals
   (`w_stocks = (I - W B) w_residuals` in `statarb/residuals.py`), and the
   weights are scaled so the absolute values add to 1,
5. turns weights into share quantities for your account size and sends
   market orders to Alpaca to move from the current positions to the
   targets.

Orders go in two phases: first everything that reduces or closes a
position, then (after those fill) everything that opens or adds. Alpaca
won't let one order flip a position from long to short, so a flip becomes
"close" in phase 1 and "open the other side" in phase 2.

Practical details the script handles for you:

- **Paper only.** The Alpaca client is created with `paper=True` and there
  is no option to change it.
- **Dry run by default.** Without `--execute` it only prints and saves the
  plan.
- **Small positions.** On a $100k account the average position is about
  $200, so longs use fractional shares. Shorts must be whole shares, so
  they're rounded to the nearest share.
- **Shorting limits.** Names Alpaca marks as not shortable or hard to
  borrow are skipped for shorts, and the remaining weights are rescaled.
- **Tiny trades** under $25 are skipped (`--min-trade` changes this).

**Why the book is net long.** The weights hedge out the 5 PCA factors (the
first one is essentially the market), so the portfolio is roughly
market-neutral in risk even though it holds more dollars long than short.
Expect roughly +20-30% net long in dollars.

### 4.2 One-time setup

1. Create a free account at alpaca.markets, open the **Paper Trading**
   account, and generate API keys for it.
2. Save the keys as permanent environment variables (run once, then open
   a new PowerShell window):

   ```powershell
   setx ALPACA_API_KEY "your-paper-key-id"
   setx ALPACA_SECRET_KEY "your-paper-secret"
   ```

   Keep these out of any file you share or commit to GitHub.

3. Train the live model on the most recent data (about 10-20 minutes):

   ```powershell
   python live\train_live.py
   ```

   This downloads ~7 years of prices from Yahoo, computes residuals, trains
   on the last 1000 trading days, and saves `live\model_cnn_trans.pt`.
   Re-run it about once a month. If Yahoo downloads fail, you can train on
   your local files instead:
   `python live\train_live.py --source local`
   (they only go up to the date you last updated them).

### 4.3 Running it

Do a dry run first. It works even before you add Alpaca keys:

```powershell
python live\paper_trade.py
```

You'll see the number of longs and shorts, dollar exposure, how much will
be traded, and the largest positions. The full plan is saved in
`live\logs\<date>_targets.csv` and `<date>_orders.csv`.

When it looks sensible, place the paper orders:

```powershell
python live\paper_trade.py --execute
```

**When to run:** once per trading day, between **3:30 and 3:50 PM US
Eastern**. The model was trained on close-to-close returns, so trading
just before the close keeps live trading close to the backtest. If the
market is closed the script refuses to trade; `--allow-closed` queues the
orders for the next open instead (worse match to the backtest).

Useful options:

| Option | Meaning |
|---|---|
| `--execute` | actually place the paper orders |
| `--gross 0.5` | use half the account (default 1.0 = gross exposure equal to equity) |
| `--min-trade 50` | skip trades smaller than $50 |
| `--allow-closed` | queue orders for the next open when the market is closed |
| `--source local` | use your local parquet files instead of Yahoo (only for testing; they're out of date) |

### 4.4 Running it automatically every day

`run_paper_trade.bat` runs the rebalance with `--execute` and appends the
output to `live\logs\scheduler.log`. To schedule it:

1. Open **Task Scheduler**, then **Create Basic Task**.
2. Trigger: **Weekly**, Monday to Friday, at **3:35 PM** (if your laptop
   isn't on Eastern time, convert).
3. Action: **Start a program**, and browse to `run_paper_trade.bat`.
4. In the task's properties, tick "Run whether user is logged on or not"
   only if you want it to run with the laptop locked; the laptop must be
   on and awake either way.

On market holidays the script sees the market is closed and exits without
trading.

### 4.5 What to expect, honestly

- **Trading costs are the big unknown.** The backtest ignored them. The
  strategy turns over a large part of the book every day (the first day
  is 100%), and on a 5.8%-a-year strategy even small costs matter.
  Alpaca charges no commission, but you still pay the bid-ask spread.
  Paper trading fills at quoted prices and understates this.
- **Paper fills are optimistic** in general (no market impact, easy
  shorting).
- **Day-to-day results are noisy.** Even at a Sharpe of 2.6, a month can
  easily be negative. Judge it over several months.
- **This is a research replication,** not a tested trading system. Don't
  move it to a real-money account without much more testing.

Compare paper performance to the backtest by looking at the account's
equity curve on the Alpaca dashboard, and to what the model intended in
`live\logs`.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `No *_1d.parquet stock files found` (or `KeyError: 'ticker'`) | `TRADING_DATA_DIR` doesn't point at the stock folder; see section 2. |
| `No trained model at ...` | Run `python live\train_live.py` first. |
| `WARNING: price data is stale` | You used `--source local`, or Yahoo didn't return today's data. Use the default Yahoo source. |
| Yahoo download errors or empty data | Yahoo rate-limits sometimes; wait a few minutes and retry. |
| Some orders rejected | See `live\logs\<date>_failed_orders.csv`. Usually a short Alpaca won't allow that day; the rest of the book still trades. |
| `Set ALPACA_API_KEY and ALPACA_SECRET_KEY` | Do section 4.2 step 2, then open a new PowerShell window. |

## 6. References

- B. Knuteson, "Strikingly Suspicious Overnight and Intraday Returns,"
  arXiv:2010.01727 (2020).
- J. Guijarro-Ordonez, M. Pelger, G. Zanotti, "Deep Learning Statistical
  Arbitrage," arXiv:2106.04028 (2022).
- M. Avellaneda, J.-H. Lee, "Statistical Arbitrage in the US Equities
  Market," Quantitative Finance (2010), for the PCA residual construction.
