# Results of this run

Data: local parquet files (adjusted daily OHLCV, ~496 current S&P 500
constituents). All performance numbers are out of sample and before
transaction costs. Survivorship bias caveat: the universe is today's
constituents, so levels are optimistic relative to a point-in-time
universe; model comparisons are still like-for-like.

## Paper 1: Knuteson (2020), overnight vs intraday

After filtering synthetic opens (see README), per-ticker results show the
paper's striking splits at the single-stock level (AAPL overnight +84,109%
vs intraday +39%; CSCO +21,290% vs -41%; GE +20,421% vs -8%; PG the
mirror image: -85% vs +120,628%). Across all ~495 tickers, 51% have
overnight > intraday. The equal-weighted cross-section:

| Period     | Overnight | Intraday |
|------------|-----------|----------|
| 1993-2007  | +302%     | +432%    |
| 2008-2019  | +116%     | +182%    |
| 2020-2026  | +113%     | +38%     |

So on this survivor large-cap universe the aggregate overnight dominance
the paper shows for indices/ETFs appears only in 2020-2026, consistent
with Knuteson's own note that the US index divergence stopped around 2008
(and his data starts in the 1990s, where this dataset's opens are
synthetic and had to be excluded).

## Paper 2: Deep Learning Statistical Arbitrage, PCA-5 residuals

Out of sample 2003-01 .. 2026-08 (5,935 trading days), daily rebalancing,
leverage constraint sum|w| = 1, Sharpe-ratio training objective:

| Model        | Ann. Sharpe | Ann. mean | Ann. vol |
|--------------|-------------|-----------|----------|
| CNN+Trans    | **2.72**    | 5.8%      | 2.2%     |
| Fourier+FFN  | 1.51        | 2.9%      | 1.9%     |
| OU+Thresh    | 0.55        | 2.0%      | 3.7%     |
| EqualWeight  | 1.19        | 1.1%      | 0.9%     |

Paper's corresponding PCA 5-factor column (1998-2016, CRSP top ~550):
CNN+Trans 3.36, Fourier+FFN 1.98, OU+Thresh 0.73. This run reproduces

- the ordering and the roughly 2x Sharpe gap between CNN+Trans and
  Fourier+FFN, and the 4-5x gap over the parametric OU rule,
- the paper's qualitative Figure 5: CNN+Trans grows almost monotonically,
  is not hurt by 2008 or 2020 (it gains in both), while the parametric
  model flattens out after ~2012,
- the finding that the signal-extraction step (not the allocation step or
  the factor count) is what separates the models.

Lower absolute Sharpe than the paper is expected here: ~25x less training
compute (fewer epochs, warm starts, 250- instead of 125-day retraining, no
hyperparameter validation), a later and harder sample (2003-2026 vs
1998-2016, with post-2016 arbitrage decay), and ~460 instead of ~550
stocks with no point-in-time universe.

## Paper 2, full version: IPCA factors, stock-space trading, trading costs

Added October 2026 (`full/`). Same data, out of sample 2002-02 to 2026-08.
Every model is evaluated **net of the paper's trading frictions** (Section
III.J): 5 bp per unit of daily turnover plus 1 bp per day on short positions.
Weights are mapped from residual portfolios to actual stocks and normalized
so that sum |w| = 1, as in the paper's main tables.

| Model | Sharpe before costs | Sharpe after costs | Return after costs | Sharpe after costs, 2021-26 | Turnover per day |
|---|---|---|---|---|---|
| CNN+Trans, PCA, cost-trained, **20%/day partial rebalance** | 1.50 | **0.75** | **+2.1%/yr** | **0.69 (+2.3%/yr)** | 9% |
| CNN+Trans, PCA, cost-trained | 2.19 | 0.64 | +2.1%/yr | 0.32 | 30% |
| CNN+Trans, IPCA, cost-trained | 1.64 | 0.07 | +0.3%/yr | -0.84 | 47% |
| CNN+Trans, IPCA, trained without costs | 2.72 | -3.38 | -8.5%/yr | -5.33 | 112% |
| CNN+Trans, PCA, trained without costs (the original live model) | 2.81 | -3.80 | -8.7%/yr | -6.76 | 110% |
| OU+Threshold, IPCA | 0.67 | -3.15 | -10.9%/yr | -3.99 | 95% |
| OU+Threshold, PCA | 0.59 | -3.86 | -11.7%/yr | -4.62 | 97% |

Paper (Table IX, IPCA 5 factors, with frictions): Sharpe 1.11 after costs.

Findings:

- **Costs decide everything.** Models trained without costs have the
  strongest raw signal (Sharpe about 2.8, comparable to the original
  replication) but trade more than the whole book every day, and lose
  8-9% a year after costs. This is the model the live bots ran until
  October 29, 2026, and it explains their steady losses.
- **Training with costs in the objective** cuts turnover to 30% a day and is
  the only approach that stays profitable after costs.
- **Partial rebalancing** (each day move 20% of the way from current holdings
  to the model's target) cuts turnover to 9% a day, raises the net Sharpe to
  0.75, and helps most in recent years: 2023 +4.0%, 2024 +3.2%, 2025 +7.5%
  after costs. Worst drawdown -6%; 7 losing years out of 25. The result is
  flat for steps between 10% and 50% a day, so it does not hinge on one
  setting. This works because the signal decays slowly (paper Section III.M).
- **IPCA underperforms PCA here**, unlike in the paper. The likely reason:
  only 13 of the paper's 46 characteristics can be built from price and
  volume data; the accounting-based ones (value, profitability, investment)
  are missing.
- Reruns reproduce every number exactly (fixed seeds).

Caveats: one random seed per model; the best setup was picked among 24
variants evaluated on the same period, so its numbers are somewhat
optimistic; survivorship bias as above; costs are the paper's simple linear
model, which ignores market impact.

Figures: `output/full/fig_full_best_net.png` (2002-2026) and
`output/full/fig_full_best_net_2021.png` (2021-2026). Tables:
`output/full/summary_table.csv`, `execution_study.csv`,
`net_returns_by_year.csv`.
