# Results of this run

Data: local parquet files (adjusted daily OHLCV, ~496 current S&P 500
constituents). All performance numbers are out of sample and before
transaction costs. Survivorship bias caveat: the universe is today's
constituents, so levels are optimistic relative to a point-in-time
universe; model comparisons are still like-for-like.

## Paper 1 — Knuteson (2020), overnight vs intraday

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
the paper shows for indices/ETFs appears only in 2020-2026 — consistent
with Knuteson's own note that the US index divergence stopped around 2008
(and his data starts in the 1990s, where this dataset's opens are
synthetic and had to be excluded).

## Paper 2 — Deep Learning Statistical Arbitrage, PCA-5 residuals

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
