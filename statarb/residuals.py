"""Arbitrage-portfolio (residual) construction for Guijarro-Ordonez,
Pelger & Zanotti (2022), "Deep Learning Statistical Arbitrage"
(arXiv:2106.04028), Section II.A / III.B, PCA factor family.

Follows the paper's rolling-window procedure (their PCA spec, which is the
Avellaneda-Lee construction):
  - at each day t, consider only stocks with no missing daily returns over
    the past 252 trading days,
  - estimate the correlation matrix of returns over those 252 days and
    extract the top-K eigenvectors; factor-mimicking portfolio weights are
    eigenvector entries scaled by inverse vol (so factors are tradeable
    portfolios of the stocks),
  - estimate each stock's loadings on the K factors by OLS over the last
    60 days,
  - the day-t residual is computed OUT OF SAMPLE:
        eps_{n,t} = R_{n,t} - beta_{n,t-1}' F_t ,
    where both the factor weights and loadings use information up to t-1
    only (no look-ahead).

Output: a Date x Ticker DataFrame of daily residuals (NaN when the stock
is not in the eligible universe that day), saved to parquet.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common_data as cd

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output")
os.makedirs(OUT_DIR, exist_ok=True)

K_FACTORS = 5          # number of PCA factors (paper's representative choice)
CORR_WINDOW = 252      # window for the correlation matrix / PCA
BETA_WINDOW = 60       # window for the loading regressions
MIN_STOCKS = 100       # minimum cross-section size to compute residuals


def fit_factor_model(X: np.ndarray, k: int = K_FACTORS,
                     beta_win: int = BETA_WINDOW):
    """Fit the PCA factor model on a (corr_win x n) block of past returns
    with no missing values.

    Returns
      W : (n, k) factor-mimicking portfolio weights, F_t = R_t @ W
      B : (k, n) loadings, so the residual of a new return vector r is
          eps = r - (r @ W) @ B = r @ (I - W B)
    """
    corr_win = X.shape[0]
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1e-12
    Z = (X - mu) / sd
    C = (Z.T @ Z) / (corr_win - 1)
    # top-k eigenvectors of the correlation matrix
    vals, vecs = np.linalg.eigh(C)
    V = vecs[:, -k:][:, ::-1]                         # (n, k), descending
    W = V / sd[:, None]                               # factor-mimicking weights
    # OLS loadings over the beta window (no intercept): B = (F'F)^-1 F'Xb
    Xb = X[-beta_win:]
    F = Xb @ W
    B = np.linalg.solve(F.T @ F + 1e-10 * np.eye(k), F.T @ Xb)   # (k, n)
    return W, B


def residual_to_stock_weights(w_eps: np.ndarray, W: np.ndarray,
                              B: np.ndarray) -> np.ndarray:
    """Map weights on residual portfolios to weights on the stocks
    (the paper's w^R = Phi' w^eps, before L1 normalization).

    Holding w_R in stocks earns exactly w_eps @ eps on the next day.
    """
    return w_eps - W @ (B @ w_eps)


def compute_residuals(returns: pd.DataFrame, k: int = K_FACTORS,
                      corr_win: int = CORR_WINDOW, beta_win: int = BETA_WINDOW,
                      start: str = None, verbose: bool = True) -> pd.DataFrame:
    dates = returns.index
    first = corr_win + 1
    if start is not None:
        first = max(first, dates.searchsorted(pd.Timestamp(start)))
    R = returns.to_numpy(dtype=np.float64)
    T, N = R.shape
    out = np.full((T, N), np.nan)

    for ti in range(first, T):
        Rw = R[ti - corr_win:ti]                      # past 252 days (t-252..t-1)
        valid = ~np.isnan(Rw).any(axis=0)
        valid &= ~np.isnan(R[ti])                     # need day-t return to trade
        idx = np.where(valid)[0]
        if len(idx) < MIN_STOCKS:
            continue
        W, B = fit_factor_model(Rw[:, idx], k, beta_win)
        r_t = R[ti, idx]
        eps_t = r_t - (r_t @ W) @ B                   # out-of-sample residual
        out[ti, idx] = eps_t
        if verbose and ti % 500 == 0:
            print(f"  {dates[ti].date()}  universe={len(idx)}")
    res = pd.DataFrame(out, index=dates, columns=returns.columns)
    res = res.dropna(how="all")
    return res


def main(start="1998-01-01"):
    close = cd.load_field_panel("Close")
    returns = close.pct_change(fill_method=None)
    returns = returns.clip(-0.5, 0.5)
    res = compute_residuals(returns, start=start)
    path = os.path.join(OUT_DIR, f"residuals_pca{K_FACTORS}.parquet")
    res.to_parquet(path)
    # also persist the aligned return panel for later mapping/benchmarks
    returns.loc[res.index].to_parquet(os.path.join(OUT_DIR, "returns_panel.parquet"))
    print(f"saved {path}  shape={res.shape}  "
          f"avg universe={res.notna().sum(axis=1).mean():.0f}")


if __name__ == "__main__":
    main()
