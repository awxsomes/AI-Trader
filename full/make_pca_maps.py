"""PCA-5 residuals plus per-day factor maps, for stock-space trading.

Writes output/full/pca5.npz with
  dates (T,), tickers (N,), eps (T,N) residuals, R (T,N) raw returns,
  W, B (T,N,K) factor maps: stock weights = w - W @ (B.T @ w).
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "statarb"))
import common_data as cd                                         # noqa: E402
from residuals import compute_residuals                          # noqa: E402

OUT = os.path.join(ROOT, "output", "full")
os.makedirs(OUT, exist_ok=True)


def main(start="1998-01-01"):
    close = cd.load_field_panel("Close")
    returns = close.pct_change(fill_method=None).clip(-0.5, 0.5)
    res, W, B = compute_residuals(returns, start=start, return_maps=True)
    R = returns.loc[res.index]
    np.savez(os.path.join(OUT, "pca5.npz"),
             dates=res.index.values.astype("datetime64[D]"),
             tickers=np.array(res.columns), eps=res.to_numpy(np.float32),
             R=R.to_numpy(np.float32), W=W, B=B)
    print("saved", res.shape)


if __name__ == "__main__":
    main()
