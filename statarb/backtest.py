"""Rolling out-of-sample backtest of the three arbitrage models of
Guijarro-Ordonez, Pelger & Zanotti (2022) on the locally computed PCA
residuals.

Protocol (Section III.C of the paper, computationally scaled):
  - signal input: window of the last L=30 cumulative residuals (no
    look-ahead: windows end at t-1; weights are applied to the day-t
    residual return),
  - deep models trained by maximizing the annualized SHARPE RATIO of the
    training-period strategy returns (the paper's Sharpe objective),
  - rolling estimation window of 1000 days; the paper re-estimates every
    125 days, here we re-estimate every RETRAIN_EVERY days (default 250)
    with warm starts to keep CPU time manageable,
  - daily leverage constraint ||w||_1 = 1 across residual portfolios.

We trade the residual portfolios directly (the paper's conceptual setup in
Section II.D; the paper additionally maps residual weights back to stock
space, which changes the leverage normalization but not the substance).

Outputs (./output):
  - statarb_oos_returns.csv       daily OOS strategy returns per model
  - statarb_performance.csv       annualized Sharpe / mean / vol table
  - fig_statarb_cumret.png        cumulative OOS return comparison
  - fig_statarb_example.png       example residual path + model weights
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import L_WINDOW, ou_threshold_weights, FourierFFN, CNNTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "output")

TRAIN_WINDOW = 1000       # days used to fit the deep models
RETRAIN_EVERY = 250       # paper: 125; scaled for CPU
EPOCHS_FIRST = 20
EPOCHS_WARM = 6
LR = 1e-3
SEED = 7


def build_windows(res: pd.DataFrame):
    """Cumulative-residual windows.

    Returns
      X   : (T, N, L) float32, X[t, n] = cumsum of residuals over
            (t-L .. t-1) for stock n (input available at end of day t-1)
      Y   : (T, N) float32, day-t residual return (what the weight earns)
      M   : (T, N) bool, True where both X and Y are valid
    """
    E = res.to_numpy(dtype=np.float32)
    T, N = E.shape
    L = L_WINDOW
    X = np.full((T, N, L), np.nan, dtype=np.float32)
    # sliding windows of residuals over (t-L .. t-1)
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(E, L, axis=0)       # (T-L+1, N, L), ends at t
    X[L:] = np.cumsum(sw[:-1], axis=2)           # window ending at t-1
    Y = E.copy()
    M = ~np.isnan(X).any(axis=2) & ~np.isnan(Y)
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    return X, Y, M


def portfolio_returns_np(w: np.ndarray, Y: np.ndarray, M: np.ndarray):
    w = w * M
    denom = np.abs(w).sum(axis=1)
    denom[denom < 1e-12] = np.inf
    return (w * Y).sum(axis=1) / denom


def sharpe_loss(w: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
    w = w * m
    denom = w.abs().sum(dim=1).clamp_min(1e-8)
    r = (w * y).sum(dim=1) / denom
    return -(r.mean() / r.std().clamp_min(1e-12)) * np.sqrt(252.0)


def train_model(model, X, Y, M, epochs, lr=LR, batch_days=60, log_prefix=""):
    """Mini-batch Sharpe maximization: each step draws a random contiguous
    block of `batch_days` days and maximizes the annualized Sharpe ratio of
    the strategy returns within the block (stochastic version of the
    paper's objective, needed to keep CPU memory bounded)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Tn, N, L = X.shape
    rng = np.random.default_rng(SEED)
    n_batches = max(Tn // batch_days, 1)
    for ep in range(epochs):
        ep_sharpes = []
        for _ in range(n_batches):
            s = int(rng.integers(0, Tn - batch_days + 1))
            xb = torch.from_numpy(X[s:s + batch_days].reshape(-1, L))
            yb = torch.from_numpy(Y[s:s + batch_days])
            mb = torch.from_numpy(M[s:s + batch_days].astype(np.float32))
            opt.zero_grad()
            w = model(xb).reshape(batch_days, N)
            loss = sharpe_loss(w, yb, mb)
            loss.backward()
            opt.step()
            ep_sharpes.append(-loss.item())
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    {log_prefix} epoch {ep:3d}  "
                  f"batch Sharpe {np.mean(ep_sharpes):.2f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict_weights(model, X, chunk=40000):
    Tn, N, L = X.shape
    flat = X.reshape(-1, L)
    parts = []
    for i in range(0, len(flat), chunk):
        parts.append(model(torch.from_numpy(flat[i:i + chunk])).numpy())
    return np.concatenate(parts).reshape(Tn, N)


def run(retrain_every=RETRAIN_EVERY, oos_start="2003-01-01"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(os.cpu_count())

    res = pd.read_parquet(os.path.join(OUT_DIR, "residuals_pca5.parquet"))
    X, Y, M = build_windows(res)
    dates = res.index
    t0 = max(dates.searchsorted(pd.Timestamp(oos_start)), TRAIN_WINDOW + L_WINDOW)
    T = len(dates)
    print(f"residuals {res.shape}, OOS {dates[t0].date()} .. {dates[-1].date()} "
          f"({T - t0} days)", flush=True)

    models = {
        "CNN+Trans": CNNTransformer(),
        "Fourier+FFN": FourierFFN(),
    }
    weights = {k: np.zeros((T, res.shape[1]), dtype=np.float32) for k in models}
    first_fit = {k: True for k in models}

    for block_start in range(t0, T, retrain_every):
        block_end = min(block_start + retrain_every, T)
        tr0, tr1 = block_start - TRAIN_WINDOW, block_start
        print(f"[block {dates[block_start].date()} .. {dates[block_end-1].date()}] "
              f"train {dates[tr0].date()} .. {dates[tr1-1].date()}", flush=True)
        Xtr, Ytr, Mtr = X[tr0:tr1], Y[tr0:tr1], M[tr0:tr1]
        for name, model in models.items():
            ep = EPOCHS_FIRST if first_fit[name] else EPOCHS_WARM
            tic = time.time()
            train_model(model, Xtr, Ytr, Mtr, ep, log_prefix=name)
            print(f"    {name}: {ep} epochs in {time.time()-tic:.0f}s", flush=True)
            first_fit[name] = False
            weights[name][block_start:block_end] = predict_weights(
                model, X[block_start:block_end])

    # OU benchmark (no training; estimated window-by-window)
    print("OU+Threshold benchmark ...", flush=True)
    w_ou = np.zeros((T, res.shape[1]), dtype=np.float32)
    Xo = X[t0:T].reshape(-1, L_WINDOW)
    w_ou[t0:T] = ou_threshold_weights(Xo).reshape(T - t0, -1).astype(np.float32)

    all_w = dict(weights)
    all_w["OU+Thresh"] = w_ou
    # Equal-weighted residual portfolio as a null benchmark
    all_w["EqualWeight"] = np.ones_like(w_ou)

    rets = {}
    for name, w in all_w.items():
        r = portfolio_returns_np(w[t0:T], Y[t0:T], M[t0:T])
        rets[name] = pd.Series(r, index=dates[t0:T])
    rets = pd.DataFrame(rets)
    rets.to_csv(os.path.join(OUT_DIR, "statarb_oos_returns.csv"))

    perf = pd.DataFrame({
        "ann_mean": 252 * rets.mean(),
        "ann_vol": np.sqrt(252) * rets.std(),
    })
    perf["sharpe"] = perf["ann_mean"] / perf["ann_vol"]
    perf.to_csv(os.path.join(OUT_DIR, "statarb_performance.csv"))
    print("\n", perf.round(3))

    # cumulative return figure (paper's Figure 5 style)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ["CNN+Trans", "Fourier+FFN", "OU+Thresh", "EqualWeight"]:
        ax.plot(rets.index, (1 + rets[name]).cumprod(), lw=1.1, label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (growth of 1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_statarb_cumret.png"), dpi=200)
    plt.close(fig)

    # example residual with model allocations (paper's Figure A.2 style)
    n_ex = int(np.argmax(M[t0:T].sum(axis=0)))
    t_ex = slice(T - 400, T)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    axes[0].plot(dates[t_ex], np.cumsum(np.where(M[t_ex, n_ex], Y[t_ex, n_ex], 0)),
                 color="k", lw=1.0)
    axes[0].set_ylabel("Cumulative residual")
    for name in ["CNN+Trans", "Fourier+FFN", "OU+Thresh"]:
        w = all_w[name][t_ex, n_ex]
        nrm = np.abs(w).max() or 1
        axes[1].plot(dates[t_ex], w / nrm, lw=0.9, label=name)
    axes[1].set_ylabel("Allocation weight (scaled)")
    axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_statarb_example.png"), dpi=200)
    plt.close(fig)
    print("done.", flush=True)


if __name__ == "__main__":
    run()
