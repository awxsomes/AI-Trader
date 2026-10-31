"""Full-paper backtest: stock-space trading with trading frictions.

For each day t (Guijarro-Ordonez, Pelger & Zanotti 2022, Sections II.C and
III.J):
  w_eps  = model(last 30 cumulative residuals up to t-1 [, previous allocation])
  w_R    = Phi_{t-1}' w_eps  = w_eps - W (B' w_eps),  scaled so sum|w_R| = 1
  gross  = w_R . R_t
  cost   = 0.0005 * sum|w_R,t - w_R,t-1|  +  0.0001 * sum|min(w_R,t, 0)|
  net    = gross - cost
Training maximizes the annualized Sharpe ratio of `net` (with --cost-train)
or of `gross` (without), on a rolling 1000-day window, in consecutive 125-day
batches (Appendix D). Evaluation is sequential and exact, always reporting
both gross and net-of-cost performance.

Compute scaling (2 CPUs instead of the paper's 8 GPUs): retrain every
RETRAIN days with warm starts (paper: 125 days, 100 epochs each).

Usage:
  python full/backtest_full.py --family ipca5 --cost-train
  python full/backtest_full.py --family pca5
  python full/backtest_full.py --family ipca5 --ou      # OU+Threshold baseline
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "statarb"))
from model_full import CNNTransformerPaper, L_WINDOW              # noqa: E402
from models import ou_threshold_weights                          # noqa: E402

OUT = os.path.join(ROOT, "output", "full")
C_TC, C_SHORT = 0.0005, 0.0001
TRAIN_WINDOW, BATCH = 1000, 125
OOS_START = "2002-01-01"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_family(name):
    d = np.load(os.path.join(OUT, f"{name}.npz"), allow_pickle=True)
    dates = pd.DatetimeIndex(d["dates"])
    return (dates,) + prepare_arrays(d["eps"], d["R"], d["W"], d["B"])


def prepare_arrays(eps, R, W, B):
    """Model inputs from residuals eps (T,N), raw returns R (T,N) and factor
    maps W, B (T,N,K): windows X, tradable mask M, and masked R, W, B."""
    T, N = eps.shape
    L = L_WINDOW
    X = np.zeros((T, N, L), dtype=np.float32)
    ok_win = np.zeros((T, N), dtype=bool)
    sw = sliding_window_view(eps, L, axis=0)                  # (T-L+1, N, L)
    X[L:] = np.nan_to_num(np.cumsum(sw[:-1], axis=2))       # window ending t-1
    ok_win[L:] = ~np.isnan(sw[:-1]).any(axis=2)
    universe = np.isfinite(eps)                               # factor model defined at t
    M = ok_win & universe
    R = np.where(universe, np.nan_to_num(R), 0).astype(np.float32)
    W = np.where(universe[..., None], W, 0).astype(np.float32)
    B = np.where(universe[..., None], B, 0).astype(np.float32)
    return X, M, R, W, B


# ---------------------------------------------------------------------------
# Portfolio mechanics (torch, batched over days)
# ---------------------------------------------------------------------------

def to_stock_weights(w_eps, W, B):
    """w_eps (D,N), W,B (D,N,K) -> normalized stock weights (D,N)."""
    proj = torch.einsum("dnk,dn->dk", B, w_eps)
    wR = w_eps - torch.einsum("dnk,dk->dn", W, proj)
    return wR / wR.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)


def pnl(wR, R, wR_prev):
    gross = (wR * R).sum(dim=1)
    turnover = (wR - wR_prev).abs().sum(dim=1)
    short = wR.clamp(max=0).abs().sum(dim=1)
    net = gross - C_TC * turnover - C_SHORT * short
    return gross, net, turnover, short


def normalized_prev(w_eps, M):
    """Previous allocation as model input: scaled so typical size is ~1."""
    n = M.sum(dim=1, keepdim=True).clamp_min(1)
    return w_eps / w_eps.abs().sum(dim=1, keepdim=True).clamp_min(1e-12) * n


def ann_sharpe(x):
    return (x.mean() / x.std().clamp_min(1e-12)) * np.sqrt(252.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, data, t0, t1, epochs, cost_train, store, log=""):
    """Approximate parallel training (Appendix D): the previous allocation
    used as input and in the turnover cost comes from the previous epoch."""
    X, M, R, W, B = data
    N = X.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        sharpes = []
        for s in range(t0, t1, BATCH):
            e = min(s + BATCH, t1)
            Xb = torch.from_numpy(X[s:e]).reshape(-1, L_WINDOW)
            Mb = torch.from_numpy(M[s:e].astype(np.float32))
            prev_in = torch.from_numpy(store["prev_in"][s:e]).reshape(-1)
            out = model(Xb, prev_in).reshape(e - s, N) * Mb
            wR = to_stock_weights(out, torch.from_numpy(W[s:e]),
                                  torch.from_numpy(B[s:e]))
            wR_prev = torch.cat([torch.from_numpy(store["wR"][s - 1:s]), wR[:-1]])
            gross, net, _, _ = pnl(wR, torch.from_numpy(R[s:e]), wR_prev)
            loss = -ann_sharpe(net if cost_train else gross)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sharpes.append(-loss.item())
            with torch.no_grad():   # allocations for the next epoch's inputs
                store["wR"][s:e] = wR.numpy()
                store["prev_in"][s + 1:e + 1] = normalized_prev(out, Mb).numpy()[: e - s]
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    {log} epoch {ep:3d}  batch Sharpe {np.mean(sharpes):.2f}",
                  flush=True)
    model.eval()


# ---------------------------------------------------------------------------
# Sequential out-of-sample evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_block(model, data, s, e, state):
    X, M, R, W, B = data
    rows = []
    for t in range(s, e):
        Mt = torch.from_numpy(M[t:t + 1].astype(np.float32))
        out = model(torch.from_numpy(X[t]), state["prev_in"][0]).unsqueeze(0) * Mt
        wR = to_stock_weights(out, torch.from_numpy(W[t:t + 1]),
                              torch.from_numpy(B[t:t + 1]))
        g, n, to, sh = pnl(wR, torch.from_numpy(R[t:t + 1]), state["wR"])
        rows.append((g.item(), n.item(), to.item(), sh.item()))
        state["wR"] = wR
        state["prev_in"] = normalized_prev(out, Mt)
        state.setdefault("targets", []).append(wR[0].numpy().astype(np.float32))
    return rows


def run_cnn(family, cost_train, retrain, first_epochs, warm_epochs, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(os.cpu_count())
    dates, *data = load_family(family)
    X, M, R, W, B = data
    T, N = M.shape
    t_start = max(dates.searchsorted(pd.Timestamp(OOS_START)),
                  TRAIN_WINDOW + L_WINDOW + 1)
    model = CNNTransformerPaper(use_prev=cost_train)
    store = {"wR": np.zeros((T, N), np.float32),
             "prev_in": np.zeros((T + 1, N), np.float32)}
    state = {"wR": torch.zeros(1, N), "prev_in": torch.zeros(1, N)}
    rows, first = [], True
    tag = f"{family}{'-cost' if cost_train else ''}"
    print(f"[{tag}] OOS {dates[t_start].date()} .. {dates[-1].date()}", flush=True)
    for s in range(t_start, T, retrain):
        e = min(s + retrain, T)
        tic = time.time()
        train(model, data, s - TRAIN_WINDOW, s,
              first_epochs if first else warm_epochs, cost_train, store, tag)
        first = False
        rows += evaluate_block(model, data, s, e, state)
        print(f"[{tag}] {dates[s].date()}..{dates[e - 1].date()} done "
              f"({time.time() - tic:.0f}s)", flush=True)
    # daily target stock weights and realized returns, for execution studies
    np.savez_compressed(
        os.path.join(OUT, f"cnn_{family}{'_cost' if cost_train else ''}_targets.npz"),
        targets=np.stack(state["targets"]), R=R[t_start:],
        dates=dates[t_start:].values.astype("datetime64[D]"))
    return pd.DataFrame(rows, index=dates[t_start:],
                        columns=["gross", "net", "turnover", "short"])


def run_ou(family):
    dates, X, M, R, W, B = load_family(family)
    T, N = M.shape
    t_start = max(dates.searchsorted(pd.Timestamp(OOS_START)),
                  TRAIN_WINDOW + L_WINDOW + 1)
    prev = torch.zeros(1, N)
    rows = []
    for t in range(t_start, T):
        w = ou_threshold_weights(X[t].astype(np.float64)) * M[t]
        wR = to_stock_weights(torch.from_numpy(w[None].astype(np.float32)),
                              torch.from_numpy(W[t:t + 1]), torch.from_numpy(B[t:t + 1]))
        if not torch.isfinite(wR).all() or wR.abs().sum() == 0:
            wR = torch.zeros(1, N)
        g, n, to, sh = pnl(wR, torch.from_numpy(R[t:t + 1]), prev)
        rows.append((g.item(), n.item(), to.item(), sh.item()))
        prev = wR
    return pd.DataFrame(rows, index=dates[t_start:],
                        columns=["gross", "net", "turnover", "short"])


def summarize(df):
    out = {}
    for c in ["gross", "net"]:
        mu, sd = 252 * df[c].mean(), np.sqrt(252) * df[c].std()
        out[f"{c}_sharpe"], out[f"{c}_mean"], out[f"{c}_vol"] = mu / sd, mu, sd
    out["avg_turnover"] = df["turnover"].mean()
    out["avg_short_share"] = df["short"].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pca5", "ipca5"], required=True)
    ap.add_argument("--cost-train", action="store_true")
    ap.add_argument("--ou", action="store_true")
    ap.add_argument("--retrain", type=int, default=250)
    ap.add_argument("--first-epochs", type=int, default=30)
    ap.add_argument("--warm-epochs", type=int, default=8)
    args = ap.parse_args()
    if args.ou:
        name, df = f"ou_{args.family}", run_ou(args.family)
    else:
        name = f"cnn_{args.family}{'_cost' if args.cost_train else ''}"
        df = run_cnn(args.family, args.cost_train, args.retrain,
                     args.first_epochs, args.warm_epochs)
    df.to_csv(os.path.join(OUT, f"{name}_returns.csv"))
    s = summarize(df)
    with open(os.path.join(OUT, f"{name}_summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(name, json.dumps({k: round(v, 4) for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    main()
