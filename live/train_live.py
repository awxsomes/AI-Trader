"""Train the live trading model on the most recent data and save it.

Default (full paper, Sections II-III.J): the cost-aware CNN+Transformer of
full/model_full.py, trained on PCA-5 residuals to maximize the Sharpe ratio
of STOCK-SPACE returns net of trading costs (5 bp per unit of turnover, 1 bp
per day on shorts), with each residual's previous allocation as an input.
In the 2002-2026 backtest this was the only variant that stayed profitable
after costs (see output/full and RESULTS.md).

--legacy trains the original version (residual space, no costs).

Re-run roughly once a month; run_daily.py does it automatically.

Usage (from the papers_impl folder):
    python live\\train_live.py                      # fresh data from Yahoo
    python live\\train_live.py --source local --data-dir "C:\\...\\trading2\\data"
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_common as lc                                         # noqa: E402
from residuals import compute_residuals, CORR_WINDOW             # noqa: E402
from models import CNNTransformer, L_WINDOW                      # noqa: E402

TRAIN_WINDOW = 1000


def train_full(returns, epochs):
    import backtest_full as bf
    from model_full import CNNTransformerPaper
    res, W, B = compute_residuals(returns, verbose=False, return_maps=True)
    R = returns.loc[res.index].to_numpy(np.float32)
    data = bf.prepare_arrays(res.to_numpy(np.float32), R, W, B)
    T, N = data[1].shape
    t0, t1 = T - TRAIN_WINDOW, T
    dates = res.index[t0:t1]
    print(f"  training on {dates[0].date()} .. {dates[-1].date()} "
          f"({data[1][t0:t1].sum(axis=1).mean():.0f} stocks/day)", flush=True)
    store = {"wR": np.zeros((T, N), np.float32),
             "prev_in": np.zeros((T + 1, N), np.float32)}
    model = CNNTransformerPaper(use_prev=True)
    bf.train(model, data, t0, t1, epochs, True, store, "CNN+Trans (cost-aware)")
    model.eval()
    return model, dates, {"arch": "paper_cost", "factors": "PCA5",
                          "costs": "5bp turnover + 1bp short"}


def train_legacy(returns, epochs):
    from backtest import build_windows, train_model
    res = compute_residuals(returns, verbose=False)
    X, Y, M = build_windows(res)
    X, Y, M = X[-TRAIN_WINDOW:], Y[-TRAIN_WINDOW:], M[-TRAIN_WINDOW:]
    dates = res.index[-TRAIN_WINDOW:]
    model = CNNTransformer()
    train_model(model, X, Y, M, epochs, log_prefix="CNN+Trans (legacy)")
    return model, dates, {"arch": "legacy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "local"], default="yahoo")
    ap.add_argument("--data-dir", default=os.environ.get("TRADING_DATA_DIR"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--legacy", action="store_true",
                    help="train the original model (no costs, residual space)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count())

    n_days = TRAIN_WINDOW + L_WINDOW + CORR_WINDOW + 10
    print(f"Loading ~{n_days} days of closes from {args.source} ...", flush=True)
    close = lc.get_closes(args.source, n_days, args.data_dir)
    # drop today's bar (it may be partial if the market is still open)
    if close.index[-1].date() >= dt.date.today():
        close = close.iloc[:-1]
    returns = lc.closes_to_returns(close)
    print(f"  {returns.shape[1]} stocks, {returns.index[0].date()} .. "
          f"{returns.index[-1].date()}", flush=True)

    tic = time.time()
    model, dates, extra = (train_legacy if args.legacy else train_full)(
        returns, args.epochs)
    print(f"  trained in {time.time() - tic:.0f}s", flush=True)

    meta = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "train_start": str(dates[0].date()),
        "train_end": str(dates[-1].date()),
        "epochs": args.epochs,
        "source": args.source,
        **extra,
    }
    lc.save_model(model, meta)
    print(f"Saved model to {lc.MODEL_PATH}")


if __name__ == "__main__":
    main()
