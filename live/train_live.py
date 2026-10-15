"""Train the CNN+Transformer on the most recent data and save it for paper
trading.

Same model and Sharpe-ratio objective as the backtest, fitted once on the
last 1000 trading days of out-of-sample PCA residuals. Re-run roughly once
a month (the backtest re-estimated once a year and still worked, so this
is conservative).

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
from backtest import build_windows, train_model, TRAIN_WINDOW   # noqa: E402
from models import CNNTransformer, L_WINDOW                      # noqa: E402
from residuals import compute_residuals, CORR_WINDOW             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "local"], default="yahoo")
    ap.add_argument("--data-dir", default=os.environ.get("TRADING_DATA_DIR"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_days = TRAIN_WINDOW + L_WINDOW + CORR_WINDOW + 10
    print(f"Loading ~{n_days} days of closes from {args.source} ...", flush=True)
    close = lc.get_closes(args.source, n_days, args.data_dir)
    # drop today's bar (it may be partial if the market is still open)
    if close.index[-1].date() >= dt.date.today():
        close = close.iloc[:-1]
    returns = lc.closes_to_returns(close)
    print(f"  {returns.shape[1]} stocks, {returns.index[0].date()} .. "
          f"{returns.index[-1].date()}", flush=True)

    print("Computing out-of-sample PCA residuals ...", flush=True)
    res = compute_residuals(returns, verbose=False)
    X, Y, M = build_windows(res)
    X, Y, M = X[-TRAIN_WINDOW:], Y[-TRAIN_WINDOW:], M[-TRAIN_WINDOW:]
    dates = res.index[-TRAIN_WINDOW:]
    print(f"  training on {dates[0].date()} .. {dates[-1].date()} "
          f"({M.sum(axis=1).mean():.0f} stocks/day)", flush=True)

    model = CNNTransformer()
    tic = time.time()
    train_model(model, X, Y, M, args.epochs, log_prefix="CNN+Trans")
    print(f"  trained in {time.time() - tic:.0f}s", flush=True)

    meta = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "train_start": str(dates[0].date()),
        "train_end": str(dates[-1].date()),
        "epochs": args.epochs,
        "source": args.source,
    }
    lc.save_model(model, meta)
    print(f"Saved model to {lc.MODEL_PATH}")


if __name__ == "__main__":
    main()
