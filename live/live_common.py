"""Shared pieces for running the CNN+Transformer statistical arbitrage
strategy live (Alpaca paper trading).

Pipeline for one trading day (same logic as the backtest):
  1. daily closes for the stock universe (Yahoo Finance via yfinance, or
     the local parquet folder),
  2. out-of-sample PCA-5 residuals for the recent past (statarb/residuals.py),
  3. the last 30 days of each stock's residuals, cumulated, go through the
     trained CNN+Transformer -> a weight on each residual portfolio,
  4. residual-portfolio weights are mapped to stock weights with today's
     factor model (w_R = (I - W B) w_eps) and normalized so sum|w_R| = 1.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "statarb"))

from models import L_WINDOW, CNNTransformer                     # noqa: E402
from residuals import (compute_residuals, fit_factor_model,      # noqa: E402
                       residual_to_stock_weights, CORR_WINDOW, K_FACTORS,
                       BETA_WINDOW, MIN_STOCKS)

MODEL_PATH = os.path.join(HERE, "model_cnn_trans.pt")
TICKERS_PATH = os.path.join(ROOT, "tickers.txt")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_universe():
    with open(TICKERS_PATH) as f:
        return f.read().split()


def to_alpaca_symbol(t: str) -> str:
    """Yahoo uses BRK-B, Alpaca uses BRK.B."""
    return t.replace("-", ".")


def fetch_closes_yahoo(tickers, period="2y") -> pd.DataFrame:
    """Date x Ticker panel of adjusted daily closes. During market hours the
    last row is today's partial bar, i.e. the latest price."""
    import yfinance as yf
    df = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                     progress=False, threads=True, group_by="column")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.loc[:, close.notna().any()]
    return close.sort_index()


def fetch_closes_local(data_dir, tickers) -> pd.DataFrame:
    import common_data as cd
    return cd.load_field_panel("Close", data_dir=data_dir, tickers=tickers)


def get_closes(source: str, n_days: int, data_dir: str = None) -> pd.DataFrame:
    tickers = load_universe()
    if source == "yahoo":
        years = int(np.ceil(n_days / 252)) + 1
        close = fetch_closes_yahoo(tickers, period=f"{years}y")
    else:
        close = fetch_closes_local(data_dir, tickers)
    return close.iloc[-n_days:]


def closes_to_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None).clip(-0.5, 0.5).iloc[1:]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def save_model(model, meta: dict, path: str = MODEL_PATH):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python live\\train_live.py")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CNNTransformer()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# Signals -> target stock weights
# ---------------------------------------------------------------------------

def target_weights(returns: pd.DataFrame, model) -> pd.Series:
    """Target stock weights (sum |w| = 1) to hold from the last date in
    `returns` until the next rebalance.

    Needs at least CORR_WINDOW + L_WINDOW + 1 rows of returns.
    """
    need = CORR_WINDOW + L_WINDOW + 1
    if len(returns) < need:
        raise ValueError(f"need {need} days of returns, got {len(returns)}")
    returns = returns.iloc[-(need + 5):]

    # (a) out-of-sample residuals for the last L days (each day uses a factor
    #     model fitted on the 252 days before it, as in the backtest)
    res = compute_residuals(returns, verbose=False)
    res = res.iloc[-L_WINDOW:]
    has_window = res.notna().all(axis=0) & (len(res) == L_WINDOW)

    # (b) today's factor model: fitted on the last 252 days INCLUDING today,
    #     it defines the residual portfolios held over the next day
    R = returns.iloc[-CORR_WINDOW:]
    valid = R.notna().all(axis=0)
    cols = R.columns[valid]
    if len(cols) < MIN_STOCKS:
        raise ValueError(f"only {len(cols)} stocks with full history")
    W, B = fit_factor_model(R[cols].to_numpy(np.float64), K_FACTORS, BETA_WINDOW)

    # (c) model weights on residual portfolios
    trade_cols = [c for c in cols if has_window.get(c, False)]
    X = res[trade_cols].cumsum().to_numpy(np.float32).T.copy()   # (n, L)
    with torch.no_grad():
        w_tr = model(torch.from_numpy(X)).numpy().astype(np.float64)
    w_eps = pd.Series(0.0, index=cols)
    w_eps[trade_cols] = w_tr

    # (d) residual weights -> stock weights, normalized to sum |w| = 1
    w_R = residual_to_stock_weights(w_eps.to_numpy(), W, B)
    w_R = pd.Series(w_R, index=cols)
    return w_R / w_R.abs().sum()
