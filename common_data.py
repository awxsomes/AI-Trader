"""Shared data loading utilities for the paper implementations.

Reads the per-ticker daily OHLCV parquet files (as produced by the user's
data pipeline: one file per ticker named <TICKER>_1d.parquet, with a Date
index and adjusted Open/High/Low/Close/Volume columns).
"""
import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("TRADING_DATA_DIR", "/mnt/user-data/uploads/data")


def list_tickers(data_dir: str = DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "*_1d.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No *_1d.parquet stock files found in '{data_dir}'. Set "
            "TRADING_DATA_DIR to the folder that contains them, or run "
            "download_data.py to create a 'data' folder here.")
    return [os.path.basename(f).replace("_1d.parquet", "") for f in files]


def load_ohlc(ticker: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(data_dir, f"{ticker}_1d.parquet"))
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_field_panel(field: str = "Close", data_dir: str = DATA_DIR,
                     tickers=None) -> pd.DataFrame:
    """Return a Date x Ticker panel of one OHLCV field."""
    if tickers is None:
        tickers = list_tickers(data_dir)
    series = {}
    for t in tickers:
        try:
            df = load_ohlc(t, data_dir)
            series[t] = df[field]
        except Exception:
            continue
    panel = pd.DataFrame(series)
    panel = panel.sort_index()
    return panel


def first_valid_open_date(df: pd.DataFrame, window: int = 126,
                          max_flat_frac: float = 0.2):
    """First date from which Open prices look genuine.

    Old Yahoo-style daily data often carries synthetic opens of two kinds:
    Open == Close (whole move pushed overnight) and Open == previous Close
    (whole move pushed intraday; dominant in pre-2000 data). We require
    that, in a trailing window, at most `max_flat_frac` of days show
    either artifact.
    """
    flat_oc = np.abs(df["Close"] / df["Open"] - 1.0) < 1e-6
    flat_opc = np.abs(df["Open"] / df["Close"].shift(1) - 1.0) < 1e-6
    frac_oc = flat_oc.astype(float).rolling(window).mean()
    frac_opc = flat_opc.astype(float).rolling(window).mean()
    ok = df.index[(frac_oc < max_flat_frac) & (frac_opc < max_flat_frac)]
    if len(ok) == 0:
        return None
    return ok[0]
