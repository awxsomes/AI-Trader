"""Download the daily stock data the paper scripts need.

Creates one file per ticker, <TICKER>_1d.parquet, in the same format as the
original trading2/data folder (adjusted daily Open/High/Low/Close/Volume,
full history).

Usage (from the papers_impl folder):
    pip install yfinance
    python download_data.py
Then:
    $env:TRADING_DATA_DIR = "<this folder>\\data"
"""
import os
import time
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, "tickers.txt")) as f:
    tickers = f.read().split()

failed = []
for i, t in enumerate(tickers, 1):
    path = os.path.join(OUT, f"{t}_1d.parquet")
    if os.path.exists(path):
        continue
    try:
        df = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        if len(df) == 0:
            raise ValueError("no rows")
        df.to_parquet(path)
        print(f"[{i}/{len(tickers)}] {t}: {len(df)} days")
    except Exception as e:
        failed.append(t)
        print(f"[{i}/{len(tickers)}] {t}: FAILED ({e})")
    time.sleep(0.3)

print(f"\nDone. Saved to {OUT}")
if failed:
    print(f"{len(failed)} failed (re-run the script to retry): {' '.join(failed)}")
