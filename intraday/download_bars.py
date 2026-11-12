"""Download intraday bars for the universe from Alpaca's market data API and
save a regular-hours close panel at 15-minute resolution. Coarser bars
(30-minute, hourly) are built from this file by build_family.py, so the
download happens once.

The paper keys work for the data API too (ALPACA_API_KEY / ALPACA_SECRET_KEY).

Usage (from the repo folder):
    python intraday\\download_bars.py                 # last 5 years (default)
    python intraday\\download_bars.py --years 3                # quicker, wider error bars
    python intraday\\download_bars.py --feed iex      # if SIP is refused

Safe to rerun: each piece is saved under output\\intraday\\raw\\ as it
arrives, and pieces already on disk are skipped.

Output: output\\intraday\\close_15m.parquet  (rows: bar end time in
New York, regular hours only; columns: tickers in Alpaca spelling)
        output\\intraday\\open_15m.parquet   (bar opens, for the overnight=flat
                                           variant that excludes the gap)
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output", "intraday")
NY = "America/New_York"


def load_universe():
    with open(os.path.join(ROOT, "tickers.txt")) as f:
        return [t.replace("-", ".") for t in f.read().split()]      # Alpaca spelling


def fetch(client, symbols, start, end, feed):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed
    req = StockBarsRequest(symbol_or_symbols=symbols,
                           timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                           start=start, end=end, adjustment=Adjustment.ALL,
                           feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX,
                           limit=10_000)
    for attempt in range(5):
        try:
            return client.get_stock_bars(req).df
        except Exception as e:                       # rate limit / transient
            if attempt == 4:
                raise
            print(f"    retry {attempt + 1}: {str(e)[:80]}", flush=True)
            time.sleep(10 * (attempt + 1))


def regular_hours(df):
    """Keep bars whose END time is within 09:45 .. 16:00 New York, i.e. bars
    starting 09:30 .. 15:45. Index becomes the bar end time."""
    ts = df.index.get_level_values("timestamp").tz_convert(NY)
    end = ts + pd.Timedelta(minutes=15)
    minutes = end.hour * 60 + end.minute
    keep = (minutes >= 9 * 60 + 45) & (minutes <= 16 * 60)
    df = df[keep].copy()
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values("symbol"), end[keep].tz_localize(None)],
        names=["symbol", "t"])
    return df


def chunk_path(raw_dir, w0, w1, b):
    return os.path.join(raw_dir, f"{w0:%Y%m%d}_{w1:%Y%m%d}_b{b:03d}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--feed", choices=["sip", "iex"], default="sip")
    ap.add_argument("--batch", type=int, default=40, help="symbols per request")
    ap.add_argument("--months", type=int, default=2, help="months per request")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel requests (Alpaca allows ~200 requests/min)")
    args = ap.parse_args()
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY first.")
    from alpaca.data.historical import StockHistoricalDataClient
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    local = threading.local()

    def client():
        if not hasattr(local, "c"):
            local.c = StockHistoricalDataClient(key, secret)
        return local.c

    raw_dir = os.path.join(OUT, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    symbols = load_universe()
    end = pd.Timestamp(dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20))
    # windows start on fixed month boundaries, so a rerun (after a crash, or on
    # a later day) finds the already-downloaded pieces and skips them
    first = (end - pd.DateOffset(days=int(365.25 * args.years))).normalize().replace(day=1)
    # window edges anchored to January 1, so they stay identical on later days
    anchor = first.replace(month=1)
    edges = [e for e in pd.date_range(anchor, end, freq=f"{args.months}MS")
             if e >= first or e == anchor]
    edges = [max(e, first) for e in edges]
    edges = sorted(set(edges))
    if edges[-1] < end:
        edges.append(end)
    jobs = [(edges[i], edges[i + 1], b) for i in range(len(edges) - 1)
            for b in range(0, len(symbols), args.batch)]
    todo = [j for j in jobs if not os.path.exists(chunk_path(raw_dir, *j))]
    print(f"{len(symbols)} symbols, {first.date()} .. {end.date()}: {len(jobs)} pieces, "
          f"{len(jobs) - len(todo)} already on disk, {len(todo)} to download "
          f"({args.workers} at a time)", flush=True)

    def work(job):
        w0, w1, b = job
        df = fetch(client(), symbols[b:b + args.batch], w0, w1, args.feed)
        if df is None or len(df) == 0:
            long = pd.DataFrame({"symbol": pd.Series(dtype=str),
                                 "t": pd.Series(dtype="datetime64[ns]"),
                                 "open": pd.Series(dtype=float),
                                 "close": pd.Series(dtype=float)})
        else:
            long = regular_hours(df)[["open", "close"]].reset_index()
        tmp = chunk_path(raw_dir, *job) + ".tmp"
        long.to_parquet(tmp, index=False)
        os.replace(tmp, chunk_path(raw_dir, *job))      # complete pieces only
        return len(long)

    tic = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()                                     # re-raise any error
            if i % 20 == 0 or i == len(futs):
                el = time.time() - tic
                print(f"  {i}/{len(futs)} pieces  {el:5.0f}s elapsed, "
                      f"~{el / i * (len(futs) - i) / 60:.0f} min left", flush=True)

    # assemble from the pieces in LONG format (symbol, time) -> one row per
    # bar, so pieces holding different symbols can never overwrite each other
    print("Assembling ...", flush=True)
    long = pd.concat([pd.read_parquet(chunk_path(raw_dir, *j)) for j in jobs],
                     ignore_index=True)
    if long.empty:
        sys.exit("No bars came back at all; check the keys and try --feed iex.")
    long = long.drop_duplicates(["symbol", "t"], keep="last")
    close = long.pivot(index="t", columns="symbol", values="close").sort_index()
    open_ = long.pivot(index="t", columns="symbol", values="open").sort_index()
    close = close.reindex(columns=symbols)
    open_ = open_.reindex(columns=symbols)
    # full regular-hours grid, so missing bars show up as NaN rather than
    # silently shortening the day (IEX feed in particular skips quiet bars)
    days = pd.DatetimeIndex(sorted(set(close.index.normalize())))
    slots = [pd.Timedelta(minutes=9 * 60 + 45 + 15 * i) for i in range(26)]
    grid = pd.DatetimeIndex([d + s for d in days for s in slots])
    close = close.reindex(grid)
    open_ = open_.reindex(grid)
    # drop days with almost no data (holidays); early closes keep NaN slots
    live_days = close.notna().mean(axis=1).groupby(close.index.normalize()).mean() > 0.3
    keep = close.index.normalize().isin(live_days[live_days].index)
    close, open_ = close[keep], open_[keep]
    if close.empty:
        sys.exit("ERROR: every day was dropped as too thin; nothing saved. "
                 "Send the output above to Claude.")

    close.to_parquet(os.path.join(OUT, "close_15m.parquet"))
    open_.to_parquet(os.path.join(OUT, "open_15m.parquet"))
    cov = close.notna().mean()
    print(f"\nSaved {close.shape[0]} bars x {close.shape[1]} symbols "
          f"({close.index[0]} .. {close.index[-1]}, "
          f"{close.index.normalize().nunique()} trading days)")
    print(f"Bar coverage: median {cov.median():.1%}, "
          f"{(cov < 0.9).sum()} symbols below 90% (will be excluded on thin bars)")


if __name__ == "__main__":
    main()
