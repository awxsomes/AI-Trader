"""Turn the 15-minute close panel into a model-ready "family" file at a chosen
bar size and overnight rule: residuals, returns and factor maps, exactly the
arrays full/backtest_full.py consumes, but with a bar instead of a day as
the period.

    python intraday\\build_family.py --bar 60 --overnight hold
    python intraday\\build_family.py --bar 15 --overnight flat

--bar       15, 30 or 60 minutes (bars per day: 26, 13, 7; the last hourly
            bar is the 15:30-16:00 half hour)
--overnight hold: positions are carried overnight; the first bar of the day
                  earns close(prev day) -> first close, gap included.
            flat: everything is liquidated at the last bar of each day
                  (cost charged in the backtest) and the first bar earns
                  open -> first close, so the gap is never held.

Windows are in BARS and default to the daily code's values (252 for the
correlation matrix, 60 for loadings, 30 for the model input): for hourly
bars 252 bars is ~7 weeks, for 15-minute bars ~10 days.

Output: output\\intraday\\fam_{bar}m_{overnight}.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "statarb"))
from residuals import compute_residuals, K_FACTORS            # noqa: E402

OUT = os.path.join(ROOT, "output", "intraday")
OPEN_MIN = 9 * 60 + 30


def load_panels():
    close = pd.read_parquet(os.path.join(OUT, "close_15m.parquet"))
    open_ = pd.read_parquet(os.path.join(OUT, "open_15m.parquet"))
    if close.empty:
        sys.exit("ERROR: output\\intraday\\close_15m.parquet is empty. Rerun "
                 "python intraday\\download_bars.py and check it ends with 'Saved ...'.")
    # early-close afternoons and holidays: slots where almost nothing traded
    alive = close.notna().mean(axis=1) > 0.1
    close, open_ = close[alive], open_[alive]
    # a missing 15-minute bar means no trade printed; carry the last close
    # for at most two bars within the same day (return 0), never overnight
    day = close.index.normalize()
    close = close.groupby(day).ffill(limit=2)
    open_ = open_.where(open_.notna(), close.groupby(day).shift(1))
    return close, open_


def aggregate(close, open_, bar):
    """Last close / first open within each bar-size bin of the trading day."""
    day = close.index.normalize()
    mins = close.index.hour * 60 + close.index.minute - OPEN_MIN   # bar END minutes since open
    bin_id = (mins - 1) // bar
    bin_end = day + pd.to_timedelta(OPEN_MIN + (bin_id + 1) * bar, unit="m")
    bin_end = pd.DatetimeIndex(np.minimum(bin_end.values,
                                          (day + pd.Timedelta(hours=16)).values))
    c = close.groupby(bin_end).last()
    o = open_.groupby(bin_end).first()
    return c, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=int, choices=[15, 30, 60], required=True)
    ap.add_argument("--overnight", choices=["hold", "flat"], required=True)
    ap.add_argument("--corr-window", type=int, default=252)
    ap.add_argument("--beta-window", type=int, default=60)
    ap.add_argument("--clip", type=float, default=0.2, help="per-bar return clip")
    args = ap.parse_args()

    close, open_ = load_panels()
    c, o = aggregate(close, open_, args.bar)
    day = c.index.normalize()
    first_of_day = ~day.duplicated()
    last_of_day = ~day.duplicated(keep="last")
    bars_per_day = int(pd.Series(1, index=c.index).groupby(day).sum().median())

    ret = c.pct_change(fill_method=None)
    if args.overnight == "flat":
        gapless = c / o - 1
        ret[first_of_day] = gapless[first_of_day]
    else:
        ret.iloc[0] = np.nan
    ret = ret.clip(-args.clip, args.clip)
    ret = ret.loc[:, ret.notna().mean() > 0.5]
    print(f"{args.bar}-minute bars, overnight={args.overnight}: {len(ret)} bars, "
          f"{bars_per_day}/day, {ret.shape[1]} symbols, "
          f"{ret.index[0]} .. {ret.index[-1]}")

    res, W, B = compute_residuals(ret, K_FACTORS, args.corr_window, args.beta_window,
                                  verbose=False, return_maps=True)
    if res.empty:
        sys.exit(f"ERROR: no residuals: fewer than 100 stocks had a full "
                 f"{args.corr_window}-bar history at any point.")
    eod = last_of_day[c.index.get_indexer(res.index)]
    R = ret.loc[res.index].to_numpy(np.float32)
    print(f"  residuals: {res.shape[0]} bars, avg universe "
          f"{res.notna().sum(axis=1).mean():.0f} stocks")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"fam_{args.bar}m_{args.overnight}.npz")
    np.savez_compressed(path, eps=res.to_numpy(np.float32), R=R, W=W, B=B,
                        dates=res.index.values.astype("datetime64[m]"),
                        eod=eod.astype(bool), symbols=np.array(res.columns),
                        bars_per_day=bars_per_day, bar=args.bar,
                        overnight=args.overnight, corr_window=args.corr_window,
                        beta_window=args.beta_window)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
