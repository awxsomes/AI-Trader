"""Backtest every hard-coded pattern on the local daily parquet files and
ask, pattern by pattern: does trading the pattern's breakout beat breakouts
of arbitrary levels? The comparison for each trade is the same stock,
direction, distance from the previous close to the trigger, stop and target
distances and holding limit, but on random days.

Rules (identical to the live trader, from patterns/detect.py):
  * a setup armed at day t's close can be entered on t+1 .. expiry when
    price breaks the trigger; if the day opens past the trigger the fill
    is the open (skipped if the gap already used half the move to target);
  * exits: stop or target, checked against each day's low/high; a day that
    touches the stop counts as stopped, even the entry day, whose low may
    have come before the breakout (conservative; the random comparison
    trades get the same treatment); otherwise the position is closed at
    the close after --max-hold days (live: --max-hold-days);
  * costs: --cost-bp per side (default 5 bp, as in RESULTS.md).
Daily bars cannot tell whether the breakout came in the first hour, so this
measures "breakout at any time during the day". Treat it as the upper
bound for the first-hour version.

Usage (from the repo folder):
    python patterns\\backtest_patterns.py
    python patterns\\backtest_patterns.py --patterns double_bottom,cup_handle
    python patterns\\backtest_patterns.py --split 2015-01-01 --cost-bp 5

Output: output\\patterns\\trades.csv (every trade with its random-entry
baseline) and output\\patterns\\summary.csv, also printed.
Caveat: the universe is today's S&P 500 names (survivorship bias); the
pattern-vs-random comparison is like-for-like, the absolute level is not.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import detect as D                                                # noqa: E402
import common_data as cd                                          # noqa: E402

OUT = os.path.join(ROOT, "output", "patterns")


def simulate_exit(d, j, fill, stop, target, O, H, L, C, max_hold):
    """Exit price and bars held for a position entered on bar j at fill.
    Daily bars cannot show whether the entry day's low (for a long) came
    before or after the breakout, so a day that touches the stop counts as
    stopped, and a day touching both levels counts as a stop (conservative).
    The random baseline below faces exactly the same rule, so the
    comparison stays fair."""
    n = len(C)
    for k in range(j, min(j + max_hold + 1, n)):
        if k > j:
            if d * (O[k] - stop) <= 0:
                return O[k], k - j                   # gapped through the stop
            if d * (O[k] - target) >= 0:
                return O[k], k - j                   # gapped through the target
        if d * ((L[k] if d > 0 else H[k]) - stop) <= 0:
            return stop, k - j
        if d * ((H[k] if d > 0 else L[k]) - target) >= 0:
            return target, k - j
    k = min(j + max_hold, n - 1)
    return C[k], k - j


def random_breakout(d, r, dist, sp, tp, O, H, L, C):
    """Entry on bar r for a breakout level placed `dist` (fraction) from the
    previous close, with the same entry rules as detect.entry_on_bar.
    Returns (fill, stop, target) or None if it did not trigger."""
    trig = C[r - 1] * (1 + dist)
    stop, target = trig * (1 - d * sp), trig * (1 + d * tp)
    lvl = trig * (1 + d * D.P["confirm_frac"])
    if d * (O[r] - lvl) >= 0:
        if d * (O[r] - trig) > D.P["max_gap_frac"] * d * (target - trig):
            return None
        fill = O[r]
    elif d * ((H[r] if d > 0 else L[r]) - lvl) >= 0:
        fill = lvl
    else:
        return None
    if not (d * (target - fill) > 0 and d * (fill - stop) > 0):
        return None
    return fill, stop, target


def run_ticker(job):
    ticker, data_dir, patterns, max_hold, cost, draws, seed = job
    try:
        df = cd.load_ohlc(ticker, data_dir)[["Open", "High", "Low", "Close"]]
    except Exception:
        return []
    start = cd.first_valid_open_date(df)               # drop synthetic opens
    if start is None:
        return []
    df = df.loc[start:].dropna()
    df = df[(df > 0).all(axis=1)]
    if len(df) < 300:
        return []
    O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    dates = df.index
    n = len(C)
    rng = np.random.default_rng(seed)

    setups = D.find_setups(df, patterns)
    # first event for each setup; then resolve two-sided groups
    events = {}
    for s in setups:
        for j in range(s.armed + 1, min(s.expiry, n - 1) + 1):
            ev = D.entry_on_bar(s, j, O, H, L, C)
            if ev is not None:
                if ev[0] == "enter":
                    events.setdefault(s.group, []).append((j, s, ev))
                break
    by_group = {}
    for s in setups:
        by_group.setdefault(s.group, []).append(s)
    rows = []
    for grp, evs in events.items():
        # the side that broke first; same day: the one nearer the open
        evs.sort(key=lambda e: (e[0], abs(O[e[0]] - e[2][1])))
        j, s, (_, fill, stop, target) = evs[0]
        d = s.direction
        exit_px, held = simulate_exit(d, j, fill, stop, target, O, H, L, C, max_hold)
        ret = d * (exit_px / fill - 1) - 2 * cost
        # baseline: the same breakout level(s) at the same distance from the
        # previous close, with the same stop/target distances, on random
        # days of this stock. Two-sided patterns get both levels and the
        # same first-side-wins rule, so the comparison is like for like.
        geo = []
        for sib in by_group[grp]:
            t_, st_, tg_ = sib.levels(j)
            geo.append((sib.direction, t_ / C[j - 1] - 1,
                        abs(t_ - st_) / t_, abs(tg_ - t_) / t_))
        base, tries = [], 0
        while len(base) < draws and tries < 60 * draws:
            tries += 1
            r = int(rng.integers(30, n - 2))
            hits = []
            for d_, dist, sp, tp in geo:
                e = random_breakout(d_, r, dist, sp, tp, O, H, L, C)
                if e is not None:
                    hits.append((abs(O[r] - e[0]), d_, e))
            if not hits:
                continue
            _, d_, (f, st, tg) = min(hits, key=lambda h: h[0])
            ex, _ = simulate_exit(d_, r, f, st, tg, O, H, L, C, max_hold)
            base.append(d_ * (ex / f - 1) - 2 * cost)
        rows.append(dict(ticker=ticker, pattern=s.pattern, direction=d,
                         armed=dates[s.armed].date(), entry=dates[j].date(),
                         fill=fill, stop=stop, target=target, exit=exit_px,
                         bars=held, ret=ret, risk=abs(fill - stop) / fill,
                         baseline=float(np.mean(base)) if base else np.nan))
    return rows


def summarize(tr, split):
    tr = tr.copy()
    tr["period"] = np.where(pd.to_datetime(tr["entry"]) < pd.Timestamp(split),
                            f"before {split[:4]}", f"{split[:4]} on")
    tr["edge"] = tr["ret"] - tr["baseline"]
    tr["win"] = tr["ret"] > 0
    out = []
    for (pat, per), g in tr.groupby(["pattern", "period"]):
        e = g["edge"].dropna()
        se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan
        out.append(dict(pattern=pat, period=per, trades=len(g),
                        long_share=(g["direction"] > 0).mean(),
                        win_rate=g["win"].mean(),
                        mean_ret_bp=1e4 * g["ret"].mean(),
                        random_bp=1e4 * g["baseline"].mean(),
                        edge_bp=1e4 * e.mean(),
                        t_stat=e.mean() / se if se and se > 0 else np.nan,
                        avg_days=g["bars"].mean()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=cd.DATA_DIR)
    ap.add_argument("--patterns", default="all")
    ap.add_argument("--max-hold", type=int, default=10,
                    help="days before closing at the close (live: --max-hold-days)")
    ap.add_argument("--cost-bp", type=float, default=5.0, help="per side")
    ap.add_argument("--draws", type=int, default=5, help="random entries per trade")
    ap.add_argument("--split", default="2015-01-01",
                    help="report before / after this date separately")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="first N tickers (quick test)")
    args = ap.parse_args()
    patterns = None if args.patterns == "all" else args.patterns.split(",")
    tickers = cd.list_tickers(args.data_dir)
    if args.limit:
        tickers = tickers[:args.limit]
    jobs = [(t, args.data_dir, patterns, args.max_hold, args.cost_bp / 1e4,
             args.draws, k) for k, t in enumerate(tickers)]
    print(f"Scanning {len(tickers)} tickers with {args.workers} workers ...", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(run_ticker, jobs, chunksize=4)):
            rows += r
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(tickers)} tickers, {len(rows)} trades", flush=True)
    if not rows:
        sys.exit("No trades. Check --data-dir.")
    tr = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tr.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    sm = summarize(tr, args.split)
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    for per, g in sm.groupby("period", sort=False):
        print(f"\n=== {per} (net of {args.cost_bp:g} bp per side; edge = pattern "
              f"minus random entries with the same stop/target) ===")
        print(g.drop(columns="period").sort_values("edge_bp", ascending=False)
              .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    # patterns that hold up in BOTH periods
    piv = sm.pivot(index="pattern", columns="period", values="t_stat")
    if piv.shape[1] == 2:
        good = piv[(piv.iloc[:, 0] > 2) & (piv.iloc[:, 1] > 2)].index.tolist()
        print("\nEdge with t > 2 in both periods: " + (", ".join(good) or "none"))
        print("Trade only these live: python patterns\\pattern_trade.py --execute "
              f"--patterns {','.join(good) if good else '<list>'}")
    print(f"\nSaved {len(tr):,} trades to {os.path.join(OUT, 'trades.csv')}")


if __name__ == "__main__":
    main()
