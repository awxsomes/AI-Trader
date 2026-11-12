"""Long-only 12-1 momentum with a trend filter, on an Alpaca PAPER account.

Rule (rebalanced once a month, on the last trading day of the month):
  1. rank the universe (tickers.txt) by the return over the past 12 months
     excluding the most recent month (the classic 12-1 momentum signal),
  2. hold the top N names equal-weighted, no shorts, no leverage,
  3. trend filter: if the equal-weighted universe index is below its
     200-day moving average, hold cash instead.

Backtest on output/returns_panel.parquet, 2003-2026, 5 bp per unit turnover:
  top 50 + trend filter: ~17%/yr, vol 16.5%, max drawdown -27%
  (the equal-weighted universe itself: 18%/yr, max drawdown -46%).
Those numbers are inflated by survivorship bias (today's S&P 500 names), so
expect the live result to look like the market with a momentum tilt and a
shallower worst case, not the backtest's absolute level.

USE A SEPARATE PAPER ACCOUNT from paper_trade.py: both scripts manage the
same universe, so on a shared account each would rebalance the other's book.

Usage (from the repo folder):
    python live\\momentum_trade.py              # dry run
    python live\\momentum_trade.py --execute    # place paper orders
    python live\\momentum_trade.py --execute --force-rebalance
Run it daily from a scheduler; it only trades on the last trading day of the
month (or when the trend filter flips), otherwise it exits without orders.
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_common as lc                                         # noqa: E402
from paper_trade import (connect, alpaca_assets, current_positions,   # noqa: E402
                         cancel_open_orders, submit, wait_for_fills,
                         plan_orders, load_state, save_state)

LOG_DIR = os.path.join(HERE, "logs")
LOOKBACK, SKIP, SMA = 252, 21, 200


def is_last_trading_day_of_month(client, today):
    """True if the next market open is in a later month."""
    cal = client.get_calendar()
    days = sorted(c.date for c in cal if c.date >= today)
    if not days or days[0] != today:
        return False
    return len(days) < 2 or days[1].month != today.month


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--top", type=int, default=50, help="number of holdings")
    ap.add_argument("--no-trend", action="store_true", help="disable the 200d filter")
    ap.add_argument("--min-trade", type=float, default=50.0)
    ap.add_argument("--force-rebalance", action="store_true",
                    help="rebalance today even if it is not month end")
    ap.add_argument("--source", choices=["yahoo", "local"], default="yahoo")
    ap.add_argument("--data-dir", default=os.environ.get("TRADING_DATA_DIR"))
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")

    # 1. prices
    close = lc.get_closes(args.source, LOOKBACK + SKIP + 30, args.data_dir)
    last = close.index[-1].date()
    universe = lc.load_universe()
    coverage = close.iloc[-1].notna().sum() / len(universe)
    print(f"Prices through {last}: {close.shape[1]} stocks, {coverage:.0%} priced today")
    if coverage < 0.95:
        sys.exit("ABORT: too many missing prices (yfinance rate limit?); retry later.")
    if (dt.date.today() - last).days > 4:
        sys.exit("ABORT: price data is stale.")

    # 2. signal: 12-1 momentum, requiring (almost) full history
    hist = close.iloc[-(LOOKBACK + 1):]
    full = hist.notna().mean() > 0.95
    mom = (close.iloc[-1 - SKIP] / close.iloc[-1 - LOOKBACK] - 1)[full].dropna()
    ret = lc.closes_to_returns(close)
    ew_index = (1 + ret.mean(axis=1)).cumprod()
    in_trend = bool(ew_index.iloc[-1] >= ew_index.rolling(SMA).mean().iloc[-1])
    risk_on = in_trend or args.no_trend
    print(f"Trend filter: universe index {'above' if in_trend else 'BELOW'} its "
          f"{SMA}-day average -> {'invested' if risk_on else 'CASH'}")

    # 3. account
    client = connect()
    if client is None:
        if args.execute:
            sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to place orders.")
        equity, current, assets, acct_id = 100_000.0, {}, None, "dry_run"
    else:
        acct = client.get_account()
        acct_id = str(acct.account_number)
        equity = float(acct.equity)
        assets = alpaca_assets(client)
        alp_universe = {lc.to_alpaca_symbol(t) for t in universe}
        current = {s: q for s, q in current_positions(client).items()
                   if s in alp_universe}
        print(f"Alpaca paper account {acct_id}: equity ${equity:,.0f}, "
              f"{len(current)} positions")
    state = load_state("momentum_" + acct_id)

    # 4. when to trade: month end, a trend flip, or --force-rebalance
    due = args.force_rebalance or client is None
    if client is not None and not due:
        due = is_last_trading_day_of_month(client, dt.date.today())
        if state.get("risk_on") is not None and state["risk_on"] != risk_on:
            print("  trend filter flipped since the last rebalance -> trading today")
            due = True
        if not current and risk_on:
            due = True                     # empty account: build the book
    if not due:
        print("Not a rebalance day (last trading day of the month); nothing to do.")
        return

    # 5. targets
    sel = mom.nlargest(args.top).index if risk_on else []
    w = pd.Series(1.0 / len(sel) if len(sel) else 0.0, index=list(sel))
    w.index = [lc.to_alpaca_symbol(t) for t in w.index]
    prices = close.iloc[-1].copy()
    prices.index = [lc.to_alpaca_symbol(t) for t in prices.index]
    if assets is not None:
        w = w[[s for s in w.index if s in assets and assets[s].tradable]]
        if len(w):
            w[:] = 1.0 / len(w)
    tgt_qty = {}
    for s, wt in w.items():
        px = prices.get(s, np.nan)
        if np.isfinite(px) and px > 0:
            raw = wt * equity / px
            frac = assets is None or getattr(assets.get(s), "fractionable", False)
            tgt_qty[s] = round(raw, 4) if frac else float(int(raw))
    phase1, phase2 = plan_orders(tgt_qty, current, prices.to_dict(), args.min_trade)
    traded = sum(q * prices[s] for s, q, _ in phase1 + phase2)
    print(f"\nTarget: {len(tgt_qty)} longs, ${sum(q * prices[s] for s, q in tgt_qty.items()):,.0f}")
    print(f"Orders: {len(phase1)} sell + {len(phase2)} buy, about ${traded:,.0f} "
          f"({traded / equity:.0%} of equity)")
    print("Top momentum names: " + ", ".join(list(w.index[:10])))
    pd.DataFrame({"weight": w, "price": prices.reindex(w.index),
                  "momentum": mom.rename(index=lc.to_alpaca_symbol).reindex(w.index),
                  "target_qty": pd.Series(tgt_qty)}).to_csv(
        os.path.join(LOG_DIR, f"{stamp}_momentum_targets.csv"))
    pd.DataFrame(phase1 + phase2, columns=["symbol", "qty", "side"]).to_csv(
        os.path.join(LOG_DIR, f"{stamp}_momentum_orders.csv"), index=False)

    if not args.execute:
        print("\nDry run only. Add --execute to place these paper orders.")
        return
    clock = client.get_clock()
    if not clock.is_open:
        sys.exit("Market is closed; run during market hours (ideally 3:30-3:50 PM ET).")
    cancel_open_orders(client)
    failures = []
    for batch in (phase1, phase2):
        for sym, qty, side in batch:
            try:
                submit(client, sym, qty, side)
            except Exception as e:
                failures.append((sym, qty, side, str(e)))
        wait_for_fills(client)
        time.sleep(2)
    save_state("momentum_" + acct_id,
               {"risk_on": risk_on, "rebalanced_at": stamp})
    print(f"Submitted {len(phase1) + len(phase2) - len(failures)} of "
          f"{len(phase1) + len(phase2)} orders.")
    if failures:
        pd.DataFrame(failures, columns=["symbol", "qty", "side", "error"]).to_csv(
            os.path.join(LOG_DIR, f"{stamp}_momentum_failed_orders.csv"), index=False)
        print(f"  {len(failures)} rejected; see the failed_orders log.")


if __name__ == "__main__":
    main()
