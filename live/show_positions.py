"""Print the current holdings of an Alpaca paper account, with the dollar
amount per stock for a given budget, so the book can be copied elsewhere.

    python live\\show_positions.py                    # weights only
    python live\\show_positions.py --budget 1000      # + dollars per stock
    python live\\show_positions.py --equal --budget 1000
        (--equal: split the budget evenly across the longs, which is what the
         momentum strategy itself does at each rebalance; without it the
         current drifted weights are used)

Also writes logs\\positions_<account>_<date>.csv.
Uses ALPACA_API_KEY / ALPACA_SECRET_KEY of the account you want to inspect.
"""
import argparse
import datetime as dt
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_trade import connect                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--equal", action="store_true")
    args = ap.parse_args()
    client = connect()
    if client is None:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for the account to show.")
    acct = client.get_account()
    pos = client.get_all_positions()
    if not pos:
        sys.exit(f"Account {acct.account_number} has no positions.")
    df = pd.DataFrame([{"symbol": p.symbol, "qty": float(p.qty),
                        "price": float(p.current_price),
                        "value": float(p.market_value),
                        "pnl": float(p.unrealized_pl)} for p in pos]).set_index("symbol")
    df["weight"] = df["value"] / df["value"].abs().sum()
    df = df.sort_values("value", ascending=False)
    longs, shorts = (df.value > 0).sum(), (df.value < 0).sum()
    if args.budget:
        if args.equal:
            df["buy_$"] = (df.value > 0) * args.budget / max(longs, 1)
        else:
            df["buy_$"] = df["weight"] * args.budget
        df["buy_$"] = df["buy_$"].round(2)
    print(f"Account {acct.account_number}: equity ${float(acct.equity):,.2f}, "
          f"{longs} longs, {shorts} shorts\n")
    print(df.round({"qty": 4, "price": 2, "value": 2, "pnl": 2, "weight": 4}).to_string())
    if shorts and args.budget:
        print("\nNOTE: negative buy_$ rows are SHORTS; a regular brokerage "
              "account (cash, no margin) cannot copy those.")
    out = os.path.join(HERE, "logs",
                       f"positions_{acct.account_number}_{dt.date.today()}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
