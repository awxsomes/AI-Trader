"""Rebalance an Alpaca PAPER account to the CNN+Transformer stat-arb targets.

Run once per trading day, ideally between 3:30 and 3:50 PM US Eastern so
the latest price is close to the closing price the model was trained on.

Usage (from the papers_impl folder):
    python live\\paper_trade.py              # dry run: show targets/orders only
    python live\\paper_trade.py --execute    # actually place paper orders

Alpaca keys are read from the environment variables ALPACA_API_KEY and
ALPACA_SECRET_KEY. This script only ever connects to the PAPER endpoint.
Without keys, a dry run still works and assumes a $100,000 account.
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

LOG_DIR = os.path.join(HERE, "logs")
CORR_WINDOW, L_WINDOW = lc.CORR_WINDOW, lc.L_WINDOW


# ---------------------------------------------------------------------------
# Alpaca helpers
# ---------------------------------------------------------------------------

def connect():
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(key, secret, paper=True)      # PAPER ONLY


def alpaca_assets(client):
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    assets = client.get_all_assets(GetAssetsRequest(
        status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    return {a.symbol: a for a in assets}


def current_positions(client):
    pos = {}
    for p in client.get_all_positions():
        q = float(p.qty)
        pos[p.symbol] = q          # qty is negative for shorts
    return pos


def submit(client, symbol, qty, side):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    if float(qty).is_integer():
        qty = int(qty)               # shorts must be whole shares
    req = MarketOrderRequest(
        symbol=symbol, qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY)
    return client.submit_order(req)


def wait_for_fills(client, timeout=180):
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    t0 = time.time()
    while time.time() - t0 < timeout:
        open_orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, limit=500))
        if not open_orders:
            return 0
        time.sleep(5)
    return len(open_orders)


# ---------------------------------------------------------------------------
# Order planning
# ---------------------------------------------------------------------------

def plan_orders(target_qty: dict, current_qty: dict, prices: dict,
                min_trade: float):
    """Split the rebalance into two phases so no single order flips a
    position from long to short (Alpaca rejects that):
      phase 1: reduce / close existing positions,
      phase 2: open / increase positions.
    """
    phase1, phase2 = [], []
    for sym in sorted(set(target_qty) | set(current_qty)):
        cur = current_qty.get(sym, 0.0)
        tgt = target_qty.get(sym, 0.0)
        if cur == tgt:
            continue
        px = prices.get(sym, np.nan)
        if tgt != 0 and np.isfinite(px) and abs(tgt - cur) * px < min_trade:
            continue                                   # not worth trading
        same_side = np.sign(cur) == np.sign(tgt)
        if cur != 0 and (tgt == 0 or not same_side):
            # close the existing position fully, then open the new side
            phase1.append((sym, abs(cur), "sell" if cur > 0 else "buy"))
            if tgt != 0:
                phase2.append((sym, abs(tgt), "buy" if tgt > 0 else "sell"))
        elif cur != 0 and abs(tgt) < abs(cur):
            phase1.append((sym, abs(cur) - abs(tgt), "sell" if cur > 0 else "buy"))
        else:
            phase2.append((sym, abs(tgt) - abs(cur), "buy" if tgt > 0 else "sell"))
    return phase1, phase2


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="place orders on the Alpaca paper account")
    ap.add_argument("--gross", type=float, default=1.0,
                    help="gross exposure as a multiple of equity (default 1.0)")
    ap.add_argument("--min-trade", type=float, default=25.0,
                    help="skip trades smaller than this many dollars")
    ap.add_argument("--source", choices=["yahoo", "local"], default="yahoo")
    ap.add_argument("--data-dir", default=os.environ.get("TRADING_DATA_DIR"))
    ap.add_argument("--allow-closed", action="store_true",
                    help="with --execute, queue orders for the next open "
                         "when the market is closed")
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    today = dt.datetime.now().strftime("%Y-%m-%d_%H%M")   # log file prefix

    # 1. model
    model, meta = lc.load_model()
    print(f"Model trained {meta.get('trained_at', '?')} on "
          f"{meta.get('train_start', '?')} .. {meta.get('train_end', '?')}")
    if meta.get("train_end"):
        age = (dt.date.today() - dt.date.fromisoformat(meta["train_end"])).days
        if age > 45:
            print(f"  NOTE: model is {age} days old; consider re-running train_live.py")

    # 2. prices and returns
    n_days = CORR_WINDOW + L_WINDOW + 40
    close = lc.get_closes(args.source, n_days, args.data_dir)
    last = close.index[-1].date()
    print(f"Prices through {last} ({close.shape[1]} stocks)")
    if (dt.date.today() - last).days > 4:
        print("  WARNING: price data is stale; signals will be out of date.")
    returns = lc.closes_to_returns(close)

    # 3. target weights
    w = lc.target_weights(returns, model)
    prices = close.iloc[-1]

    # 4. account
    client = connect()
    if client is None:
        if args.execute:
            sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to place orders.")
        print("No Alpaca keys found: dry run assuming a $100,000 account.")
        equity, current, assets = 100_000.0, {}, None
    else:
        acct = client.get_account()
        equity = float(acct.equity)
        current = current_positions(client)
        # Only manage the strategy's own universe; leave anything else in the
        # account (manual trades, other bots) untouched.
        universe = {lc.to_alpaca_symbol(t) for t in lc.load_universe()}
        other = sorted(s for s in current if s not in universe)
        if other:
            print(f"  leaving {len(other)} positions outside the strategy "
                  f"universe untouched: {', '.join(other[:10])}"
                  f"{' ...' if len(other) > 10 else ''}")
        current = {s: q for s, q in current.items() if s in universe}
        assets = alpaca_assets(client)
        print(f"Alpaca paper account: equity ${equity:,.0f}, "
              f"{len(current)} open positions")

    # 5. drop what Alpaca can't trade (non-tradable, or shorts on
    #    non-shortable names) and re-normalize
    w.index = [lc.to_alpaca_symbol(t) for t in w.index]
    prices.index = [lc.to_alpaca_symbol(t) for t in prices.index]
    if assets is not None:
        ok = []
        for sym, wt in w.items():
            a = assets.get(sym)
            if a is None or not a.tradable:
                continue
            if wt < 0 and not (a.shortable and a.easy_to_borrow):
                continue
            ok.append(sym)
        dropped = len(w) - len(ok)
        w = w[ok]
        w = w / w.abs().sum()
        if dropped:
            print(f"  dropped {dropped} names Alpaca can't trade / short")

    # 6. dollar and share targets. Longs use fractional shares where Alpaca
    #    allows it (positions are small: ~$200 each on a $100k account);
    #    shorts must be whole shares, rounded to the nearest share.
    dollars = args.gross * equity * w
    tgt_qty = {}
    for s in w.index:
        px = prices.get(s, np.nan)
        if not (np.isfinite(px) and px > 0):
            continue
        raw = dollars[s] / px
        fractionable = assets is None or getattr(assets.get(s), "fractionable", False)
        if raw > 0 and fractionable:
            q = round(raw, 4)
        else:
            q = float(round(raw))
        if q != 0:
            tgt_qty[s] = q
    phase1, phase2 = plan_orders(tgt_qty, current, prices.to_dict(),
                                 args.min_trade)

    long_d = sum(q * prices[s] for s, q in tgt_qty.items() if q > 0)
    short_d = -sum(q * prices[s] for s, q in tgt_qty.items() if q < 0)
    traded = sum(q * prices.get(s, 0) for s, q, _ in phase1 + phase2)
    print(f"\nTarget book: {sum(q > 0 for q in tgt_qty.values())} longs "
          f"(${long_d:,.0f}), {sum(q < 0 for q in tgt_qty.values())} shorts "
          f"(${short_d:,.0f}), net ${long_d - short_d:,.0f}")
    print(f"Orders: {len(phase1)} reduce/close + {len(phase2)} open/increase, "
          f"about ${traded:,.0f} traded ({traded / equity:.0%} of equity)")

    top = w.reindex(w.abs().sort_values(ascending=False).index)[:10]
    print("\nLargest target weights:")
    for s, v in top.items():
        print(f"  {s:6s} {v:+.3%}")

    pd.DataFrame({"weight": w, "price": prices.reindex(w.index),
                  "target_qty": pd.Series(tgt_qty)}).to_csv(
        os.path.join(LOG_DIR, f"{today}_targets.csv"))
    pd.DataFrame(phase1 + phase2, columns=["symbol", "qty", "side"]).assign(
        phase=[1] * len(phase1) + [2] * len(phase2)).to_csv(
        os.path.join(LOG_DIR, f"{today}_orders.csv"), index=False)
    print(f"\nSaved targets and order plan to {LOG_DIR}")

    if not args.execute:
        print("\nDry run only. Add --execute to place these paper orders.")
        return

    # 7. execute
    clock = client.get_clock()
    if not clock.is_open and not args.allow_closed:
        sys.exit("Market is closed. Run during market hours (ideally 3:30-3:50 PM "
                 "ET), or add --allow-closed to queue orders for the next open.")
    failures = []
    print("\nPhase 1: reducing / closing positions ...")
    for sym, qty, side in phase1:
        try:
            submit(client, sym, qty, side)
        except Exception as e:
            failures.append((sym, qty, side, str(e)))
    if clock.is_open:
        left = wait_for_fills(client)
        if left:
            print(f"  {left} orders still open after waiting; continuing")
    else:
        # queued orders can't be sequenced, so skip new legs of flips today
        closing = {s for s, _, _ in phase1}
        phase2 = [o for o in phase2 if o[0] not in closing]
    print("Phase 2: opening / increasing positions ...")
    for sym, qty, side in phase2:
        try:
            submit(client, sym, qty, side)
        except Exception as e:
            failures.append((sym, qty, side, str(e)))
    if clock.is_open:
        wait_for_fills(client)
    n = len(phase1) + len(phase2)
    print(f"Submitted {n - len(failures)} of {n} orders.")
    if failures:
        pd.DataFrame(failures, columns=["symbol", "qty", "side", "error"]).to_csv(
            os.path.join(LOG_DIR, f"{today}_failed_orders.csv"), index=False)
        print(f"  {len(failures)} rejected; see logs\\{today}_failed_orders.csv")


if __name__ == "__main__":
    main()
