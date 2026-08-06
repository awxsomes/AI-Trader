"""
live_trader.py — forward-testing loop for the trained models.

SIMULATION AND PAPER TRADING ONLY. This never places a live-money order: the
Alpaca client is pinned to the paper endpoint and there is no flag to change it.

    # offline: replay historical bars through the real decision loop
    python live_trader.py --mode replay --bars 500

    # paper: real IEX bars, fake money, real order lifecycle
    python live_trader.py --mode paper --cash 500

    # paper but log decisions without sending orders
    python live_trader.py --mode paper --dry-run

HOW THIS DIFFERS FROM THE BACKTEST
----------------------------------
The backtest loads a whole price array and steps an index over it. Live is a
streaming loop: one bar arrives, you compute features from a rolling buffer,
predict once, act once, persist state, and wait for the next bar. The model is
LOADED here, never trained — train with run.py first.

Because it must survive a restart mid-position, all position state is written
to disk after every decision. Killing the process and restarting picks up where
it left off rather than silently thinking it is flat.
"""

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import nn_stock_predictor as N
import dqn_trader as D


STATE_FILE = "models/live_state.json"
LOG_FILE = "models/live_trades.jsonl"


# =============================================================================
# PERSISTENT STATE
# =============================================================================
DEFAULT_STATE = {
    "position": 0,            # 0 flat, 1 long, -1 short
    "position_size": 0.0,     # fraction of equity deployed
    "avg_entry": 0.0,
    "shares": 0.0,
    "holding_time": 0,
    "cash": 0.0,
    "realized_pnl": 0.0,
    "trades_today": 0,
    "session_date": None,
    "bars_seen": 0,
}


def load_state(initial_cash):
    """Resume from disk so a restart mid-position doesn't lose the position."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        print(f"Resumed state: position={state['position']} "
              f"shares={state['shares']:.4f} cash=${state['cash']:,.2f}")
        return state
    state = dict(DEFAULT_STATE)
    state["cash"] = initial_cash
    return state


def save_state(state):
    os.makedirs("models", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_event(record):
    os.makedirs("models", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# =============================================================================
# BAR SOURCES
# =============================================================================
class ReplaySource:
    """
    Feeds historical bars one at a time. Same interface as the live source, so
    the decision loop is identical — this is how you test the live path without
    a broker account or waiting for market hours.
    """

    def __init__(self, config, n_bars=500):
        df, source = N.load_market_data(config)
        self.df = df.tail(n_bars + config.LOOKBACK_WINDOW + 120)
        self.source = source
        self.i = 0
        print(f"Replay: {len(self.df)} bars from {source} data")

    def warmup(self, n):
        """Bars to prime the rolling buffer before the first decision."""
        rows = self.df.iloc[:n]
        self.i = n
        return rows

    def next_bar(self):
        df = self._fetch(500)          # was 5
        if df.empty:
            return None
        df = df.sort_index()
        row = df.iloc[-1]
        if self.last_ts is not None and df.index[-1] <= self.last_ts:
            return None
        self.last_ts = df.index[-1]
        return row

    def wait(self):
        pass  # no waiting in replay


class AlpacaSource:
    """Polls Alpaca for the most recent completed minute bar."""

    def __init__(self, config, poll_seconds=60):
        from api_caller import APICaller
        self.api = APICaller(provider="alpaca")   # paper endpoint by default
        if not self.api.connect():
            raise RuntimeError(
                "Alpaca connection failed. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in your environment.")
        self.symbol = config.TICKER
        self.poll_seconds = poll_seconds
        self.last_ts = None
        self.source = "alpaca_iex"
    def initialize_cursor(self):
        """Set last_ts based on what the live feed actually has right now."""
        df = self._fetch(5)
        if df.empty:
            raise RuntimeError("Live feed returned no bars — check API key and feed=iex")
        self.last_ts = df.index[-1]
        print(f"Live cursor initialized at {self.last_ts}")
        return df
    def _fetch(self, limit):
        bars = self.api.get_bars(self.symbol, timeframe="1Min", limit=limit)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                "c": "Close", "v": "Volume", "t": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
        return df.sort_index()  # never trust API ordering for "latest bar" logic

    def warmup(self, n):
        df = self._fetch(n)
        if len(df) < n:
            raise RuntimeError(f"Only {len(df)} bars available, need {n} to warm up")
        # DON'T set last_ts here — live feed may be delayed
        return df

    def next_bar(self):
        df = self._fetch(5)
        if df.empty:
            print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] df.empty → API returned no bars")
            return None
        print(f"[DEBUG] fetched {len(df)} bars, newest={df.index[-1]}, last_ts={self.last_ts}")
        row = df.iloc[-1]
        if self.last_ts is not None and df.index[-1] <= self.last_ts:
            print(f"[DEBUG] bar is old ({df.index[-1]} <= {self.last_ts}), waiting...")
            return None
        self.last_ts = df.index[-1]
        print(f"[DEBUG] NEW BAR ACCEPTED: {self.last_ts}")
        return row

    def wait(self):
        time.sleep(self.poll_seconds)


# =============================================================================
# FEATURE PIPELINE ON A ROLLING BUFFER
# =============================================================================
class RollingFeatures:
    """
    Holds the last N raw bars and recomputes indicators over the window.

    Indicators need history (ATR/ADX use 14 periods, the percentile ranks use
    100), so the buffer is deliberately much longer than the LSTM's lookback.
    Recomputing over the whole buffer each bar is slightly wasteful but it
    guarantees the live features are produced by exactly the same code as the
    training features — which is the failure mode that quietly ruins live
    deployments.
    """

    def __init__(self, lookback_window, buffer_size=250):
        self.buffer = deque(maxlen=buffer_size)
        self.lookback_window = lookback_window

    def add(self, row):
        self.buffer.append({
            "Open": float(row["Open"]), "High": float(row["High"]),
            "Low": float(row["Low"]), "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
        })

    def ready(self):
        return len(self.buffer) >= 150

    def frame(self):
        df = pd.DataFrame(list(self.buffer))
        return N.add_technical_indicators(df)


def build_sequence(feat_df, bundle):
    """Scale the last `lookback` rows into one LSTM input sequence."""
    full_data, names = N.build_feature_matrix(feat_df)
    if names != bundle["feature_names"]:
        raise RuntimeError(
            "Live feature names differ from training. Retrain or fix the "
            f"pipeline.\ntrain={bundle['feature_names']}\nlive={names}")
    window = full_data[-bundle["lookback_window"]:]
    scaled = bundle["feature_scaler"].transform(window)
    return scaled[np.newaxis, :, :].astype(np.float32)


# =============================================================================
# DECISION LOOP
# =============================================================================
def run(args):
    ncfg = N.Config()
    ncfg.SEED = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dcfg = D.DQNConfig()
    dcfg.INITIAL_CASH = args.cash

    bundle_path = args.bundle
    if not os.path.exists(bundle_path):
        raise SystemExit(
            f"No predictor bundle at {bundle_path}. Train first:\n"
            f"    python run.py pipeline --seeds 1")
    model, bundle = N.load_inference_bundle(bundle_path)
    print(f"Predictor loaded (trained on {bundle['data_source']} "
          f"{bundle['ticker']} {bundle['interval']} data)")

    agent = D.DQNAgent(dcfg)
    if not os.path.exists(args.agent):
        raise SystemExit(f"No agent checkpoint at {args.agent}. Train first.")
    agent.load(args.agent)
    agent.q_network.eval()
    print(f"Agent loaded from {args.agent}")

    source = (ReplaySource(ncfg, args.bars) if args.mode == "replay"
              else AlpacaSource(ncfg))

    state = load_state(args.cash)
    if state["cash"] == 0.0:
        state["cash"] = args.cash
    start_equity = state["cash"]

    feats = RollingFeatures(bundle["lookback_window"])
    warmup_df = source.warmup(150)
    if args.mode == "paper":
        source.initialize_cursor()  # sets last_ts from live reality
    for _, row in warmup_df.iterrows():
        feats.add(row)

    print(f"\n{'='*66}")
    print(f"{'REPLAY' if args.mode=='replay' else 'PAPER'} SESSION | "
          f"cash ${state['cash']:,.2f} | dry_run={args.dry_run}")
    print(f"{'='*66}\n")

    while True:
        row = source.next_bar()
        if row is None:
            if args.mode == "replay":
                break
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No new bar, sleeping...")
            source.wait()
            continue

        feats.add(row)
        state["bars_seen"] += 1
        if not feats.ready():
            continue

        price = float(row["Close"])
        feat_df = feats.frame()
        if len(feat_df) < bundle["lookback_window"]:
            continue

        # --- predict the forward return, convert to a price -----------------
        seq = build_sequence(feat_df, bundle)
        with torch.no_grad():
            raw = model(torch.FloatTensor(seq).to(ncfg.DEVICE)).cpu().numpy()
        pred_return = float(bundle["target_scaler"].inverse_transform(raw)[0, 0])
        predicted_price = price * (1 + pred_return)

        # --- regime + volatility, same definitions as training --------------
        last = feat_df.iloc[-1]
        regime = N.get_regime_features(last)
        closes = feat_df["Close"].values[-11:]
        volatility = float(np.std(np.diff(closes) / closes[:-1]))

        # --- ask the agent --------------------------------------------------
        action, action_name, confidence, size_frac = D.predict_action(
            agent, predicted_price, price,
            purchase_price=state["avg_entry"],
            position=state["position"],
            position_size=state["position_size"],
            holding_time=state["holding_time"],
            volatility=volatility,
            is_chop=regime["is_chop"], is_trending=regime["is_trending"],
            is_low_vol=regime["is_low_vol"], is_high_vol=regime["is_high_vol"])

        # --- risk gates BEFORE anything is sent -----------------------------
        blocked = None
        if not N.is_trading_allowed(last) and action != 0:
            blocked = "regime_filter"
        elif state["trades_today"] >= args.max_trades and action != 0:
            blocked = "max_trades"

        equity = state["cash"] + state["position"] * state["shares"] * price
        if equity < start_equity * (1 - args.max_daily_loss):
            print(f"KILL SWITCH: equity ${equity:,.2f} is more than "
                  f"{args.max_daily_loss*100:.0f}% below session start. Flattening.")
            action, action_name, blocked = (4 if state["position"] == 1 else 8), "FLATTEN", None

        if blocked:
            action, action_name = 0, f"HOLD (blocked: {blocked})"

        # --- apply ----------------------------------------------------------
        executed = apply_action(state, action, price, size_frac, args, equity)

        if state["position"] != 0:
            state["holding_time"] += 1
        equity = state["cash"] + state["position"] * state["shares"] * price

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "bar": state["bars_seen"], "price": round(price, 4),
            "pred_return": round(pred_return, 6),
            "action": action_name, "confidence": round(confidence, 4),
            "executed": executed, "position": state["position"],
            "shares": round(state["shares"], 4),
            "equity": round(equity, 2), "blocked": blocked,
        }
        log_event(record)
        save_state(state)

        if executed or state["bars_seen"] % args.print_every == 0:
            print(f"[{state['bars_seen']:>5}] ${price:>9.2f} | pred {pred_return:+.5f} "
                  f"| {action_name:<24} | pos {state['position']:>2} "
                  f"| equity ${equity:>10,.2f}")

    final_equity = state["cash"] + state["position"] * state["shares"] * price
    print(f"\n{'='*66}")
    print(f"Session end | equity ${final_equity:,.2f} "
          f"({(final_equity/start_equity-1)*100:+.2f}%) | "
          f"trades {state['trades_today']} | bars {state['bars_seen']}")
    print(f"Decision log: {LOG_FILE}")
    print(f"{'='*66}")


def apply_action(state, action, price, size_frac, args, equity):
    """
    Update simulated position state, and send a paper order when live.

    Uses the same signed accounting as the backtest environment: shorts credit
    proceeds to cash, so equity = cash + position*shares*price on both sides.
    """
    cost = D.DQNConfig.TOTAL_COST_PER_TRADE

    def send(side, qty):
        if args.mode != "paper" or args.dry_run:
            return f"{'DRY' if args.dry_run else 'SIM'}:{side}:{qty:.4f}"
        from api_caller import APICaller
        api = APICaller(provider="alpaca")
        api.connect()
        result = api.place_order(symbol=N.Config.TICKER, qty=round(qty, 4),
                                 side=side, order_type="market")
        return f"PAPER:{side}:{result.get('id', 'sent')}"

    # entries
    if action in (1, 2, 3) and state["position"] in (0, 1):
        # Divide by (1+cost) so the fee fits inside available cash — sizing to
        # the full cash balance and then charging the fee overdraws the account.
        notional = min(size_frac * equity, state["cash"] / (1 + cost))
        if notional <= 1e-6:
            return None
        qty = notional / price
        state["cash"] -= notional + notional * cost
        total = state["shares"] + qty
        state["avg_entry"] = ((state["avg_entry"] * state["shares"] + price * qty) / total
                              if total > 0 else price)
        state["shares"] = total
        if state["position"] == 0:
            state["position"], state["holding_time"] = 1, 0
        state["position_size"] = min(state["position_size"] + size_frac, 1.0)
        state["trades_today"] += 1
        return send("buy", qty)

    if action in (5, 6, 7) and state["position"] in (0, -1):
        notional = min(size_frac * equity,
                       D.DQNConfig.MAX_POSITION_SIZE * equity - state["shares"] * price)
        if notional <= 1e-6:
            return None
        qty = notional / price
        state["cash"] += notional - notional * cost
        total = state["shares"] + qty
        state["avg_entry"] = ((state["avg_entry"] * state["shares"] + price * qty) / total
                              if total > 0 else price)
        state["shares"] = total
        if state["position"] == 0:
            state["position"], state["holding_time"] = -1, 0
        state["position_size"] = min(state["position_size"] + size_frac, 1.0)
        state["trades_today"] += 1
        return send("sell", qty)

    # exits
    if action in (4, 8) and state["position"] != 0:
        qty = state["shares"]
        notional = qty * price
        if state["position"] == 1:
            state["cash"] += notional - notional * cost
            pnl = (price - state["avg_entry"]) / state["avg_entry"]
            side = "sell"
        else:
            state["cash"] -= notional + notional * cost
            pnl = (state["avg_entry"] - price) / state["avg_entry"]
            side = "buy"
        state["realized_pnl"] += pnl
        state.update(position=0, position_size=0.0, avg_entry=0.0,
                     shares=0.0, holding_time=0)
        state["trades_today"] += 1
        return send(side, qty)

    return None


def main():
    p = argparse.ArgumentParser(description="Forward-testing loop (paper/replay only)")
    p.add_argument("--mode", choices=["replay", "paper"], default="replay")
    p.add_argument("--cash", type=float, default=500.0)
    p.add_argument("--bars", type=int, default=500, help="replay length")
    p.add_argument("--bundle", default="models/nn_predictor_seed1_bundle.pkl")
    p.add_argument("--agent", default="models/dqn_trader_seed1.pt")
    p.add_argument("--dry-run", action="store_true",
                   help="paper mode but log decisions instead of sending orders")
    p.add_argument("--max-trades", type=int, default=50,
                   help="stop opening new positions after this many trades")
    p.add_argument("--max-daily-loss", type=float, default=0.05,
                   help="flatten and stop if equity falls this fraction")
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42,
                   help="makes replay reproducible")
    p.add_argument("--reset", action="store_true", help="clear saved state first")
    args = p.parse_args()

    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("Cleared saved state")

    run(args)


if __name__ == "__main__":
    main()