"""First-hour chart-pattern breakout trader on its own Alpaca PAPER account.

Every trading day:
  1. before the open, scan yesterday's daily charts of the universe
     (tickers.txt) for the hard-coded patterns in patterns/detect.py and
     list every setup that is armed for today (breakout level, stop, target);
  2. from the open (9:30 ET) until 60 minutes later, poll prices; when a
     stock breaks through its trigger, buy (or short) it with a BRACKET
     order: entry + take-profit at the pattern target + stop-loss;
  3. at the end of the window, cancel entries that did not fill and either
     keep the positions (default, --exit hold: Alpaca keeps working the
     stop and target after this program stops, and positions older than
     --max-hold-days are closed at a later open) or close everything this
     program opened (--exit flatten).

KEYS: this program uses its OWN key pair so it can never touch the stat-arb
or momentum accounts:
    setx PATTERN_ALPACA_API_KEY "paper-key-id-of-the-pattern-account"
    setx PATTERN_ALPACA_SECRET_KEY "its-secret"
It also refuses to trade an account that live\\paper_trade.py or
live\\momentum_trade.py has used (they leave a file in live\\state\\).

Usage (from the repo folder):
    python patterns\\pattern_trade.py                 # dry run: list today's setups
    python patterns\\pattern_trade.py --watch         # dry run through the hour, no orders
    python patterns\\pattern_trade.py --replay        # after 10:46 ET: replay today's hour
    python patterns\\pattern_trade.py --replay 2026-11-16   # or any past day
    python patterns\\pattern_trade.py --execute       # trade today's first hour
    python patterns\\pattern_trade.py --execute --loop   # stay running, trade every day
    python patterns\\pattern_trade.py --execute --exit flatten
Start it any time before 10:30 ET; before the open it waits. On days the
market is closed it exits (or, with --loop, sleeps until the next open).
Logs: patterns\\logs\\<date>_setups.csv, <date>_orders.csv, session.log.
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import detect as D                                                # noqa: E402

NY = "America/New_York"
LOG_DIR = os.path.join(HERE, "logs")
STATE_DIR = os.path.join(HERE, "state")
OTHER_STATE_DIR = os.path.join(ROOT, "live", "state")
KEY_VAR, SECRET_VAR = "PATTERN_ALPACA_API_KEY", "PATTERN_ALPACA_SECRET_KEY"
HISTORY_DAYS = 400          # calendar days of daily bars to scan


# ---------------------------------------------------------------------------
# Logging and state
# ---------------------------------------------------------------------------

def log(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "session.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state(acct):
    try:
        with open(os.path.join(STATE_DIR, f"{acct}.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"positions": {}}


def save_state(acct, state):
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f"{acct}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

def keys():
    return os.environ.get(KEY_VAR), os.environ.get(SECRET_VAR)


def connect():
    k, s = keys()
    if not k or not s:
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(k, s, paper=True)                   # PAPER ONLY


def data_keys():
    """Market data only needs any Alpaca key pair: the pattern account's,
    else the main ALPACA_API_KEY / ALPACA_SECRET_KEY. (Orders always use
    the pattern account's keys.)"""
    k, s = keys()
    if not k or not s:
        k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    return k, s


def data_client():
    k, s = data_keys()
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(k, s)


def guard_account(acct):
    """Never trade an account the stat-arb or momentum bots use."""
    for name in (f"{acct}.json", f"momentum_{acct}.json"):
        if os.path.exists(os.path.join(OTHER_STATE_DIR, name)):
            sys.exit(f"ABORT: account {acct} is used by another bot "
                     f"(live\\state\\{name} exists). Point {KEY_VAR} / "
                     f"{SECRET_VAR} at the pattern account's keys.")


def alpaca_assets(client):
    raw = client.get("/assets", {"status": "active", "asset_class": "us_equity"})
    return {a["symbol"]: a for a in raw if "symbol" in a}


def positions(client):
    return {p["symbol"]: float(p["qty"]) for p in client.get("/positions")}


def open_orders(client):
    return client.get("/orders", {"status": "open", "limit": 500, "nested": "false"})


def cancel_symbol_orders(client, symbols):
    n = 0
    for o in open_orders(client):
        if o["symbol"] in symbols:
            try:
                client.delete(f"/orders/{o['id']}")
                n += 1
            except Exception as e:
                log(f"  could not cancel order {o['id']} ({o['symbol']}): {e}")
    return n


def close_symbols(client, symbols):
    """Cancel the symbols' bracket legs, then close the positions."""
    symbols = set(symbols)
    if not symbols:
        return
    cancel_symbol_orders(client, symbols)
    time.sleep(2)
    for sym in sorted(symbols):
        try:
            client.delete(f"/positions/{sym}")
            log(f"  closed {sym}")
        except Exception as e:
            log(f"  could not close {sym}: {e}")


def session_today(client):
    """(open, close) of today's session as NY timestamps, or None."""
    today = pd.Timestamp.now(tz=NY).date().isoformat()
    cal = client.get("/calendar", {"start": today, "end": today})
    if not cal or cal[0]["date"] != today:
        return None
    o = pd.Timestamp(f"{today} {cal[0]['open']}", tz=NY)
    c = pd.Timestamp(f"{today} {cal[0]['close']}", tz=NY)
    return o, c


def previous_session(client):
    today = pd.Timestamp.now(tz=NY).date()
    cal = client.get("/calendar", {"start": (today - dt.timedelta(days=10)).isoformat(),
                                   "end": (today - dt.timedelta(days=1)).isoformat()})
    return pd.Timestamp(cal[-1]["date"]).date() if cal else None


def submit_bracket(client, sym, qty, d, limit, stop, target, tif):
    data = {
        "symbol": sym, "qty": str(int(qty)),
        "side": "buy" if d > 0 else "sell",
        "type": "limit", "limit_price": f"{limit:.2f}",
        "time_in_force": tif, "order_class": "bracket",
        "take_profit": {"limit_price": f"{target:.2f}"},
        "stop_loss": {"stop_price": f"{stop:.2f}"},
    }
    return client.post("/orders", data)


# ---------------------------------------------------------------------------
# Daily bars (completed sessions only, never today's partial bar)
# ---------------------------------------------------------------------------

def load_universe():
    with open(os.path.join(ROOT, "tickers.txt")) as f:
        return f.read().split()


def to_alpaca(t):
    return t.replace("-", ".")


def bars_alpaca(symbols, feed, asof=None):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    client = data_client()
    today = (pd.Timestamp(asof, tz=NY) if asof is not None
             else pd.Timestamp.now(tz=NY).normalize())
    start = (today - pd.Timedelta(days=HISTORY_DAYS)).to_pydatetime()
    end = today.to_pydatetime()                      # midnight: excludes today
    frames = []
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                               start=start, end=end, adjustment=Adjustment.ALL,
                               feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
        for attempt in range(4):
            try:
                frames.append(client.get_stock_bars(req).df)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log(f"  daily bars retry {attempt + 1}: {str(e)[:80]}")
                time.sleep(5 * (attempt + 1))
    df = pd.concat(frames)
    out = {}
    for sym, g in df.groupby(level="symbol"):
        g = g.droplevel("symbol")
        g.index = pd.to_datetime(g.index).tz_convert(NY).tz_localize(None).normalize()
        g = g.rename(columns=str.capitalize)[["Open", "High", "Low", "Close"]]
        # Alpaca's `end` is inclusive and today's bar is stamped at midnight,
        # so drop it here: only COMPLETED sessions may feed the scan
        out[sym] = g[g.index < today.tz_localize(None)]
    return out


def _cutoff(asof):
    return (pd.Timestamp(asof) if asof is not None
            else pd.Timestamp.now(tz=NY).normalize().tz_localize(None))


def bars_yahoo(tickers, asof=None):
    import yfinance as yf
    df = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                     progress=False, threads=True, group_by="ticker")
    today = _cutoff(asof)
    out = {}
    for t in tickers:
        try:
            g = df[t][["Open", "High", "Low", "Close"]].dropna()
        except KeyError:
            continue
        g.index = pd.to_datetime(g.index).tz_localize(None).normalize()
        out[to_alpaca(t)] = g[g.index < today]
    return out


def bars_local(tickers, data_dir, asof=None):
    import common_data as cd
    today = _cutoff(asof)
    out = {}
    for t in tickers:
        try:
            g = cd.load_ohlc(t, data_dir)[["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            continue
        out[to_alpaca(t)] = g[g.index < today].iloc[-300:]
    return out


def get_bars(args, asof=None):
    """Daily bars for completed sessions before `asof` (default: today)."""
    tickers = load_universe()
    if args.source == "yahoo":
        return bars_yahoo(tickers, asof)
    if args.source == "local":
        return bars_local(tickers, args.data_dir, asof)
    syms = [to_alpaca(t) for t in tickers]
    try:
        return bars_alpaca(syms, "sip", asof)
    except Exception as e:
        log(f"  SIP daily bars refused ({str(e)[:60]}); falling back to Yahoo")
        return bars_yahoo(tickers, asof)


# ---------------------------------------------------------------------------
# Setups and the live entry rule
# ---------------------------------------------------------------------------

def todays_setups(bars, patterns, last_session):
    """One row per armed setup for today."""
    rows = []
    stale = 0
    for sym, df in bars.items():
        df = df[(df > 0).all(axis=1)]
        if len(df) < 60:
            continue
        if last_session is not None and df.index[-1].date() != last_session:
            stale += 1
            continue
        j = len(df)                               # today's bar index
        for s in D.live_setups(df, patterns):
            trig, stop, target = s.levels(j)
            rows.append(dict(symbol=sym, setup=s, pattern=s.pattern,
                             direction=s.direction, group=f"{sym}:{s.group}",
                             j=j, trigger=trig, stop=stop, target=target,
                             armed_on=df.index[s.armed].date()))
    if stale:
        log(f"  {stale} symbols skipped: last daily bar is not {last_session}")
    return rows


def decide(s, j, o, hi, lo, last):
    """Live version of detect.entry_on_bar: today's open, high and low so
    far plus the last price. Returns None (not triggered yet),
    ("cancel", why) or ("enter", stop, target)."""
    d = s.direction
    trig, stop, target = s.levels(j)
    lvl = trig * (1 + d * D.P["confirm_frac"])
    room = D.P["max_gap_frac"] * d * (target - trig)
    gap = d * (o - lvl) >= 0
    crossed = gap or d * ((hi if d > 0 else lo) - lvl) >= 0
    if not crossed:
        return None
    if gap and d * (o - trig) > room:
        return ("cancel", "opened past the trigger")
    if d * (last - trig) > room:
        return ("cancel", "ran too far before entry")
    if not (d * (target - last) > 0 and d * (last - stop) > 0):
        return ("cancel", "price outside stop/target")
    return ("enter", stop, target)


class Book:
    """Caps and bookkeeping for one trading window. Shared by the live loop
    and --replay, so a replay applies exactly the live rules."""

    def __init__(self, args, equity, busy=(), gross=0.0, n_pos=0):
        self.args, self.equity = args, equity
        self.busy, self.gross, self.n_pos = set(busy), gross, n_pos
        self.per_pattern, self.dead_groups, self.done = {}, set(), set()

    def step(self, watch, px, label, send):
        """One price check. px: symbol -> (open, high, low, last) for the
        session so far. send(entry, row) places the order (or not) and
        returns (status, order_id). Returns the entries planned now."""
        a = self.args
        hits = []
        for k, r in enumerate(watch):
            if k in self.done or r["group"] in self.dead_groups or r["symbol"] not in px:
                continue
            o, hi, lo, last = px[r["symbol"]]
            dec = decide(r["setup"], r["j"], o, hi, lo, last)
            if dec is None:
                continue
            self.done.add(k)
            if dec[0] == "cancel":
                log(f"  {label} skip {r['symbol']} {r['pattern']} "
                    f"{'long' if r['direction'] > 0 else 'short'}: {dec[1]}")
                continue
            # how much of the move to the target is already gone at this
            # price; when several break out in the same check, the caps take
            # the least-chased first (not alphabetical order)
            d = r["direction"]
            trig = r["setup"].levels(r["j"])[0]
            used = d * (last - trig) / max(d * (dec[2] - trig), 1e-9)
            hits.append((used, r["symbol"], k, r, dec, last))
        hits.sort(key=lambda h: (h[0], h[1]))
        out = []
        for used, _, k, r, dec, last in hits:
            if r["group"] in self.dead_groups:
                continue                          # sibling broke out first
            _, stop, target = dec
            d, sym = r["direction"], r["symbol"]
            why = None
            if sym in self.busy:
                why = "already traded/held today"
            elif self.n_pos >= a.max_positions:
                why = f"max positions ({a.max_positions})"
            elif self.per_pattern.get(r["pattern"], 0) >= a.max_per_pattern:
                why = f"max {a.max_per_pattern} {r['pattern']} trades today"
            limit = round(last * (1 + d * a.slippage), 2)
            stop, target = round(stop, 2), round(target, 2)
            qty = 0
            if why is None:
                if d * (target - limit) < 0.02 or d * (limit - stop) < 0.02:
                    why = "stop/target too close after rounding"
                else:
                    qty = math.floor(min(a.risk * self.equity / abs(limit - stop),
                                         a.max_pos_frac * self.equity / limit))
                    if self.gross + qty * limit > a.max_gross * self.equity:
                        qty = math.floor((a.max_gross * self.equity - self.gross) / limit)
                    if qty < 1:
                        why = "no room left (gross exposure cap)"
            side = "BUY" if d > 0 else "SHORT"
            if why is not None:
                log(f"  {label} {side} {sym} {r['pattern']} triggered but skipped: {why}")
                continue
            log(f"  {label} {side} {qty} {sym} @ limit {limit:.2f}  stop {stop:.2f}  "
                f"target {target:.2f}  [{r['pattern']}, armed {r['armed_on']}]")
            entry = dict(time=label, symbol=sym, pattern=r["pattern"], side=side,
                         qty=qty, last=last, limit=limit, stop=stop, target=target,
                         trigger=round(r["trigger"], 2))
            status, oid = send(entry, r)
            if not status.startswith("rejected"):
                self.busy.add(sym)
                self.n_pos += 1
                self.gross += qty * limit
                self.per_pattern[r["pattern"]] = self.per_pattern.get(r["pattern"], 0) + 1
            entry.update(status=status, order_id=oid)
            out.append(entry)
        return out


def snapshots(symbols, feed):
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.enums import DataFeed
    client = data_client()
    out = {}
    for i in range(0, len(symbols), 200):
        req = StockSnapshotRequest(symbol_or_symbols=symbols[i:i + 200],
                                   feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)
        try:
            res = client.get_stock_snapshot(req)
        except Exception as e:
            log(f"  price snapshot failed ({str(e)[:80]}); retrying next poll")
            continue
        today = pd.Timestamp.now(tz=NY).date()
        for sym, sn in res.items():
            b, tr = getattr(sn, "daily_bar", None), getattr(sn, "latest_trade", None)
            if b is None or tr is None:
                continue
            if pd.Timestamp(b.timestamp).tz_convert(NY).date() != today:
                continue                          # no trade yet today
            out[sym] = (float(b.open), float(b.high), float(b.low), float(tr.price))
    return out


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------

def run_session(args, client):
    now = pd.Timestamp.now(tz=NY)
    stamp = now.strftime("%Y-%m-%d")
    patterns = (None if args.patterns == "all"
                else [p.strip() for p in args.patterns.split(",")])

    # --- dry run without keys: just list today's setups ------------------
    if client is None:
        bars = get_bars(args)
        rows = todays_setups(bars, patterns, None)
        report_setups(rows, stamp)
        log("No pattern-account keys set: listing only. Set "
            f"{KEY_VAR} and {SECRET_VAR} to watch prices or trade.")
        return

    sess = session_today(client)
    if sess is None:
        log("Market closed today; nothing to do.")
        return
    open_t, close_t = sess
    end_t = open_t + pd.Timedelta(minutes=args.window_minutes)
    if now >= end_t:
        log(f"Trading window ({open_t:%H:%M}-{end_t:%H:%M} ET) is over for today.")
        return

    acct = client.get_account()
    acct_id = str(acct.account_number)
    guard_account(acct_id)
    state = load_state(acct_id)
    log(f"Pattern account {acct_id}: equity ${float(acct.equity):,.0f}; "
        f"window {open_t:%H:%M}-{end_t:%H:%M} ET; exit={args.exit}; "
        f"{'LIVE PAPER ORDERS' if args.execute else 'dry run'}")

    # --- prepare before the open --------------------------------------------
    bars = get_bars(args)
    last_session = previous_session(client)
    rows = todays_setups(bars, patterns, last_session)
    assets = alpaca_assets(client)
    usable = []
    for r in rows:
        a = assets.get(r["symbol"])
        if a is None or not a.get("tradable"):
            continue
        if r["direction"] < 0 and not (a.get("shortable") and a.get("easy_to_borrow")):
            continue
        usable.append(r)
    report_setups(usable, stamp)
    if not usable:
        log("No armed setups today.")

    # --- wait for the open -----------------------------------------------------
    while pd.Timestamp.now(tz=NY) < open_t:
        wait = (open_t - pd.Timestamp.now(tz=NY)).total_seconds()
        log(f"waiting {wait / 60:.0f} min for the open")
        time.sleep(min(max(wait, 1), 300))

    # --- housekeeping at the open: forget closed trades, age out old ones ----
    held = positions(client)
    mine = state.setdefault("positions", {})
    for sym in [s for s in mine if s not in held]:
        log(f"  {sym}: closed by its stop or target since last session")
        mine.pop(sym)
    foreign = [s for s in held if s not in mine]
    if foreign:
        log(f"  note: {len(foreign)} positions not opened by this program "
            f"({', '.join(foreign[:8])}...); they are left alone")
    if args.exit == "hold" and args.execute:
        old = []
        for sym, info in mine.items():
            entered = dt.date.fromisoformat(info["entered"])
            df = bars.get(sym)
            if df is not None:              # completed sessions since entry
                age = int((df.index.date > entered).sum())
            else:                           # not in the universe any more
                age = len(pd.bdate_range(entered, pd.Timestamp.now(tz=NY).date())) - 1
            if age >= args.max_hold_days:
                old.append(sym)
        if old:
            log(f"closing {len(old)} positions held {args.max_hold_days}+ days: {old}")
            close_symbols(client, old)
            for s in old:
                mine.pop(s, None)
            held = positions(client)
    if args.execute:
        save_state(acct_id, state)

    # --- watch the first hour --------------------------------------------------
    equity = float(client.get_account().equity)
    busy = set(held) | {o["symbol"] for o in open_orders(client)}
    gross = sum(abs(q) * (bars[s]["Close"].iloc[-1] if s in bars else 0)
                for s, q in held.items())
    n_pos = len(held)
    orders = []
    tif = "gtc" if args.exit == "hold" else "day"
    watch = [r for r in usable if r["symbol"] not in busy]
    log(f"watching {len(watch)} setups on {len({r['symbol'] for r in watch})} "
        f"symbols until {end_t:%H:%M} ET")
    book = Book(args, equity, busy, gross, n_pos)

    def send(e, r):
        if not args.execute:
            return "dry_run", ""
        try:
            res = submit_bracket(client, e["symbol"], e["qty"], r["direction"],
                                 e["limit"], e["stop"], e["target"], tif)
            oid = res.get("id", "")
            mine[e["symbol"]] = {"entered": stamp, "pattern": r["pattern"],
                                 "direction": r["direction"], "order_id": oid}
            save_state(acct_id, state)
            return res.get("status", "submitted"), oid
        except Exception as ex:
            log(f"    order rejected: {ex}")
            return f"rejected: {str(ex)[:120]}", ""

    while pd.Timestamp.now(tz=NY) < end_t and len(book.done) < len(watch):
        px = snapshots(sorted({r["symbol"] for r in watch}), args.feed)
        orders += book.step(watch, px, pd.Timestamp.now(tz=NY).strftime("%H:%M:%S"), send)
        time.sleep(args.poll_seconds)

    # --- end of the window -------------------------------------------------------
    log(f"window over: {len(orders)} entries sent")
    if args.execute:
        # entries that never filled: cancel them (their stop/target legs
        # go with them); partly filled ones keep working
        time.sleep(5)
        for x in orders:
            if not x["order_id"]:
                continue
            try:
                o = client.get(f"/orders/{x['order_id']}")
            except Exception as e:
                log(f"  could not look up order for {x['symbol']}: {e}")
                continue
            x["status"] = o.get("status", x["status"])
            x["filled_qty"] = o.get("filled_qty")
            x["filled_avg_price"] = o.get("filled_avg_price")
            if (o.get("status") in ("new", "accepted", "pending_new")
                    and float(o.get("filled_qty") or 0) == 0):
                try:
                    client.delete(f"/orders/{o['id']}")
                    mine.pop(x["symbol"], None)
                    x["status"] = "cancelled_unfilled"
                    log(f"  cancelled unfilled entry {x['symbol']}")
                except Exception as e:
                    log(f"  could not cancel {x['symbol']}: {e}")
        if args.exit == "flatten":
            log(f"flattening {len(mine)} positions opened by this program")
            close_symbols(client, list(mine))
            mine.clear()
        save_state(acct_id, state)
    if orders:
        pd.DataFrame(orders).to_csv(os.path.join(LOG_DIR, f"{stamp}_orders.csv"),
                                    index=False)


# ---------------------------------------------------------------------------
# Replay: run a past (or today's finished) window on minute bars
# ---------------------------------------------------------------------------

def minute_bars(symbols, day):
    """1-minute bars for `day`, 9:30-16:00 ET: symbol -> DataFrame indexed
    by the bar's START time (NY). Tries SIP, then IEX."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed
    client = data_client()
    start = pd.Timestamp(f"{day} 09:30", tz=NY)
    end = min(pd.Timestamp(f"{day} 16:00", tz=NY),
              pd.Timestamp.now(tz=NY) - pd.Timedelta(minutes=16))
    for feed in (DataFeed.SIP, DataFeed.IEX):
        frames = []
        try:
            for i in range(0, len(symbols), 100):
                req = StockBarsRequest(symbol_or_symbols=symbols[i:i + 100],
                                       timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                       start=start.to_pydatetime(), end=end.to_pydatetime(),
                                       adjustment=Adjustment.ALL, feed=feed)
                frames.append(client.get_stock_bars(req).df)
        except Exception as e:
            log(f"  minute bars refused on {feed} ({str(e)[:70]})")
            continue
        frames = [f for f in frames if len(f)]
        if not frames:
            log(f"  no minute bars on {feed}")
            continue
        log(f"  minute bars from {feed}")
        df = pd.concat(frames)
        out = {}
        for sym, g in df.groupby(level="symbol"):
            g = g.droplevel("symbol").rename(columns=str.capitalize)
            g.index = pd.to_datetime(g.index).tz_convert(NY)
            out[sym] = g[["Open", "High", "Low", "Close"]]
        return out
    return {}


def run_replay(args):
    """What the trader would have done in the window of one day, using the
    same scan, entry rules and caps as live; then how each trade stood at
    that day's close. No orders are sent."""
    day = (pd.Timestamp.now(tz=NY).date() if args.replay == "today"
           else pd.Timestamp(args.replay).date())
    k, s_ = data_keys()
    if not k or not s_:
        sys.exit(f"Replay needs Alpaca data keys: set {KEY_VAR}/{SECRET_VAR} "
                 "(or ALPACA_API_KEY/ALPACA_SECRET_KEY).")
    patterns = (None if args.patterns == "all"
                else [p.strip() for p in args.patterns.split(",")])
    open_t = pd.Timestamp(f"{day} 09:30", tz=NY)
    end_t = open_t + pd.Timedelta(minutes=args.window_minutes)
    if pd.Timestamp.now(tz=NY) < end_t + pd.Timedelta(minutes=16):
        sys.exit(f"The {day} window ends {end_t:%H:%M} ET; Alpaca's free data "
                 "lags 15 minutes, so replay it after "
                 f"{end_t + pd.Timedelta(minutes=16):%H:%M} ET.")
    log(f"REPLAY of {day} {open_t:%H:%M}-{end_t:%H:%M} ET (no orders are sent)")

    bars = get_bars(args, asof=day)
    lasts = pd.Series([df.index[-1].date() for df in bars.values() if len(df)])
    last_session = lasts.mode().iloc[0] if len(lasts) else None
    if last_session is None or last_session >= day:
        sys.exit(f"ABORT: daily bars reach {last_session}, not before {day}; "
                 "the scan would see that day's own prices.")
    log(f"  pattern scan on daily bars through {last_session}")
    rows = todays_setups(bars, patterns, last_session)
    report_setups(rows, f"replay_{day}")
    if not rows:
        log("No armed setups that day.")
        return
    mins = minute_bars(sorted({r["symbol"] for r in rows}), day)
    if not mins:
        sys.exit(f"No minute bars for {day}: was the market open that day?")

    # cumulative open/high/low/last per minute, aligned on one clock
    clock = pd.date_range(open_t, end_t - pd.Timedelta(minutes=1), freq="1min")
    path = {}
    for sym, m in mins.items():
        w = m[m.index < end_t]
        if w.empty:
            continue
        cum = pd.DataFrame({"o": w["Open"].iloc[0], "h": w["High"].cummax(),
                            "l": w["Low"].cummin(), "c": w["Close"]})
        path[sym] = cum.reindex(clock, method="ffill")

    book = Book(args, args.equity)
    entries = []
    for t in clock:
        px = {sym: tuple(p.loc[t]) for sym, p in path.items()
              if not p.loc[t].isna().any()}
        label = (t + pd.Timedelta(minutes=1)).strftime("%H:%M")
        entries += book.step(rows, px, label, lambda e, r: ("replay", ""))

    log(f"  {len(mins)} of {len({r['symbol'] for r in rows})} armed symbols had "
        f"minute bars; {len(book.done)} setups triggered or were skipped, "
        f"{len(rows) - len(book.done)} never reached their trigger")
    if not entries:
        log("No entries in the window.")
        return

    # Follow each trade the way the live program would manage it.
    # Entry day, on minute bars: stop or target first (a minute touching
    # both counts as the stop); with --exit flatten, closed at window end.
    out = []
    for e in entries:
        m = mins[e["symbol"]]
        d = 1 if e["side"] == "BUY" else -1
        fill = e["last"]
        after = m[m.index >= pd.Timestamp(f"{day} {e['time']}", tz=NY)]
        if args.exit == "flatten":
            after = after[after.index < end_t]
        res = dict(result="open", exit=after["Close"].iloc[-1] if len(after) else fill,
                   exit_date=day, days=0)
        for _, b in after.iterrows():
            if d * ((b["Low"] if d > 0 else b["High"]) - e["stop"]) <= 0:
                res.update(result="stopped", exit=e["stop"])
                break
            if d * ((b["High"] if d > 0 else b["Low"]) - e["target"]) >= 0:
                res.update(result="target hit", exit=e["target"])
                break
        else:
            if args.exit == "flatten":
                res["result"] = f"closed {end_t:%H:%M}"
        out.append(dict(e, d=d, fill=fill, **res))

    # Later days, on daily bars (hold mode): the stop and target stay on at
    # Alpaca; after --max-hold-days full sessions the position is closed at
    # the next open, as the live program does.
    still = [x for x in out if x["result"] == "open"]
    if still:
        later = daily_after(sorted({x["symbol"] for x in still}), day)
        for x in still:
            b = later.get(x["symbol"])
            if b is None or b.empty:
                x["result"] = "still open"
                continue
            d = x["d"]
            x.update(result="still open", exit=b["Close"].iloc[-1],
                     exit_date=b.index[-1].date(), days=len(b))
            for n, (dte, r) in enumerate(b.iterrows(), start=1):
                hit = None
                if n > args.max_hold_days:
                    hit = (f"closed after {args.max_hold_days} days", r["Open"])
                elif d * (r["Open"] - x["stop"]) <= 0:
                    hit = ("stopped (gap)", r["Open"])
                elif d * (r["Open"] - x["target"]) >= 0:
                    hit = ("target hit (gap)", r["Open"])
                elif d * ((r["Low"] if d > 0 else r["High"]) - x["stop"]) <= 0:
                    hit = ("stopped", x["stop"])
                elif d * ((r["High"] if d > 0 else r["Low"]) - x["target"]) >= 0:
                    hit = ("target hit", x["target"])
                if hit:
                    x.update(result=hit[0], exit=hit[1], exit_date=dte.date(), days=n)
                    break

    for x in out:
        x["pnl"] = round(x["d"] * x["qty"] * (x["exit"] - x["fill"]), 2)
        x["fill"], x["exit"] = round(x["fill"], 2), round(x["exit"], 2)
    df = pd.DataFrame(out).drop(columns=["d"])
    df.to_csv(os.path.join(LOG_DIR, f"replay_{day}_orders.csv"), index=False)
    cols = ["time", "side", "qty", "symbol", "pattern", "fill", "stop", "target",
            "result", "exit_date", "days", "exit", "pnl"]
    print("\n" + df[cols].to_string(index=False))
    done = ~df["result"].isin(["still open", "open"])
    print(f"\n{len(df)} entries; closed: {int(done.sum())} "
          f"(P&L ${df.loc[done, 'pnl'].sum():,.2f}); still open: {int((~done).sum())} "
          f"(marked at the latest close: ${df.loc[~done, 'pnl'].sum():,.2f}); "
          f"total ${df['pnl'].sum():,.2f} on ${args.equity:,.0f}")


def daily_after(symbols, day):
    """Daily bars for the sessions after `day`, up to now (the current
    session's bar, if any, is partial)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    client = data_client()
    start = pd.Timestamp(day, tz=NY) + pd.Timedelta(days=1)
    end = pd.Timestamp.now(tz=NY) - pd.Timedelta(minutes=16)
    if end <= start:
        return {}
    for feed in (DataFeed.SIP, DataFeed.IEX):
        try:
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                                   start=start.to_pydatetime(), end=end.to_pydatetime(),
                                   adjustment=Adjustment.ALL, feed=feed)
            df = client.get_stock_bars(req).df
        except Exception as e:
            log(f"  daily bars after {day} refused on {feed} ({str(e)[:60]})")
            continue
        out = {}
        for sym, g in df.groupby(level="symbol"):
            g = g.droplevel("symbol").rename(columns=str.capitalize)
            g.index = pd.to_datetime(g.index).tz_convert(NY).tz_localize(None).normalize()
            out[sym] = g[g.index > pd.Timestamp(day)][["Open", "High", "Low", "Close"]]
        return out
    return {}


def report_setups(rows, stamp):
    os.makedirs(LOG_DIR, exist_ok=True)
    if not rows:
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "setup"} for r in rows])
    df["side"] = np.where(df["direction"] > 0, "long", "short")
    df = df.drop(columns=["direction", "j"])
    df.to_csv(os.path.join(LOG_DIR, f"{stamp}_setups.csv"), index=False)
    counts = df["pattern"].value_counts()
    log(f"{len(df)} armed setups on {df['symbol'].nunique()} symbols: "
        + ", ".join(f"{p} {n}" for p, n in counts.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true", help="place paper orders")
    ap.add_argument("--watch", action="store_true",
                    help="dry run through the trading window (needs keys)")
    ap.add_argument("--loop", action="store_true", help="run every trading day")
    ap.add_argument("--replay", nargs="?", const="today", default=None, metavar="DATE",
                    help="replay a finished window on minute bars, no orders: "
                         "--replay (today) or --replay 2026-11-16")
    ap.add_argument("--equity", type=float, default=100_000,
                    help="account size assumed by --replay")
    ap.add_argument("--exit", choices=["hold", "flatten"], default="hold",
                    help="hold: keep positions with their stop/target (default); "
                         "flatten: close them at the end of the window")
    ap.add_argument("--window-minutes", type=float, default=60)
    ap.add_argument("--poll-seconds", type=float, default=20)
    ap.add_argument("--patterns", default="all",
                    help="comma list from detect.ALL_PATTERNS, or 'all'")
    ap.add_argument("--risk", type=float, default=0.005,
                    help="equity risked per trade at the stop (0.005 = 0.5%%)")
    ap.add_argument("--max-pos-frac", type=float, default=0.10,
                    help="largest single position, share of equity")
    ap.add_argument("--max-gross", type=float, default=1.0,
                    help="total long + short exposure, share of equity")
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--max-per-pattern", type=int, default=2,
                    help="new trades per pattern per day, so every pattern gets tested")
    ap.add_argument("--max-hold-days", type=int, default=10)
    ap.add_argument("--slippage", type=float, default=0.002,
                    help="entry limit this far past the last price")
    ap.add_argument("--source", choices=["alpaca", "yahoo", "local"], default="alpaca",
                    help="daily bars for the pattern scan")
    ap.add_argument("--data-dir", default=os.environ.get("TRADING_DATA_DIR"))
    ap.add_argument("--feed", choices=["iex", "sip"], default="iex",
                    help="live prices; free Alpaca accounts get iex in real time")
    args = ap.parse_args()
    bad = [p for p in (args.patterns.split(",") if args.patterns != "all" else [])
           if p.strip() not in D.ALL_PATTERNS]
    if bad:
        sys.exit(f"Unknown patterns: {bad}. Choose from {D.ALL_PATTERNS}")

    if args.replay:
        run_replay(args)
        return
    client = connect()
    if client is None and (args.execute or args.watch or args.loop):
        sys.exit(f"Set {KEY_VAR} and {SECRET_VAR} (the pattern account's paper keys).")
    if client is not None and not (args.execute or args.watch):
        client_for_run = None                     # plain dry run: list setups only
    else:
        client_for_run = client
    while True:
        try:
            run_session(args, client_for_run)
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            log(f"*** session failed: {e!r} ***")
            if not args.loop:
                raise
        if not args.loop:
            return
        nxt = pd.Timestamp(client.get_clock().next_open).tz_convert(NY)
        wake = nxt - pd.Timedelta(minutes=20)
        log(f"next session opens {nxt:%a %b %d %H:%M} ET; sleeping")
        while pd.Timestamp.now(tz=NY) < wake:
            time.sleep(min(1800, max(30, (wake - pd.Timestamp.now(tz=NY)).total_seconds())))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped by user")
