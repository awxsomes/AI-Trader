"""Leave this running and it rebalances the Alpaca PAPER account once per
trading day, shortly before the close.

    python live\\run_daily.py                  # trade ~25 min before each close
    python live\\run_daily.py --now            # also trade right away on start
    python live\\run_daily.py --minutes-before-close 20
    python live\\run_daily.py --every-minutes 60   # hourly + near the close

Stop it any time with Ctrl+C. Market hours, early closes and holidays come
from Alpaca's market clock, so no time-zone setup is needed. The computer
must stay on and awake. If the model is over 30 days old, it is retrained
automatically while the market is closed.

Each trade runs live\\paper_trade.py --execute; its output is appended to
live\\logs\\run_daily.log as well as printed.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_common as lc                                         # noqa: E402
from paper_trade import connect                                  # noqa: E402

LOG = os.path.join(HERE, "logs", "run_daily.log")
RETRAIN_AFTER_DAYS = 30


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(name, extra=()):
    cmd = [sys.executable, os.path.join(HERE, name), *extra]
    log("running: " + " ".join(os.path.basename(c) for c in cmd[1:]))
    # time limit so a stuck download can't freeze the loop for hours
    limit = 45 * 60 if name == "train_live.py" else 15 * 60
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, cwd=os.path.dirname(HERE), timeout=limit)
        out, code = p.stdout, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += f"\n{name} timed out after {limit // 60} min and was stopped.\n"
        code = -1
    print(out, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(out)
    if code != 0:
        log(f"*** {name} FAILED (exit code {code}); no trades this round. "
            f"See the lines above for the error. ***")
    else:
        log(f"{name} finished OK")
    return code


def model_age_days():
    try:
        _, meta = lc.load_model()
        return (dt.date.today() - dt.date.fromisoformat(meta["train_end"])).days
    except Exception:
        return 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true",
                    help="trade immediately on start (if the market is open)")
    ap.add_argument("--minutes-before-close", type=float, default=25)
    ap.add_argument("--every-minutes", type=float, default=0,
                    help="also rebalance every N minutes while the market is "
                         "open (e.g. 60 = hourly); 0 = once a day (default)")
    ap.add_argument("--gross", type=float, default=None,
                    help="passed through to paper_trade.py")
    ap.add_argument("--lam", type=float, default=None,
                    help="passed through to paper_trade.py: fraction of the way "
                         "to the target per rebalance (default 0.2 per day for "
                         "the cost-aware model; scaled down automatically when "
                         "--every-minutes rebalances several times a day)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    client = connect()
    if client is None:
        sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY first (README 4.2).")
    extra = ["--execute"]
    if args.gross is not None:
        extra += ["--gross", str(args.gross)]
    every = args.every_minutes
    lam_day = args.lam
    if lam_day is None:
        try:
            lam_day = 0.2 if lc.load_model()[1].get("arch") == "paper_cost" else 1.0
        except FileNotFoundError:
            lam_day = 0.2
    if every > 0 and lam_day < 1.0:
        # several rebalances a day: pick a per-run step that compounds to
        # the same daily speed, (1 - lam_run)^runs = 1 - lam_day
        runs = int(390 // every) + 1
        lam_run = 1 - (1 - lam_day) ** (1 / runs)
    else:
        lam_run = lam_day
    extra += ["--lam", f"{lam_run:.4f}"]

    traded_on = None          # session date of the near-close rebalance
    last_trade_at = None      # time of the last rebalance of any kind
    retrain_tried_on = None
    force_now = args.now
    mode = f"every {every:g} min + near the close" if every > 0 else "once a day"
    log(f"started ({mode}; {lam_run:.1%} of the way to target per rebalance); "
        f"press Ctrl+C to stop")

    while True:
        try:
            clock = client.get_clock()
        except Exception as e:
            log(f"could not reach Alpaca ({e}); retrying in 60s")
            time.sleep(60)
            continue
        now = clock.timestamp

        if clock.is_open:
            session = clock.next_close.date()
            mins_left = (clock.next_close - now).total_seconds() / 60
            before = args.minutes_before_close
            close_due = traded_on != session and mins_left <= before
            since = (None if last_trade_at is None
                     else (now - last_trade_at).total_seconds() / 60)
            interval_due = (every > 0 and mins_left > before and
                            (since is None or since >= every))
            if close_due or interval_due or force_now:
                run_script("paper_trade.py", extra)
                last_trade_at = clock.timestamp
                if close_due or every == 0:
                    traded_on = session      # once-a-day mode: done for today
                force_now = False
                continue
            if traded_on == session:
                wait = (clock.next_close - now).total_seconds() + 60
                log("done for today; waiting for the market to close")
            else:
                wait = (mins_left - before) * 60
                if every > 0:
                    wait = min(wait, (every - since) * 60)
                log(f"market open; next rebalance in {wait / 60:.0f} min")
        else:
            force_now = False
            today = dt.date.today()
            if model_age_days() > RETRAIN_AFTER_DAYS and retrain_tried_on != today:
                retrain_tried_on = today
                run_script("train_live.py")
                continue
            wait = (clock.next_open - now).total_seconds()
            log(f"market closed; next open {clock.next_open:%a %b %d %H:%M %Z}")

        # wake up at least every 30 minutes to re-check the clock
        time.sleep(max(30, min(wait, 1800)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped by user")
