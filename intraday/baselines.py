"""Rule-based intraday strategies on the same family files, costs and
reporting as backtest_intraday.py, so each can be read next to the CNN.

    python intraday\\baselines.py --bar 60 --overnight hold
    python intraday\\baselines.py --bar 60 --overnight hold --only reversal3 ou

Strategies (all long-short weights normalized to sum|w| = 1 unless noted):
  reversal{k}  short residual portfolios that rose over the last k bars, long
               those that fell (k = 1, 3, 6 bars); the simplest version of
               what the CNN is supposed to learn.
  ou           Avellaneda-Lee OU + threshold on the last 30 bars of
               cumulative residuals (the paper's parametric benchmark).
  olmar        On-Line Moving Average Reversion (Li & Hoi 2012), window 5,
               eps 10, long-only, on raw stock returns: the "follow the
               loser" online portfolio algorithm from the NYU Stern notes.
  intraday_mom Gao, Han, Li & Zhou (2018) market intraday momentum: at the
               last bar of the day hold the equal-weight market with the sign
               of the first bar's return; flat otherwise. Long or short the
               whole market, so it is not market-neutral.
  overnight    Knuteson: long the equal-weight market across the overnight
               bar only (hold families; the gap is in the first bar of the
               day), flat all day. Also not market-neutral.
Each strategy is also run with partial rebalancing (--lam 0.3) where that
makes sense, since costs dominate at these frequencies.

Writes output\\intraday\\{tag}_base_{name}_summary.json and _returns.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "full"))
sys.path.insert(0, os.path.join(ROOT, "statarb"))
import backtest_full as bf                                        # noqa: E402
from backtest_intraday import Costs, summarize, OUT               # noqa: E402
from model_full import L_WINDOW                                   # noqa: E402
from models import ou_threshold_weights                          # noqa: E402


def l1(w):
    s = np.abs(w).sum()
    return w / s if s > 1e-12 else w


# ---------------------------------------------------------------------------
# residual-space signals: return w_eps (N,) from information up to bar t-1
# ---------------------------------------------------------------------------

def sig_reversal(k):
    def f(t, eps, X, M, R, first, last):
        if t < k:
            return np.zeros(eps.shape[1])
        return -np.nansum(eps[t - k:t], axis=0) * M[t]
    return f


def sig_ou(t, eps, X, M, R, first, last):
    w = ou_threshold_weights(X[t].astype(np.float64)) * M[t]
    return np.nan_to_num(w)


# ---------------------------------------------------------------------------
# stock-space signals: return stock weights w_R (N,) directly
# ---------------------------------------------------------------------------

class Olmar:
    """OLMAR-1 (Li & Hoi 2012): predict next price relative from a moving
    average, then take the passive-aggressive step toward portfolios whose
    predicted return exceeds eps. Long-only, fully invested."""

    def __init__(self, window=5, eps=10.0):
        self.window, self.eps, self.b = window, eps, None

    def __call__(self, t, eps_, X, M, R, first, last):
        N = R.shape[1]
        if self.b is None:
            self.b = M[t] / max(M[t].sum(), 1)
        if t < self.window + 1:
            return self.b
        # price relatives over the window from returns; p_t / p_{t-w+1} etc.
        rel = 1 + R[t - self.window:t]                        # (w, N)
        cum = np.cumprod(rel[::-1], axis=0)                   # products back from t-1
        x_pred = np.mean(1.0 / np.vstack([np.ones(N), cum[:-1]]), axis=0)  # MA / p_{t-1}
        x_pred = np.where(M[t] > 0, x_pred, 0.0)
        xbar = x_pred[M[t] > 0].mean() if M[t].any() else 0.0
        denom = ((x_pred - xbar) ** 2).sum()
        lam = max(0.0, (self.eps - self.b @ x_pred) / denom) if denom > 1e-12 else 0.0
        b = self.b + lam * (x_pred - xbar)
        b = np.where(M[t] > 0, b, 0.0)
        # project onto the simplex
        u = np.sort(b)[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, N + 1) > (css - 1))[0]
        theta = (css[rho[-1]] - 1) / (rho[-1] + 1) if len(rho) else 0.0
        self.b = np.maximum(b - theta, 0.0)
        self.b = np.where(M[t] > 0, self.b, 0.0)
        s = self.b.sum()
        self.b = self.b / s if s > 0 else M[t] / max(M[t].sum(), 1)
        return self.b.copy()


def sig_intraday_mom(t, eps, X, M, R, first, last):
    """Last bar of the day: equal-weight market with the sign of the first
    bar's market return today."""
    if not last[t]:
        return np.zeros(R.shape[1])
    f = t
    while f > 0 and not first[f]:
        f -= 1
    mkt = R[f][M[f] > 0].mean() if M[f].any() else 0.0
    return np.sign(mkt) * M[t] / max(M[t].sum(), 1)


def sig_overnight(t, eps, X, M, R, first, last):
    """Hold the equal-weight market only across the first bar of the day
    (which contains the overnight gap in hold families)."""
    if not first[t]:
        return np.zeros(R.shape[1])
    return M[t] / max(M[t].sum(), 1)


# ---------------------------------------------------------------------------

@torch.no_grad()
def run(name, signal, space, lam, data, eps, eod, first, costs, t_start):
    X, M, R, W, B = data
    T, N = M.shape
    prev = torch.zeros(1, N)
    rows = []
    w_held = np.zeros(N)
    for t in range(t_start, T):
        w = signal(t, eps, X, M, R, first, eod)
        if space == "residual":
            wR = bf.to_stock_weights(torch.from_numpy(w[None].astype(np.float32)),
                                     torch.from_numpy(W[t:t + 1]), torch.from_numpy(B[t:t + 1]))
            if not torch.isfinite(wR).all():
                wR = torch.zeros(1, N)
            w = wR[0].numpy()
        w = np.nan_to_num(w).astype(np.float32)
        if lam < 1.0 and np.abs(w).sum() > 0:
            w = (1 - lam) * w_held + lam * w
            w = l1(w) if np.abs(w).sum() > 0 else w
        wR = torch.from_numpy(w[None])
        eod_t = torch.tensor([bool(eod[t])])
        g, n, to, sh = costs.pnl(wR, torch.from_numpy(R[t:t + 1]), prev, eod_t)
        rows.append((g.item(), n.item(), to.item(), sh.item()))
        prev = costs.carry(wR, eod_t)
        w_held = prev[0].numpy()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=int, choices=[15, 30, 60], required=True)
    ap.add_argument("--overnight", choices=["hold", "flat"], required=True)
    ap.add_argument("--tc-bp", type=float, default=5.0)
    ap.add_argument("--short-bp-day", type=float, default=1.0)
    ap.add_argument("--lam", type=float, nargs="*", default=[1.0, 0.3],
                    help="partial-rebalance steps to try for the residual strategies")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--start-bar", type=int, default=2000 + L_WINDOW + 1,
                    help="first OOS bar; default matches backtest_intraday.py")
    args = ap.parse_args()

    tag = f"{args.bar}m_{args.overnight}"
    d = np.load(os.path.join(OUT, f"fam_{tag}.npz"), allow_pickle=True)
    dates = pd.DatetimeIndex(d["dates"])
    bars_per_day = int(d["bars_per_day"])
    eod = d["eod"].astype(bool)
    eps = d["eps"]
    first = np.r_[True, eod[:-1]]
    data = bf.prepare_arrays(eps, d["R"], d["W"], d["B"])
    T = eps.shape[0]
    t_start = min(args.start_bar, T - 1)
    costs = Costs(args.tc_bp, args.short_bp_day, bars_per_day, args.overnight == "flat")

    strategies = {}
    for k in (1, 3, 6):
        for lam in args.lam:
            strategies[f"reversal{k}" + ("" if lam == 1 else f"_lam{lam:g}")] = \
                (sig_reversal(k), "residual", lam)
    for lam in args.lam:
        strategies["ou" + ("" if lam == 1 else f"_lam{lam:g}")] = (sig_ou, "residual", lam)
    strategies["olmar"] = (Olmar(), "stock", 1.0)
    strategies["intraday_mom"] = (sig_intraday_mom, "stock", 1.0)
    if args.overnight == "hold":
        strategies["overnight"] = (sig_overnight, "stock", 1.0)
    if args.only:
        strategies = {k: v for k, v in strategies.items()
                      if any(k.startswith(o) for o in args.only)}

    print(f"[{tag}] {T} bars, OOS from {dates[t_start]}: {', '.join(strategies)}", flush=True)
    for name, (sig, space, lam) in strategies.items():
        rows = run(name, sig, space, lam, data, eps, eod, first, costs, t_start)
        df = pd.DataFrame(rows, index=dates[t_start:], columns=["gross", "net", "turnover", "short"])
        df.to_csv(os.path.join(OUT, f"{tag}_base_{name}_returns.csv"))
        s = summarize(df, costs, bars_per_day)
        s.update(bar=args.bar, overnight=args.overnight, tc_bp=args.tc_bp, model=name)
        with open(os.path.join(OUT, f"{tag}_base_{name}_summary.json"), "w") as f:
            json.dump(s, f, indent=1)
        print(f"  {name:16s} gross SR {s['gross_sharpe']:6.2f}  net SR {s['net_sharpe']:6.2f}  "
              f"net {s['net_mean']:+6.1%}/yr  turnover {s['turnover_per_day']:5.2f}/day", flush=True)


if __name__ == "__main__":
    main()
