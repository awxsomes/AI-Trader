"""Intraday version of full/backtest_full.py: the cost-aware CNN+Transformer
trading PCA residual portfolios in stock space, with a bar as the period.

Differences from the daily backtest, all driven by the family file:
  - annualization uses bars_per_day * 252 periods per year,
  - the 1 bp/day short charge is spread over the day's bars,
  - overnight=flat: at the last bar of each day the whole book is sold
    (turnover cost on |w|), and the next bar starts from zero.
Everything else (model, Sharpe objective, rolling retraining with warm
starts, sequential out-of-sample evaluation) is shared with the daily code.

    python intraday\\backtest_intraday.py --bar 60 --overnight hold
    python intraday\\backtest_intraday.py --bar 15 --overnight flat --epochs 20

Outputs in output\\intraday\\:  {tag}_returns.csv, {tag}_summary.json and
model_{tag}.pt (loadable by live\\live_common.load_model).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "full"))
sys.path.insert(0, os.path.join(ROOT, "statarb"))
import backtest_full as bf                                        # noqa: E402
from model_full import CNNTransformerPaper, L_WINDOW              # noqa: E402

OUT = os.path.join(ROOT, "output", "intraday")


class Costs:
    def __init__(self, tc_bp, short_bp_day, bars_per_day, flat):
        self.tc = tc_bp * 1e-4
        self.short = short_bp_day * 1e-4 / bars_per_day
        self.flat = flat
        self.periods_per_year = bars_per_day * 252.0

    def pnl(self, wR, R, wR_prev, eod):
        """wR, R, wR_prev: (D, N); eod: (D,) bool, True on the last bar of a
        day. Returns per-bar gross, net, turnover, short exposure."""
        gross = (wR * R).sum(dim=1)
        turnover = (wR - wR_prev).abs().sum(dim=1)
        if self.flat:                     # sell everything after the last bar
            turnover = turnover + wR.abs().sum(dim=1) * eod
        short = wR.clamp(max=0).abs().sum(dim=1)
        net = gross - self.tc * turnover - self.short * short
        return gross, net, turnover, short

    def carry(self, wR, eod):
        """Book carried into the next bar."""
        return wR * (~eod).unsqueeze(1) if self.flat else wR

    def ann_sharpe(self, x):
        return x.mean() / x.std().clamp_min(1e-12) * np.sqrt(self.periods_per_year)


def train(model, data, eod, t0, t1, epochs, costs, store, batch, log=""):
    X, M, R, W, B = data
    N = X.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    eod_t = torch.from_numpy(eod)
    for ep in range(epochs):
        sharpes = []
        for s in range(t0, t1, batch):
            e = min(s + batch, t1)
            Xb = torch.from_numpy(X[s:e]).reshape(-1, L_WINDOW)
            Mb = torch.from_numpy(M[s:e].astype(np.float32))
            prev_in = torch.from_numpy(store["prev_in"][s:e]).reshape(-1)
            out = model(Xb, prev_in).reshape(e - s, N) * Mb
            wR = bf.to_stock_weights(out, torch.from_numpy(W[s:e]), torch.from_numpy(B[s:e]))
            prev_rows = torch.cat([torch.from_numpy(store["wR"][s - 1:s]), wR[:-1]])
            wR_prev = costs.carry(prev_rows, eod_t[s - 1:e - 1])
            _, net, _, _ = costs.pnl(wR, torch.from_numpy(R[s:e]), wR_prev, eod_t[s:e])
            loss = -costs.ann_sharpe(net)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sharpes.append(-loss.item())
            with torch.no_grad():
                store["wR"][s:e] = wR.numpy()
                store["prev_in"][s + 1:e + 1] = bf.normalized_prev(out, Mb).numpy()[: e - s]
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    {log} epoch {ep:3d}  batch Sharpe {np.mean(sharpes):.2f}", flush=True)
    model.eval()


@torch.no_grad()
def evaluate_block(model, data, eod, s, e, costs, state):
    X, M, R, W, B = data
    rows = []
    for t in range(s, e):
        Mt = torch.from_numpy(M[t:t + 1].astype(np.float32))
        out = model(torch.from_numpy(X[t]), state["prev_in"][0]).unsqueeze(0) * Mt
        wR = bf.to_stock_weights(out, torch.from_numpy(W[t:t + 1]), torch.from_numpy(B[t:t + 1]))
        eod_t = torch.tensor([bool(eod[t])])
        g, n, to, sh = costs.pnl(wR, torch.from_numpy(R[t:t + 1]), state["wR"], eod_t)
        rows.append((g.item(), n.item(), to.item(), sh.item()))
        state["wR"] = costs.carry(wR, eod_t)
        state["prev_in"] = bf.normalized_prev(out, Mt)
    return rows


def summarize(df, costs, bars_per_day):
    P = costs.periods_per_year
    out = {}
    for c in ["gross", "net"]:
        mu, sd = P * df[c].mean(), np.sqrt(P) * df[c].std()
        out[f"{c}_sharpe"], out[f"{c}_mean"], out[f"{c}_vol"] = mu / sd, mu, sd
    out["turnover_per_day"] = df["turnover"].mean() * bars_per_day
    out["avg_short_share"] = df["short"].mean()
    daily = df["net"].groupby(df.index.normalize()).sum()
    out["net_sharpe_daily_agg"] = daily.mean() / daily.std() * np.sqrt(252)
    eq = (1 + daily).cumprod()
    out["max_drawdown"] = float((eq / eq.cummax() - 1).min())
    out["days"] = int(len(daily))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=int, choices=[15, 30, 60], required=True)
    ap.add_argument("--overnight", choices=["hold", "flat"], required=True)
    ap.add_argument("--train-window", type=int, default=2000, help="bars")
    ap.add_argument("--retrain", type=int, default=500, help="bars between retrains")
    ap.add_argument("--batch", type=int, default=250, help="bars per gradient step")
    ap.add_argument("--epochs", type=int, default=30, help="first training")
    ap.add_argument("--warm-epochs", type=int, default=8)
    ap.add_argument("--tc-bp", type=float, default=5.0, help="cost per unit turnover")
    ap.add_argument("--short-bp-day", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count())

    tag = f"{args.bar}m_{args.overnight}"
    d = np.load(os.path.join(OUT, f"fam_{tag}.npz"), allow_pickle=True)
    dates = pd.DatetimeIndex(d["dates"])
    bars_per_day = int(d["bars_per_day"])
    eod = d["eod"].astype(bool)
    data = bf.prepare_arrays(d["eps"], d["R"], d["W"], d["B"])
    X, M, R, W, B = data
    T, N = M.shape
    costs = Costs(args.tc_bp, args.short_bp_day, bars_per_day, args.overnight == "flat")
    t_start = args.train_window + L_WINDOW + 1
    if t_start >= T - args.retrain:
        sys.exit(f"only {T} bars; need more than {t_start + args.retrain} "
                 f"(lower --train-window or download more years)")
    print(f"[{tag}] {T} bars ({bars_per_day}/day), {N} symbols; OOS "
          f"{dates[t_start]} .. {dates[-1]} ({(T - t_start) / bars_per_day:.0f} days)",
          flush=True)

    model = CNNTransformerPaper(use_prev=True)
    store = {"wR": np.zeros((T, N), np.float32), "prev_in": np.zeros((T + 1, N), np.float32)}
    state = {"wR": torch.zeros(1, N), "prev_in": torch.zeros(1, N)}
    rows, first = [], True
    for s in range(t_start, T, args.retrain):
        e = min(s + args.retrain, T)
        tic = time.time()
        train(model, data, eod, s - args.train_window, s,
              args.epochs if first else args.warm_epochs, costs, store, args.batch, tag)
        first = False
        rows += evaluate_block(model, data, eod, s, e, costs, state)
        print(f"[{tag}] {dates[s]} .. {dates[e - 1]} done ({time.time() - tic:.0f}s)", flush=True)

    df = pd.DataFrame(rows, index=dates[t_start:], columns=["gross", "net", "turnover", "short"])
    df.to_csv(os.path.join(OUT, f"{tag}_returns.csv"))
    summ = summarize(df, costs, bars_per_day)
    summ.update(bar=args.bar, overnight=args.overnight, tc_bp=args.tc_bp,
                train_window=args.train_window, retrain=args.retrain)
    with open(os.path.join(OUT, f"{tag}_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    torch.save({"state_dict": model.state_dict(),
                "meta": {"arch": "paper_cost", "factors": "PCA5", "bar": args.bar,
                         "overnight": args.overnight, "bars_per_day": bars_per_day,
                         "corr_window": int(d["corr_window"]),
                         "beta_window": int(d["beta_window"]),
                         "train_end": str(dates[-1]), "symbols": list(d["symbols"]),
                         "costs": f"{args.tc_bp}bp turnover + {args.short_bp_day}bp/day short"}},
               os.path.join(OUT, f"model_{tag}.pt"))
    print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()
