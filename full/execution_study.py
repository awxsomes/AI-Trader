"""Partial rebalancing toward the model's daily targets.

Each day the portfolio moves a fraction lam of the way from yesterday's
holdings to today's target:
    w_t = w_{t-1} + lam * (target_t - w_{t-1})
lam = 1 is the paper's daily full rebalance. Smaller lam trades less (lower
cost) but follows the signal with a lag; the paper's Section III.M shows the
signal decays slowly (half-life about a week), which is what makes this
worthwhile. Costs as in Section III.J: 5 bp x turnover + 1 bp x short share.

Usage: python full/execution_study.py cnn_ipca5_cost [cnn_pca5 ...]
"""
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "output", "full")
C_TC, C_SHORT = 0.0005, 0.0001
LAMBDAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


def simulate(targets, R, lam):
    T, N = targets.shape
    w = np.zeros(N)
    out = np.zeros((T, 4))
    for t in range(T):
        new = w + lam * (targets[t] - w)
        to = np.abs(new - w).sum()
        sh = np.abs(np.minimum(new, 0)).sum()
        g = new @ R[t]
        out[t] = g, g - C_TC * to - C_SHORT * sh, to, sh
        w = new
    return out


def sr(x):
    return np.sqrt(252) * x.mean() / x.std()


def main(names):
    rows = []
    for name in names:
        d = np.load(os.path.join(OUT, f"{name}_targets.npz"))
        tg, R = d["targets"].astype(np.float64), np.nan_to_num(d["R"]).astype(np.float64)
        dates = pd.DatetimeIndex(d["dates"])
        recent = dates >= pd.Timestamp("2021-01-01")
        for lam in LAMBDAS:
            o = simulate(tg, R, lam)
            gross_exp = np.abs(tg).sum(1).mean()  # = 1 for targets
            rows.append({
                "model": name, "lambda": lam,
                "gross SR": sr(o[:, 0]), "net SR": sr(o[:, 1]),
                "net mu": 252 * o[:, 1].mean(),
                "net SR 2021+": sr(o[recent, 1]),
                "net mu 2021+": 252 * o[recent, 1].mean(),
                "turnover/day": o[:, 2].mean(),
            })
            pd.DataFrame(o, index=dates, columns=["gross", "net", "turnover", "short"]
                         ).to_csv(os.path.join(OUT, f"{name}_lam{lam:g}_returns.csv"))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "execution_study.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(tab.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
