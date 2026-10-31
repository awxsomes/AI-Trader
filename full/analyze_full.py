"""Summarize the full-paper backtests: gross vs net performance, turnover,
short share, by-year net returns, and cumulative net-return figures."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "output", "full")
RUNS = [("cnn_ipca5_cost", "CNN+Trans, IPCA, cost-trained"),
        ("cnn_pca5_cost", "CNN+Trans, PCA, cost-trained"),
        ("cnn_ipca5", "CNN+Trans, IPCA, gross-trained"),
        ("cnn_pca5", "CNN+Trans, PCA, gross-trained"),
        ("ou_ipca5", "OU+Threshold, IPCA"),
        ("ou_pca5", "OU+Threshold, PCA")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # validated slots 1-4


def stats(x):
    mu, sd = 252 * x.mean(), np.sqrt(252) * x.std()
    return mu / sd if sd > 0 else np.nan, mu


def main():
    rets, rows = {}, []
    for key, label in RUNS:
        p = os.path.join(OUT, f"{key}_returns.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        rets[label] = df
        r = {"model": label}
        for per, sl in [("2002-2026", slice(None)), ("2021-2026", slice("2021", None))]:
            d = df.loc[sl]
            r[f"gross SR {per}"], r[f"gross mu {per}"] = stats(d["gross"])
            r[f"net SR {per}"], r[f"net mu {per}"] = stats(d["net"])
        r["turnover/day"] = df["turnover"].mean()
        r["short share"] = df["short"].mean()
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("model")
    tab.to_csv(os.path.join(OUT, "summary_table.csv"))
    pd.set_option("display.width", 200)
    fmt = tab.copy()
    for c in fmt.columns:
        if " mu " in c:
            fmt[c] = (100 * fmt[c]).map("{:+.1f}%".format)
        elif "SR" in c:
            fmt[c] = fmt[c].map("{:.2f}".format)
        else:
            fmt[c] = fmt[c].map("{:.2f}".format)
    print(fmt.to_string())

    yearly = pd.DataFrame({lab: df["net"].groupby(df.index.year).apply(
        lambda s: (1 + s).prod() - 1) for lab, df in rets.items()})
    yearly.to_csv(os.path.join(OUT, "net_returns_by_year.csv"))
    print("\nNet return by year (%):")
    print((100 * yearly.loc[2015:]).round(1).to_string())

    # cumulative net returns of the CNN variants (OU omitted: its large
    # negative net drift would flatten the scale)
    cnn = [lab for lab in rets if lab.startswith("CNN")]
    for fname, start in [("fig_full_cum_net.png", None),
                         ("fig_full_cum_net_2021.png", "2021")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for lab, col in zip(cnn, COLORS):
            s = rets[lab]["net"].loc[start:] if start else rets[lab]["net"]
            c = (1 + s).cumprod()
            ax.plot(c.index, c.values, color=col, lw=2, label=lab)
            ax.annotate(f"{c.iloc[-1]:.2f}", (c.index[-1], c.iloc[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color="#52514e", va="center")
        ax.axhline(1, color="#9a9994", lw=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of 1, net of costs")
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
