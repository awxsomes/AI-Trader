"""Implementation of Knuteson (2020), arXiv:2010.01727,
"Strikingly Suspicious Overnight and Intraday Returns".

Decomposes each day's total close-to-close return into
  overnight:  r_on(t) = Open(t) / Close(t-1) - 1   (previous close -> open)
  intraday:   r_id(t) = Close(t) / Open(t) - 1      (open -> close)
and cumulates (compounds) each component separately, reproducing the
paper's Figure 1 on the stocks in the local data folder (the paper uses
index ETFs; here we use the S&P 500 constituents available locally, plus
an equal-weighted cross-sectional average as a market proxy).

Safeguard: early Yahoo-style data often has synthetic opens
(Open == Close), which would mechanically push the whole daily move into
the overnight component. Each ticker's series therefore starts at the
first date from which opens look genuine (see common_data.first_valid_open_date).

Outputs (written to ./output):
  - overnight_intraday_summary.csv : per-ticker cumulative overnight and
    intraday returns, sample start/end, and number of days
  - fig_overnight_intraday_grid.png : 21-panel figure in the style of the
    paper's Figure 1 (well-known long-history names)
  - fig_overnight_intraday_market.png : equal-weighted average across all
    tickers (market proxy)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_data as cd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_TICKERS = [
    "AAPL", "MSFT", "IBM", "GE", "KO", "JNJ", "PG", "XOM", "JPM",
    "WMT", "MCD", "DIS", "HD", "INTC", "CSCO", "MRK", "PFE", "BA",
    "CAT", "MMM", "T",
]


def overnight_intraday_returns(ticker: str):
    """Return (overnight, intraday) daily simple-return series for a ticker."""
    df = cd.load_ohlc(ticker)
    start = cd.first_valid_open_date(df)
    if start is None:
        return None, None
    df = df.loc[start:]
    prev_close = df["Close"].shift(1)
    r_on = df["Open"] / prev_close - 1.0
    r_id = df["Close"] / df["Open"] - 1.0
    r_on = r_on.replace([np.inf, -np.inf], np.nan).dropna()
    r_id = r_id.replace([np.inf, -np.inf], np.nan).dropna()
    idx = r_on.index.intersection(r_id.index)
    # Clip absurd data errors (e.g. unadjusted split artifacts)
    r_on, r_id = r_on.loc[idx].clip(-0.5, 0.5), r_id.loc[idx].clip(-0.5, 0.5)
    return r_on, r_id


def cumulate(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod() - 1.0


def run():
    tickers = cd.list_tickers()
    rows = []
    cum = {}
    for t in tickers:
        try:
            r_on, r_id = overnight_intraday_returns(t)
        except Exception:
            continue
        if r_on is None or len(r_on) < 500:
            continue
        c_on, c_id = cumulate(r_on), cumulate(r_id)
        cum[t] = (r_on, r_id)
        rows.append({
            "ticker": t,
            "start": r_on.index[0].date(),
            "end": r_on.index[-1].date(),
            "n_days": len(r_on),
            "cum_overnight": c_on.iloc[-1],
            "cum_intraday": c_id.iloc[-1],
            "cum_total_closetoclose": (1 + c_on.iloc[-1]) * (1 + c_id.iloc[-1]) - 1,
            "mean_overnight_bps": 1e4 * r_on.mean(),
            "mean_intraday_bps": 1e4 * r_id.mean(),
        })
    summary = pd.DataFrame(rows).set_index("ticker").sort_index()
    summary.to_csv(os.path.join(OUT_DIR, "overnight_intraday_summary.csv"))

    # ---- 21-panel grid figure (style of the paper's Figure 1) ----
    grid = [t for t in GRID_TICKERS if t in cum]
    n = len(grid)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.1 * nrow), sharex=False)
    axes = np.atleast_2d(axes)
    for i, t in enumerate(grid):
        ax = axes[i // ncol, i % ncol]
        r_on, r_id = cum[t]
        c_on, c_id = cumulate(r_on), cumulate(r_id)
        ax.plot(c_on.index, 100 * c_on.values, color="tab:blue", lw=0.9,
                label="overnight")
        ax.plot(c_id.index, 100 * c_id.values, color="tab:green", lw=0.9,
                label="intraday")
        ax.axhline(0, color="k", lw=0.5)
        ax.text(0.02, 0.92, t, transform=ax.transAxes, fontsize=8, va="top")
        ax.text(0.98, 0.92,
                f"on: {100*c_on.iloc[-1]:+,.0f}%  in: {100*c_id.iloc[-1]:+,.0f}%",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="right")
        ax.set_yscale("symlog", linthresh=100)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, loc="center left")
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Date", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative return (%)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overnight_intraday_grid.png"), dpi=200)
    plt.close(fig)

    # ---- Equal-weighted market proxy ----
    on_panel = pd.DataFrame({t: v[0] for t, v in cum.items()})
    id_panel = pd.DataFrame({t: v[1] for t, v in cum.items()})
    # Require a reasonably broad cross-section before averaging
    enough = on_panel.notna().sum(axis=1) >= 30
    mkt_on = on_panel.loc[enough].mean(axis=1)
    mkt_id = id_panel.loc[enough].mean(axis=1)
    c_on, c_id = cumulate(mkt_on), cumulate(mkt_id)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(c_on.index, 100 * c_on.values, color="tab:blue", lw=1.2,
            label="overnight (close " + r"$\rightarrow$" + " open)")
    ax.plot(c_id.index, 100 * c_id.values, color="tab:green", lw=1.2,
            label="intraday (open " + r"$\rightarrow$" + " close)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overnight_intraday_market.png"), dpi=200)
    plt.close(fig)

    print(summary[["cum_overnight", "cum_intraday"]].describe())
    print("\nEqual-weighted market proxy: overnight "
          f"{100*c_on.iloc[-1]:+,.0f}% vs intraday {100*c_id.iloc[-1]:+,.0f}% "
          f"({c_on.index[0].date()} to {c_on.index[-1].date()})")
    frac = (summary["cum_overnight"] > summary["cum_intraday"]).mean()
    print(f"Fraction of tickers with overnight > intraday: {frac:.1%} "
          f"({len(summary)} tickers)")


if __name__ == "__main__":
    run()
