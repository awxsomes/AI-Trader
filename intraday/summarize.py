"""Collect every intraday run next to the daily baseline in one table.

    python intraday\\summarize.py
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output", "intraday")

rows = []
for p in sorted(glob.glob(os.path.join(OUT, "*_summary.json"))):
    s = json.load(open(p))
    rows.append({"config": f"{s['bar']}-min, overnight={s['overnight']}",
                 "model": s.get("model", "CNN+Trans (cost-trained)"),
                 "days OOS": s["days"],
                 "gross SR": s["gross_sharpe"], "net SR": s["net_sharpe"],
                 "net SR (daily agg)": s["net_sharpe_daily_agg"],
                 "net return/yr": s["net_mean"], "turnover/day": s["turnover_per_day"],
                 "max DD": s["max_drawdown"]})
daily = os.path.join(ROOT, "output", "full", "execution_study.csv")
if os.path.exists(daily):
    ex = pd.read_csv(daily)
    for lam in (1.0, 0.2):
        r = ex[(ex.model == "cnn_pca5_cost") & (ex["lambda"] == lam)].iloc[0]
        rows.append({"config": f"DAILY (2002-2026)", "model": f"CNN+Trans lam={lam}", "days OOS": None,
                     "gross SR": r["gross SR"], "net SR": r["net SR"],
                     "net SR (daily agg)": r["net SR"], "net return/yr": r["net mu"],
                     "turnover/day": r["turnover/day"], "max DD": None})
df = pd.DataFrame(rows).sort_values(["config", "net SR (daily agg)"], ascending=[True, False])
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
df.to_csv(os.path.join(OUT, "grid_summary.csv"), index=False)
print(f"\nsaved {os.path.join(OUT, 'grid_summary.csv')}")
print("\nRead the 'net SR (daily agg)' column against the daily baseline; a config "
      "has to beat ~0.75 clearly, on a sample this short, to be worth going live.")
