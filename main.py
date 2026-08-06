"""
run.py — single entry point for the autotrader.

    python run.py predictor                      train the return predictor only
    python run.py dqn                            train the DQN on synthetic prices
    python run.py pipeline                       predictor -> DQN (real data by default)

    python run.py pipeline --cash 500 --seeds 5  five seeds, summary table at the end
    python run.py dqn --seeds 5 --parallel       run those five at once

    --data real | synth | auto   force the data source (default: auto)
    --episodes N                 DQN training episodes (default 500)
    --quick                      tiny run to check the plumbing

SIMULATION ONLY. Nothing here connects to a broker or moves real money.
"""

import argparse
import json
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor


def _run_one(mode, seed, cash, episodes, data, quick):
    """One complete run at one seed. Module-level so it can be subprocessed."""
    import numpy as np
    import torch
    import random

    import nn_stock_predictor as N
    import dqn_trader as D

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs("models", exist_ok=True)

    ncfg, dcfg = N.Config(), D.DQNConfig()
    ncfg.SOURCE = {"real": "real", "synth": "synth", "auto": "auto"}[data]
    dcfg.INITIAL_CASH = cash
    dcfg.SEED = seed
    dcfg.EPISODES = episodes
    dcfg.MODEL_SAVE_PATH = f"models/dqn_trader_seed{seed}.pt"
    ncfg.MODEL_SAVE_PATH = f"models/nn_predictor_seed{seed}.pt"

    if quick:
        ncfg.EPOCHS, ncfg.PATIENCE = 5, 3
        ncfg.HIDDEN_SIZE, ncfg.NUM_LAYERS = 32, 2
        dcfg.EPISODES = min(dcfg.EPISODES, 10)
        dcfg.MAX_STEPS_PER_EPISODE = 300
        dcfg.EPSILON_DECAY = 0.85

    print(f"\n{'='*60}\nSEED {seed} | mode={mode} | cash=${cash:,.2f}\n{'='*60}")

    if mode == "predictor":
        # Run the steps directly rather than calling N.main(), which builds its
        # own Config and would ignore --quick / --data.
        from torch.utils.data import DataLoader

        df, source = N.load_market_data(ncfg)
        df = N.add_technical_indicators(df)
        (Xtr, ytr, Xv, yv, Xte, yte, sc, tsc,
         names, mask_test, base_test) = N.prepare_data(
            df, ncfg.LOOKBACK_WINDOW, ncfg.PREDICTION_HORIZON, ncfg.TRAIN_SPLIT,
            target=ncfg.TARGET, vol_horizon=ncfg.VOL_HORIZON)

        tr = DataLoader(N.StockDataset(Xtr, ytr), batch_size=ncfg.BATCH_SIZE,
                        shuffle=True, drop_last=True)
        va = DataLoader(N.StockDataset(Xv, yv), batch_size=ncfg.BATCH_SIZE)
        te = DataLoader(N.StockDataset(Xte, yte), batch_size=ncfg.BATCH_SIZE)

        model = N.StockPredictor(len(names), ncfg.HIDDEN_SIZE, ncfg.NUM_LAYERS,
                                 ncfg.DROPOUT, ncfg.BIDIRECTIONAL)
        model, _ = N.train_model(model, tr, va, ncfg)
        results = N.evaluate_model(model, te, tsc, ncfg,
                                   trade_mask=mask_test, baseline_returns=base_test,
                                   target=ncfg.TARGET)
        N.save_inference_bundle(
            ncfg.MODEL_SAVE_PATH.replace('.pt', '_bundle.pkl'),
            ncfg.MODEL_SAVE_PATH, sc, tsc, names, ncfg, source)
        if ncfg.TARGET == "volatility":
            return {"seed": seed, "mode": mode, "target": "volatility",
                    "data_source": source,
                    "r2_vs_persistence": results["r2_vs_persistence"],
                    "r2_vs_mean": results["r2_vs_mean"],
                    "corr": results["corr"],
                    "qlike": results["qlike"],
                    "qlike_persistence": results["qlike_persistence"],
                    "beats_persistence": results["beats_persistence"],
                    "spread_ratio": results["spread_ratio"],
                    "near_constant": results["near_constant"],
                    "n_eval": results["n_eval"]}
        return {"seed": seed, "mode": mode, "target": "direction",
                "data_source": source,
                "directional_acc": results["directional_acc"],
                "directional_acc_se": results["directional_acc_se"],
                "baseline_acc": results["baseline_acc"],
                "majority_acc": results["majority_acc"],
                "edge_over_majority": results["edge_over_majority"],
                "pred_up_rate": results["pred_up_rate"],
                "collapsed": results["collapsed"],
                "n_eval": results["n_eval"]}

    price_data = predictions = regime_data = None
    source = "synthetic"
    if mode == "pipeline":
        price_data, predictions, regime_data, source = N.train_and_forecast(
            config=ncfg, seed=seed)

    agent, env, rewards = D.train_dqn(dcfg, price_data, predictions, regime_data)
    summary = env.get_trade_summary()
    return {"seed": seed, "mode": mode, "data_source": source,
            "initial_cash": cash, **summary}


def _worker(args):
    return _run_one(*args)


def summarize(results):
    """Print the across-seed table. The spread matters more than any one run."""
    if not results:
        return
    if results[0].get("target") == "volatility":
        print(f"\n{'seed':>5} {'R2 vs persist':>14} {'R2 vs mean':>11} "
              f"{'corr':>7} {'QLIKE':>8} {'spread':>7} {'':>10}")
        for r in sorted(results, key=lambda x: x["seed"]):
            flag = ("BEATS" if r["beats_persistence"]
                    else ("flat-fcst" if r.get("near_constant") else ""))
            print(f"{r['seed']:>5} {r['r2_vs_persistence']:>14.4f} "
                  f"{r['r2_vs_mean']:>11.4f} {r['corr']:>7.4f} "
                  f"{r['qlike']:>8.4f} {r.get('spread_ratio',0):>7.3f} {flag:>10}")
        n_beat = sum(1 for r in results if r["beats_persistence"])
        print(f"\n{n_beat}/{len(results)} seeds show real skill (positive R2 vs "
              f"BOTH mean and persistence, better QLIKE, varying forecast).")
        vals = [r["r2_vs_persistence"] for r in results]
        if len(vals) > 1:
            print(f"across {len(vals)} seeds: mean R2 vs persistence "
                  f"{st.mean(vals):.4f}  sd {st.stdev(vals):.4f}  "
                  f"range {min(vals):.4f} .. {max(vals):.4f}")
        with open("models/sweep_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print("\nSaved: models/sweep_summary.json")
        return

    if results[0]["mode"] == "predictor":
        print(f"\n{'seed':>5} {'dir acc%':>9} {'SE':>6} {'persist%':>9} "
              f"{'major%':>8} {'up%':>7} {'':>10}")
        for r in sorted(results, key=lambda x: x["seed"]):
            flag = "COLLAPSED" if r.get("collapsed") else ""
            print(f"{r['seed']:>5} {r['directional_acc']:>9.2f} "
                  f"{r['directional_acc_se']:>6.2f} {r['baseline_acc']:>9.2f} "
                  f"{r.get('majority_acc',0):>8.2f} {r.get('pred_up_rate',0):>7.1f} "
                  f"{flag:>10}")
        n_collapsed = sum(1 for r in results if r.get("collapsed"))
        if n_collapsed:
            print(f"\n{n_collapsed}/{len(results)} seeds collapsed to a constant "
                  f"sign — exclude them before averaging.")
        vals = [r["directional_acc"] for r in results]
    else:
        print(f"\n{'seed':>5} {'source':>10} {'trades':>7} {'return%':>9} "
              f"{'B&H%':>8} {'maxDD%':>8} {'final$':>10}")
        for r in sorted(results, key=lambda x: x["seed"]):
            print(f"{r['seed']:>5} {r.get('data_source','-'):>10} "
                  f"{r.get('total_trades',0):>7} {r.get('total_return',0.0):>9.2f} "
                  f"{r.get('buy_and_hold_return',0.0):>8.2f} "
                  f"{r.get('max_drawdown',0.0):>8.2f} "
                  f"{r['initial_cash']*(1+r.get('total_return',0.0)/100):>10.2f}")
        vals = [r.get("total_return", 0.0) for r in results]

    if len(vals) > 1:
        print(f"\nacross {len(vals)} seeds: mean {st.mean(vals):.2f}  "
              f"sd {st.stdev(vals):.2f}  range {min(vals):.2f} .. {max(vals):.2f}")
        print("Report the spread, not your best seed — RL results swing widely "
              "by initialization, and a single good run is not evidence.")

    with open("models/sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved: models/sweep_summary.json")


def main():
    p = argparse.ArgumentParser(description="Autotrader runner (simulation only)")
    p.add_argument("mode", choices=["predictor", "dqn", "pipeline"])
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of seeds to run (1..N)")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--data", choices=["auto", "real", "synth"], default="auto")
    p.add_argument("--parallel", action="store_true",
                   help="run seeds concurrently (uses one CPU core each)")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    os.makedirs("models", exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    jobs = [(args.mode, s, args.cash, args.episodes, args.data, args.quick)
            for s in seeds]

    if args.parallel and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]

    summarize(results)


if __name__ == "__main__":
    main()