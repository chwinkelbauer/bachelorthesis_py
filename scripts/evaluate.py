"""
Evaluate a trained run against the ZI-C baseline on held-out economies and detect change points in its
training curve.

    python -m scripts.evaluate --run-dir results/A_s0
    python -m scripts.evaluate --run-dir results/D_s0 --set ledger.switch_round=12 --out evaluation_ledger_off.json
"""

import argparse
import json
import os

import pandas as pd

from analysis.changepoints import binary_segmentation
from learning.evaluate import evaluate_all
from configs.base import apply_overrides
from learning.trainer import load_checkpoint

CURVES = {
    "exchange": ("no_use_share", "resale_rate", "money_hhi", "trade_rate", "efficiency_vs_walras"),
    "production": ("pass_through_rate", "money_hhi", "trade_rate", "mean_flow_utility", "survival_rate"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides for evaluation, e.g. ledger.switch_round=12")
    parser.add_argument("--out", default="evaluation.json")
    args = parser.parse_args()

    cfg, policy, _, _, _ = load_checkpoint(os.path.join(args.run_dir, "checkpoint.pt"))
    cfg = apply_overrides(cfg, args.set)
    report = evaluate_all(cfg, policy, args.batches)

    log = pd.read_csv(os.path.join(args.run_dir, "metrics.csv"))
    bench = log.dropna(subset=["stage"])
    report["change_points"] = {
        c: [int(bench["iteration"].iloc[t]) for t in binary_segmentation(bench[c].to_numpy())]
        for c in CURVES[cfg.market.economy] if c in bench
    }
    report["overrides"] = args.set
    with open(os.path.join(args.run_dir, args.out), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
