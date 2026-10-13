"""
Cross-run summary: one table over all runs in results/ (held-out evaluation, learned stochastic policy) and IQM with
95% bootstrap CIs across seeds per condition.

    python -m scripts.summarize                      # markdown to stdout
    python -m scripts.summarize --out results/summary.md

A condition is the run-directory name without its seed suffix (e.g. "A", "B_fromA"). Archived pilots (pilot_*) are
listed but excluded from the aggregates. Runs without evaluation.json are listed as partial.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from analysis.stats import iqm, bootstrap_ci

EXCHANGE = [("efficiency_vs_walras", "eff. vs Walras"), ("resale_rate", "resale"), ("null_resale_rate", "null resale"),
            ("costly_resale_rate", "costly resale"), ("money_hhi", "HHI"), ("null_money_hhi", "null HHI")]
PRODUCTION = [("mean_flow_utility", "flow utility"), ("survival_rate", "survival"), ("trade_rate", "trade rate"),
              ("pass_through_rate", "pass-through"), ("null_pass_through_rate", "null pass"), ("money_hhi", "HHI"),
              ("null_money_hhi", "null HHI")]


def load_runs(results: str) -> List[Dict]:
    runs = []
    for name in sorted(os.listdir(results)):
        path = os.path.join(results, name)
        if not os.path.isfile(os.path.join(path, "config.json")):
            continue
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        economy = cfg.get("market", {}).get("economy", "exchange")
        iterations = 0
        if os.path.exists(os.path.join(path, "metrics.csv")):
            log = pd.read_csv(os.path.join(path, "metrics.csv"))
            iterations = int(log["iteration"].max()) if len(log) else 0
        evaluation = None
        if os.path.exists(os.path.join(path, "evaluation.json")):
            with open(os.path.join(path, "evaluation.json"), encoding="utf-8") as f:
                evaluation = json.load(f)
        match = re.match(r"(.+)_s(\d+)$", name)
        runs.append({
            "run": name, "condition": match.group(1) if match else name, "seed": int(match.group(2)) if match else None,
            "economy": economy, "iterations": iterations, "planned": cfg.get("ppo", {}).get("iterations"),
            "pilot": name.startswith("pilot_"), "evaluation": evaluation,
        })
    return runs


def _fmt(v) -> str:
    return "—" if v is None or (isinstance(v, float) and v != v) else (f"{v:.3f}" if isinstance(v, float) else str(v))


def run_table(runs: List[Dict], economy: str) -> str:
    cols = EXCHANGE if economy == "exchange" else PRODUCTION
    extra = ["no-trade flow utility"] if economy == "production" else []
    lines = ["| run | iterations | " + " | ".join(c[1] for c in cols) + (" | " + " | ".join(extra) if extra else "")
             + " | stage (batches) |",
             "|" + "---|" * (len(cols) + len(extra) + 3)]
    for r in runs:
        if r["economy"] != economy:
            continue
        ev = r["evaluation"]
        if ev is None:
            lines.append(f"| {r['run']} | {r['iterations']}/{r['planned']} | " + " | ".join("—" for _ in cols + extra)
                         + " | partial (not evaluated) |")
            continue
        m = ev["learned_stochastic"]
        vals = [_fmt(m.get(k)) for k, _ in cols]
        if economy == "production":
            vals.append(_fmt(ev.get("no_trade_counterfactual", {}).get("mean_flow_utility")))
        counts = m.get("stage_counts", {})
        stage = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) if counts else m.get("stage")
        lines.append(f"| {r['run']} | {r['iterations']}/{r['planned']} | " + " | ".join(vals) + f" | {stage} |")
    return "\n".join(lines)


def aggregate_table(runs: List[Dict]) -> str:
    groups = defaultdict(list)
    for r in runs:
        if r["evaluation"] is not None and not r["pilot"]:
            groups[(r["condition"], r["economy"])].append(r)
    lines = ["| condition | seeds | metric | IQM | 95% CI | per seed |", "|---|---|---|---|---|---|"]
    for (condition, economy), members in sorted(groups.items()):
        cols = EXCHANGE if economy == "exchange" else PRODUCTION
        seeds = ", ".join(str(m["seed"]) for m in members)
        for key, label in cols:
            values = [m["evaluation"]["learned_stochastic"].get(key) for m in members]
            values = [v for v in values if isinstance(v, (int, float))]
            if not values:
                continue
            lo, hi = bootstrap_ci(values) if len(values) > 1 else (values[0], values[0])
            lines.append(f"| {condition} | {seeds} | {label} | {iqm(values):.3f} | [{lo:.3f}, {hi:.3f}] | "
                         + " / ".join(f"{v:.3f}" for v in values) + " |")
        if economy == "exchange":
            excess = [m["evaluation"]["learned_stochastic"]["money_hhi"] - m["evaluation"]["learned_stochastic"]["null_money_hhi"]
                      for m in members]
            lines.append(f"| {condition} | {seeds} | HHI excess over null (money ≥ 0.15) | {iqm(excess):.3f} | — | "
                         + " / ".join(f"{v:+.3f}" for v in excess) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    runs = load_runs(args.results)
    text = "\n\n".join([
        "## Exchange economy (stage A): runs", run_table(runs, "exchange"),
        "## Production economy (stages B–D): runs", run_table(runs, "production"),
        "## Aggregates across seeds (pilots excluded; IQM, percentile bootstrap)", aggregate_table(runs),
    ])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
