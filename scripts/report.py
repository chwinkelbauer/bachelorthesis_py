"""
Results report for one run: training curves (learned vs. paired null), stage timeline and change points.

    python -m scripts.report --run-dir results/A_s0          # -> results/A_s0/report.png + printed summary

Reads metrics.csv (benchmark iterations only) and, if present, evaluation.json. Works on interrupted runs.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

from analysis.changepoints import binary_segmentation

# validated palette (light surface): learned vs. null, ordinal stage ramp, chart chrome
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LEARNED, NULL = "#2a78d6", "#eb6834"
STAGES = ("autarky", "barter", "indirect_exchange", "money")
STAGE_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

PANELS = {
    "exchange": [
        ("resale_rate", "Completed indirect exchange (resale rate)", True),
        ("money_hhi", "Concentration on one medium (HHI)", True),
        ("no_use_share", "No-use acceptance share", True),
        ("costly_resale_rate", "Costly indirect exchange (descriptive)", False),
        ("efficiency_vs_walras", "Welfare gain / Walras gain", False),
        ("trade_rate", "Trades per meeting", False),
    ],
    "production": [
        ("pass_through_rate", "Completed pass-through rate", True),
        ("money_hhi", "Concentration on one medium (HHI)", True),
        ("mean_flow_utility", "Flow utility per living agent-round", True),
        ("survival_rate", "Survival over the rollout", False),
        ("trade_rate", "Trades per meeting", False),
        ("labour_specialization", "Labour specialisation (HHI)", False),
    ],
}


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def summarize(bench: pd.DataFrame, economy: str) -> dict:
    """First iteration of each stage, stage counts, change points and first/last values."""
    out = {"benchmark_iterations": int(len(bench)), "last_iteration": int(bench["iteration"].iloc[-1])}
    out["stage_counts"] = bench["stage"].value_counts().to_dict()
    out["first_iteration_of_stage"] = {s: int(bench.loc[bench["stage"] == s, "iteration"].iloc[0])
                                       for s in STAGES if (bench["stage"] == s).any()}
    out["change_points"], out["first_last"] = {}, {}
    for key, _, _ in PANELS[economy]:
        if key in bench and bench[key].notna().sum() > 1:
            y = bench[key].to_numpy()
            out["change_points"][key] = [int(bench["iteration"].iloc[t]) for t in binary_segmentation(y)]
            out["first_last"][key] = (round(float(y[0]), 3), round(float(y[-1]), 3))
    return out


def plot(bench: pd.DataFrame, economy: str, title: str, path: str, change_points: dict) -> None:
    panels = [p for p in PANELS[economy] if p[0] in bench and bench[p[0]].notna().any()]
    ncols = 2
    nrows = (len(panels) + 1) // ncols
    fig = plt.figure(figsize=(11, 2.6 * nrows + 1.2), facecolor=SURFACE)
    grid = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.22], hspace=0.55, wspace=0.18)
    x = bench["iteration"]
    step = float(x.iloc[-1] - x.iloc[-2]) if len(x) > 1 else 1.0
    xlim = (0.0, float(x.iloc[-1]) + step)          # shared by every panel and the stage strip
    for idx, (key, label, has_null) in enumerate(panels):
        ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
        _style(ax, label)
        ax.set_xlim(*xlim)
        ax.plot(x, bench[key], color=LEARNED, linewidth=2, marker="o", markersize=3.5)
        null_key = f"null_{key}"
        if has_null and null_key in bench:
            ax.plot(x, bench[null_key], color=NULL, linewidth=2, linestyle=(0, (4, 3)))
        for t in change_points.get(key, []):
            ax.axvline(t, color=INK_2, linewidth=1, linestyle=":")
    # stage timeline (ordinal ramp), one cell per benchmark iteration
    ax = fig.add_subplot(grid[nrows, :])
    _style(ax, "Stage (classified against the paired null)")
    ax.grid(False)
    edges = [xlim[0]] + list(x.iloc[1:]) + [xlim[1]]   # each benchmark classifies the span until the next one
    for i, stage in enumerate(bench["stage"]):
        color = STAGE_COLORS[STAGES.index(stage)] if stage in STAGES else GRID
        ax.axvspan(edges[i], edges[i + 1], color=color, linewidth=0)
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_xlabel("training iteration", color=INK_2, fontsize=9)

    handles = [Line2D([], [], color=LEARNED, linewidth=2, marker="o", markersize=3.5, label="learned policy"),
               Line2D([], [], color=NULL, linewidth=2, linestyle=(0, (4, 3)), label="paired null (stronger of random, ZI-C)"),
               Line2D([], [], color=INK_2, linewidth=1, linestyle=":", label="change point")]
    handles += [Patch(color=c, label=s.replace("_", " ")) for s, c in zip(STAGES, STAGE_COLORS)]
    fig.legend(handles=handles, loc="lower left", ncol=7, frameon=False, fontsize=8.5, labelcolor=INK_2,
               bbox_to_anchor=(0.11, 0.955))
    fig.suptitle(title, x=0.12, y=1.0, ha="left", fontsize=12, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    economy = cfg["market"].get("economy", "exchange")
    log = pd.read_csv(os.path.join(args.run_dir, "metrics.csv"))
    bench = log.dropna(subset=["stage"]).reset_index(drop=True)
    summary = summarize(bench, economy)
    title = f"Run {cfg['name']} (seed {cfg['seed']}, {economy} economy), up to iteration {summary['last_iteration']}"
    path = os.path.join(args.run_dir, "report.png")
    plot(bench, economy, title, path, summary["change_points"])

    evaluation = os.path.join(args.run_dir, "evaluation.json")
    if os.path.exists(evaluation):
        with open(evaluation) as f:
            ev = json.load(f)
        keys = [k for k, _, _ in PANELS[economy]]
        summary["evaluation"] = {pol: {k: round(v[k], 3) for k in keys + ["stage"] if k in v and not isinstance(v[k], str)}
                                 | {"stage": v.get("stage")} for pol, v in ev.items() if isinstance(v, dict) and "stage" in v}
    print(json.dumps(summary, indent=2))
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
