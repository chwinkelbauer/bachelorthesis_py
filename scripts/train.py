"""
Train one experimental condition.

    python -m scripts.train --preset A --seed 0
    python -m scripts.train --preset A --seed 0 --iterations 50 --run-dir results/A_s0_short
    python -m scripts.train --resume --run-dir results/B_s0          # continue an interrupted run
"""

import argparse
import json
import os

from configs.base import apply_overrides, ExperimentConfig
from configs.presets import get_preset, PRESETS
from learning.trainer import train, load_checkpoint
from learning.evaluate import evaluate_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="A", choices=sorted(PRESETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--worlds", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--init", default=None, help="warm-start checkpoint (curriculum)")
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. preferences.goods_valued=2")
    parser.add_argument("--resume", action="store_true", help="continue the run in --run-dir from its checkpoint")
    args = parser.parse_args()

    if args.resume:
        if not args.run_dir:
            parser.error("--resume needs --run-dir")
        with open(os.path.join(args.run_dir, "config.json")) as f:
            cfg = ExperimentConfig.from_dict(json.load(f))   # the run's own configuration
    else:
        cfg = apply_overrides(get_preset(args.preset), args.set)
        cfg.seed = args.seed
    if args.iterations is not None:
        cfg.ppo.iterations = args.iterations
    if args.worlds is not None:
        cfg.ppo.worlds = args.worlds
    run_dir = args.run_dir or os.path.join("results", f"{cfg.name}_s{cfg.seed}")

    ckpt = train(cfg, run_dir, init_checkpoint=args.init, resume=args.resume)
    cfg, policy, _, _, _ = load_checkpoint(ckpt)
    report = evaluate_all(cfg, policy, args.eval_batches)
    with open(os.path.join(run_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
