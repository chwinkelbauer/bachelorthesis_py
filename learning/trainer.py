"""
Training loop: rollout -> MAPPO update -> logging. Writes into a run directory:
    config.json      full experiment configuration
    metrics.csv      one row per iteration (money metrics only on benchmark iterations)
    checkpoint.pt    policy + critic + optimizer + iteration + RNG states (written atomically)

A run can be resumed from its checkpoint (resume=True): weights, optimizer, iteration counter, elapsed time and RNG
states are restored and metrics.csv is continued (rows after the checkpoint are dropped). With RNG states in the
checkpoint the resumed run is bit-identical to an uninterrupted one; checkpoints without them (older runs) are resumed
with a deterministic reseed (seed + iteration).

Learning sessions (cfg.curriculum.sessions): every iteration is trained on the configuration of its session (the base
configuration with the session's overrides); after the last session the base configuration (the full economy) is used.
The world is rebuilt at session boundaries, which does not consume random numbers, so resumes stay bit-identical.
"""

import copy
import csv
import json
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch

from agents.policy import PolicyNet, CriticNet
from analysis.money_metrics import compute_metrics, classify
from configs.base import ExperimentConfig, apply_overrides
from economy.world import EconomyBatch
from learning.ppo import ppo_update
from learning.rollout import collect_rollout
from learning.seeding import set_seed
from learning.evaluate import paired_null_metrics

TRAIN_FIELDS = ("iteration", "seconds", "mean_episode_reward", "trade_rate", "session")
LOSS_FIELDS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")
EXCHANGE_FIELDS = (
    "no_use_share", "resale_rate", "money_hhi", "costly_no_use_share", "costly_resale_rate",
    "worlds_with_indirect_exchange", "double_coincidence_share",
    "speculative_acceptance_share", "efficiency_vs_walras", "efficiency_vs_planner", "ex_post_ir_violation",
    "money_good_spoilage_rank", "money_good_weight_rank",
)
PRODUCTION_FIELDS = (
    "pass_through_rate", "money_hhi", "worlds_with_pass_through", "traded_share_of_supply", "double_coincidence_share",
    "money_good_spoilage_rank", "money_good_weight_rank", "money_good_tier_rank", "mean_flow_utility", "survival_rate",
    "consumed_share_of_supply", "labour_specialization", "idle_share", "harvest_share", "process_share",
)
NULL_KEYS = {"exchange": ("no_use_share", "resale_rate", "money_hhi"),
             "production": ("pass_through_rate", "money_hhi", "mean_flow_utility")}


def log_fields(cfg: ExperimentConfig):
    metrics = PRODUCTION_FIELDS if cfg.market.economy == "production" else EXCHANGE_FIELDS
    nulls = tuple(f"null_{k}" for k in NULL_KEYS[cfg.market.economy])
    return TRAIN_FIELDS + metrics + nulls + ("stage",) + LOSS_FIELDS


def _progress_line(cfg: ExperimentConfig, row: Dict[str, object]) -> str:
    head = f"[{cfg.name} s{cfg.seed}] it {row['iteration']:4d} | reward {row['mean_episode_reward']:.3f} | trade {row['trade_rate']:.2f}"
    if cfg.market.economy == "production":
        body = (f" | pass {row['pass_through_rate']:.3f} (null {row['null_pass_through_rate']:.3f})"
                f" | hhi {row['money_hhi']:.2f} (null {row['null_money_hhi']:.2f})"
                f" | u {row['mean_flow_utility']:.3f} | surv {row['survival_rate']:.2f}")
    else:
        body = (f" | no-use {row['no_use_share']:.3f} | resale {row['resale_rate']:.2f} (null {row['null_resale_rate']:.2f})"
                f" | hhi {row['money_hhi']:.2f} (null {row['null_money_hhi']:.2f}) | eff(W) {row['efficiency_vs_walras']:.2f}")
    session = f" | S{row['session']}" if cfg.curriculum.sessions else ""
    return head + body + f" | {row['stage']}{session} | {row['seconds']}s"


def validate_sessions(cfg: ExperimentConfig) -> None:
    """Session ends must increase and leave at least one iteration on the full economy."""
    previous = 0
    for s in cfg.curriculum.sessions:
        until = int(s["until"])
        if until <= previous:
            raise ValueError(f"Session ends must be strictly increasing, got {until} after {previous}")
        apply_overrides(copy.deepcopy(cfg), s.get("set", []))   # fails early on unknown keys
        previous = until
    if cfg.curriculum.sessions and previous >= cfg.ppo.iterations:
        raise ValueError(f"The last session ends at {previous}, so the full economy is never trained "
                         f"({cfg.ppo.iterations} iterations)")


def session_config(cfg: ExperimentConfig, iteration: int):
    """(session number starting at 1, configuration) for an iteration; the full economy is the last session."""
    for k, s in enumerate(cfg.curriculum.sessions):
        if iteration <= int(s["until"]):
            return k + 1, apply_overrides(copy.deepcopy(cfg), s.get("set", []))
    return len(cfg.curriculum.sessions) + 1, cfg


def build_models(cfg: ExperimentConfig):
    policy, critic = PolicyNet(cfg.model), CriticNet(cfg.model)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=cfg.ppo.lr)
    return policy, critic, optimizer


def save_checkpoint(path: str, cfg: ExperimentConfig, policy, critic, optimizer, iteration: int,
                    elapsed: float = 0.0) -> None:
    data = {"config": cfg.to_dict(), "policy": policy.state_dict(), "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(), "iteration": iteration, "elapsed": elapsed,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)          # atomic: an interrupted save never corrupts the previous checkpoint


def _restore_rng(data: dict, cfg: ExperimentConfig) -> bool:
    rng = data.get("rng")
    if rng is None:
        set_seed(cfg.seed + int(data["iteration"]))
        return False
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    return True


def _truncate_log(path: str, last_iteration: int, fields) -> None:
    """Keeps rows up to the checkpoint and rewrites the file with the current columns (new columns stay empty)."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in rows if int(r["iteration"]) <= last_iteration)


def load_checkpoint(path: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(data["config"])
    policy, critic, optimizer = build_models(cfg)
    policy.load_state_dict(data["policy"])
    critic.load_state_dict(data["critic"])
    optimizer.load_state_dict(data["optimizer"])
    return cfg, policy, critic, optimizer, data["iteration"]


def train(cfg: ExperimentConfig, run_dir: str, init_checkpoint: Optional[str] = None, verbose: bool = True,
          resume: bool = False) -> str:
    os.makedirs(run_dir, exist_ok=True)
    set_seed(cfg.seed)
    policy, critic, optimizer = build_models(cfg)
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    log_path = os.path.join(run_dir, "metrics.csv")
    first, elapsed0 = 1, 0.0

    if resume:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Nothing to resume: {ckpt} does not exist")
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(data["policy"])
        critic.load_state_dict(data["critic"])
        optimizer.load_state_dict(data["optimizer"])
        first, elapsed0 = int(data["iteration"]) + 1, float(data.get("elapsed", 0.0))
        exact = _restore_rng(data, cfg)
        _truncate_log(log_path, first - 1, log_fields(cfg))
        if verbose:
            print(f"[{cfg.name} s{cfg.seed}] resuming at iteration {first} "
                  f"({'exact RNG state' if exact else 'reseeded: checkpoint has no RNG state'})", flush=True)
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        if init_checkpoint:  # warm start (curriculum); optimizer state is not carried over
            _, p_init, c_init, _, _ = load_checkpoint(init_checkpoint)
            policy.load_state_dict(p_init.state_dict())
            critic.load_state_dict(c_init.state_dict())

    validate_sessions(cfg)
    world, session = None, None
    with open(log_path, "a" if resume else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields(cfg), extrasaction="ignore")
        if not resume:
            writer.writeheader()
        start = time.time() - elapsed0
        for it in range(first, cfg.ppo.iterations + 1):
            k, cfg_it = session_config(cfg, it)
            if k != session:   # new learning session: rebuild the world with the session's configuration
                world, session = EconomyBatch(cfg_it, cfg_it.ppo.worlds), k
                if verbose and cfg.curriculum.sessions:
                    print(f"[{cfg.name} s{cfg.seed}] session {k} from iteration {it}: "
                          + (", ".join(cfg.curriculum.sessions[k - 1].get("set", [])) if k <= len(cfg.curriculum.sessions)
                             else "full economy"), flush=True)
            rng_state = torch.get_rng_state()
            rollout = collect_rollout(world, policy)
            row: Dict[str, object] = {"iteration": it, "session": k}
            row["mean_episode_reward"] = float(rollout.rewards.sum(0).mean())
            row["trade_rate"] = len(world.record.trades()["b"]) / max(world.record.meetings, 1)
            benchmark = it % cfg.ppo.benchmark_every == 0 or it == 1 or it == cfg.ppo.iterations
            if benchmark:
                row.update(compute_metrics(world.record, cfg.metrics))
            row.update(ppo_update(rollout, world, policy, critic, optimizer, cfg_it.ppo))
            if benchmark:  # after the update: the null run overwrites the world state
                null = paired_null_metrics(world, rng_state, cfg_it, policy)["stronger"]
                row.update({f"null_{k}": null[k] for k in NULL_KEYS[cfg.market.economy]})
                row["stage"] = classify(row, null, cfg.metrics)
            row["seconds"] = round(time.time() - start, 1)
            writer.writerow(row)
            f.flush()
            if verbose and ("stage" in row):
                print(_progress_line(cfg, row), flush=True)
            if it % cfg.ppo.checkpoint_every == 0:
                save_checkpoint(ckpt, cfg, policy, critic, optimizer, it, time.time() - start)
    save_checkpoint(ckpt, cfg, policy, critic, optimizer, cfg.ppo.iterations, time.time() - start)
    return ckpt
