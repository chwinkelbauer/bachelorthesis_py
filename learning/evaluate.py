"""
Evaluation on held-out episodes (fixed evaluation seeds, disjoint from training seeds). Every evaluated policy plays
the identical economies. Each batch is classified against its paired null run:
    exchange economy   : RandomPolicy (random trading)
    production economy : RandomTradeNull(policy) (the policy's own labour/consumption, random trading)
Baselines: exchange -> ZI-C and random; production -> no-trade counterfactual and random-trade null.
"""

import warnings
from typing import Dict

import numpy as np
import torch

from agents.baselines import ZICPolicy, RandomPolicy, RandomTradeNull, NoTradePolicy
from analysis.money_metrics import compute_metrics, classify
from configs.base import ExperimentConfig
from economy.world import EconomyBatch
from learning.rollout import collect_rollout
from learning.seeding import set_seed

EVAL_SEED_OFFSET = 1_000_000


def null_policy(cfg: ExperimentConfig, policy):
    return RandomTradeNull(policy) if cfg.market.economy == "production" else RandomPolicy()


def paired_null_metrics(world: EconomyBatch, rng_state: torch.Tensor, cfg: ExperimentConfig, policy,
                        greedy: bool = False) -> Dict[str, float]:
    """
    Re-creates the economies generated from rng_state (the RNG state right before a rollout), runs the null policy on
    them and returns its metrics. The caller's RNG state is restored afterwards.
    """
    current = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    collect_rollout(world, null_policy(cfg, policy), greedy=greedy)
    null = compute_metrics(world.record, cfg.metrics)
    torch.set_rng_state(current)
    return null


def evaluate_policy(cfg: ExperimentConfig, policy, batches: int = 4, greedy: bool = False, null_base=None) -> Dict[str, float]:
    """Mean metrics over held-out batches; each batch's stage is classified against its paired random null."""
    world = EconomyBatch(cfg, cfg.ppo.worlds)
    if hasattr(policy, "eval"):
        policy.eval()
    rows = []
    for k in range(batches):
        set_seed(EVAL_SEED_OFFSET + k)   # identical economies for every evaluated policy
        state = torch.get_rng_state()
        collect_rollout(world, policy, greedy=greedy)
        m = compute_metrics(world.record, cfg.metrics)
        null = paired_null_metrics(world, state, cfg, null_base if null_base is not None else policy, greedy)
        m["stage"] = classify(m, null, cfg.metrics)
        m.update({f"null_{key}": v for key, v in null.items()})
        rows.append(m)
    with warnings.catch_warnings():  # all-NaN columns (e.g. no money good without trade) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        numeric = {key: float(np.nanmean([r[key] for r in rows])) for key in rows[0] if key != "stage"}
    stages = [r["stage"] for r in rows]
    numeric["stage"] = max(set(stages), key=stages.count)
    numeric["stage_counts"] = {s: stages.count(s) for s in set(stages)}
    return numeric


def evaluate_all(cfg: ExperimentConfig, policy, batches: int = 4) -> Dict[str, Dict[str, float]]:
    report = {
        "learned_stochastic": evaluate_policy(cfg, policy, batches, greedy=False),
        "learned_greedy": evaluate_policy(cfg, policy, batches, greedy=True),
    }
    if cfg.market.economy == "production":
        report["no_trade_counterfactual"] = evaluate_policy(cfg, NoTradePolicy(policy), batches, null_base=policy)
        report["random_trade_null"] = evaluate_policy(cfg, RandomTradeNull(policy), batches, null_base=policy)
    else:
        report["zic_baseline"] = evaluate_policy(cfg, ZICPolicy(), batches)
        report["random_baseline"] = evaluate_policy(cfg, RandomPolicy(), batches)
    return report
