"""
Evaluation on held-out episodes (fixed evaluation seeds, disjoint from training seeds). Every evaluated policy plays
the identical economies. Each batch is classified against the *stronger* of two paired null runs (element-wise
maximum of their metrics):
    exchange economy   : RandomPolicy and ZICPolicy
    production economy : RandomTradeNull(policy) and ZICTradeNull(policy) (the policy's own labour/consumption)
Baselines: exchange -> ZI-C and random; production -> no-trade counterfactual and random-trade null.
"""

import warnings
from typing import Dict

import numpy as np
import torch

from agents.baselines import ZICPolicy, RandomPolicy, RandomTradeNull, ZICTradeNull, NoTradePolicy
from analysis.money_metrics import compute_metrics, classify
from configs.base import ExperimentConfig
from economy.world import EconomyBatch
from learning.rollout import collect_rollout
from learning.seeding import set_seed

EVAL_SEED_OFFSET = 1_000_000


def null_policies(cfg: ExperimentConfig, policy):
    if cfg.market.economy == "production":
        return {"random": RandomTradeNull(policy), "zic": ZICTradeNull(policy)}
    return {"random": RandomPolicy(), "zic": ZICPolicy()}


def stronger_null(nulls: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Element-wise maximum over the null runs (numeric metrics only)."""
    runs = list(nulls.values())
    return {k: max(r[k] for r in runs) for k, v in runs[0].items()
            if isinstance(v, (int, float)) and all(isinstance(r.get(k), (int, float)) for r in runs)}


def paired_null_metrics(world: EconomyBatch, rng_state: torch.Tensor, cfg: ExperimentConfig, policy,
                        greedy: bool = False) -> Dict[str, float]:
    """
    Re-creates the economies generated from rng_state (the RNG state right before a rollout), runs both null policies
    on them and returns {"random": ..., "zic": ..., "stronger": element-wise max}. The caller's RNG state is restored.
    """
    current = torch.get_rng_state()
    nulls = {}
    for name, null in null_policies(cfg, policy).items():
        torch.set_rng_state(rng_state)
        collect_rollout(world, null, greedy=greedy)
        nulls[name] = compute_metrics(world.record, cfg.metrics)
    torch.set_rng_state(current)
    nulls["stronger"] = stronger_null({k: nulls[k] for k in ("random", "zic")})
    return nulls


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
        nulls = paired_null_metrics(world, state, cfg, null_base if null_base is not None else policy, greedy)
        m["stage"] = classify(m, nulls["stronger"], cfg.metrics)
        for name in ("random", "zic", "stronger"):
            prefix = "null_" if name == "stronger" else f"null_{name}_"
            m.update({f"{prefix}{key}": v for key, v in nulls[name].items() if isinstance(v, (int, float))})
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
