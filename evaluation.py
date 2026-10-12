"""
Evaluation module: runs a fixed policy (trained brain, rule-based expert or random baseline) on fresh maps
without any learning and reports survival statistics with confidence intervals.
"""

from typing import Optional, Dict, Any

import numpy as np
import torch

from config import EnvConfig, DEFAULT_CONFIG
from environment import calculate_utility
from agent import execute_move, execute_gather, execute_consumption, get_action_mask
from dataset import get_expert_decisions
from network import MultiHeadGlobalBrain
from metrics import EpisodeStats
from training import run_simulation_step, _new_episode


def _scripted_step(agents, env_grid, policy: str, config: EnvConfig) -> None:
    """Executes one simulation step for all living agents with the expert or random policy."""
    n_rows, n_cols, _ = env_grid.shape
    for agent in agents:
        if not agent.alive:
            continue
        mask = get_action_mask(agent, n_rows, n_cols, agents, config.mask_occupied_tiles)
        if policy == "expert":
            d = get_expert_decisions(agent, env_grid, config)
            action_idx = d["action_idx"] if mask[d["action_idx"] - 1] else 11
            consume_type, consume_fraction = d["consume_type"], d["consume_fraction"]
        else:  # random: uniform over valid actions, consume types and fractions
            action_idx = int(np.random.choice(np.flatnonzero(mask))) + 1
            consume_type = int(np.random.randint(1, 4))
            consume_fraction = float(np.random.uniform(0.0, 1.0))
        execute_move(agent, action_idx, n_rows, n_cols)
        execute_gather(agent, action_idx, env_grid, 0.0, config)
        execute_consumption(agent, consume_type, consume_fraction, config)


def run_evaluation(
    policy: str = "brain",
    brain: Optional[MultiHeadGlobalBrain] = None,
    config: EnvConfig = DEFAULT_CONFIG,
    episodes: int = 20,
    episode_len: Optional[int] = None,
    deterministic: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a policy on `episodes` fresh maps at full drain for `episode_len` steps each (default: mappo.epoch_max).
    policy: 'brain' (requires brain), 'expert' (rule-based) or 'random'. No learning takes place.
    Returns the aggregated statistics; prints one line per episode and a summary.
    """
    if policy == "brain" and brain is None:
        raise ValueError("policy='brain' requires a loaded brain (--load-brain)")
    episode_len = episode_len or config.mappo.epoch_max
    mode = " (deterministic)" if policy == "brain" and deterministic else ""
    print(f"-> Evaluating policy '{policy}'{mode} on {episodes} fresh maps x {episode_len} steps "
          f"({config.grid.n_agents} agents, {config.grid.n_grid}x{config.grid.k_grid} grid, no learning)...")

    stats = EpisodeStats(config.dead_time)
    final_utility, final_n, final_h, final_gold = [], [], [], []

    for ep in range(1, episodes + 1):
        env_grid, agents = _new_episode(config)
        for _ in range(episode_len):
            if not any(a.alive for a in agents):
                break
            for a in agents:
                if a.alive:
                    a.survival_steps += 1
            if policy == "brain":
                with torch.no_grad():
                    agents = run_simulation_step(agents, env_grid, brain, config, deterministic=deterministic)["agents"]
            else:
                _scripted_step(agents, env_grid, policy, config)

        truncated = any(a.alive for a in agents)
        stats.record(agents, truncated)
        survivors = [a for a in agents if a.alive]
        final_utility.extend(calculate_utility(a, config) for a in survivors)
        final_n.extend(a.n for a in survivors)
        final_h.extend(a.h for a in survivors)
        final_gold.extend(a.inv["g"] for a in survivors)
        dead = [a.survival_steps for a in agents if not a.alive]
        print(f"   Episode {ep:03d} | survived {len(survivors)}/{len(agents)} | "
              f"deaths at steps: {sorted(dead) if dead else '-'}")

    surv = np.array(stats.episode_survival)
    ci = 1.96 * surv.std(ddof=1) / np.sqrt(len(surv)) if len(surv) > 1 else float("nan")

    def fmt(values):
        return f"{np.mean(values):.2f} (sd {np.std(values):.2f})" if values else "n/a"

    print("=" * 70)
    print(f"  EVALUATION SUMMARY  policy='{policy}'{mode}")
    print(f"  Survival rate        : {surv.mean():.1%}  (95% CI +/- {ci:.1%}, per-episode sd {surv.std(ddof=1) if len(surv) > 1 else float('nan'):.1%})")
    print(f"  {stats.summary()}")
    print(f"  Survivors at end     : utility {fmt(final_utility)} | N {fmt(final_n)} | H {fmt(final_h)} | gold {fmt(final_gold)}")
    print("=" * 70)

    return {
        "policy": policy,
        "survival_rate": float(surv.mean()),
        "survival_ci95": float(ci),
        "episode_survival": surv.tolist(),
        "deaths": stats.deaths,
        "mean_death_age": stats.mean_death_age,
        "causes": dict(stats.causes),
        "final_utility": float(np.mean(final_utility)) if final_utility else float("nan"),
        "final_gold": float(np.mean(final_gold)) if final_gold else float("nan"),
    }
