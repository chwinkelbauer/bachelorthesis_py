"""
Survival statistics shared by curriculum training, MAPPO training and evaluation.
Separates agent deaths (with age and cause) from agents that survived until their episode was truncated,
so that reported lifespans are never inflated by agents that are still alive.
"""

import random
from typing import List, Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch RNGs so that runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def death_cause(agent: Any, dead_time) -> str:
    """Returns 'hunger', 'cold' or 'both' for a dead agent."""
    hunger = agent.hunger_steps >= dead_time[0]
    cold = agent.cold_steps >= dead_time[1]
    if hunger and cold:
        return "both"
    return "hunger" if hunger else "cold"


class EpisodeStats:
    """Accumulates survival statistics over finished episodes."""

    def __init__(self, dead_time):
        self.dead_time = dead_time
        self.reset()

    def reset(self) -> None:
        self.episodes = 0
        self.died_out = 0                     # Episodes ended because all agents died
        self.capped = 0                       # Episodes truncated at the episode length cap
        self.agents_total = 0
        self.survivors = 0                    # Agents alive at the end of their episode
        self.death_ages: List[int] = []
        self.causes: Dict[str, int] = {"hunger": 0, "cold": 0, "both": 0}
        self.episode_survival: List[float] = []  # Per-episode survival fraction

    def record(self, agents: List[Any], truncated: bool) -> None:
        self.episodes += 1
        if truncated:
            self.capped += 1
        else:
            self.died_out += 1
        alive = 0
        for a in agents:
            if a.alive:
                alive += 1
            else:
                self.death_ages.append(a.survival_steps)
                self.causes[death_cause(a, self.dead_time)] += 1
        self.survivors += alive
        self.agents_total += len(agents)
        self.episode_survival.append(alive / max(1, len(agents)))

    @property
    def deaths(self) -> int:
        return len(self.death_ages)

    @property
    def survival_rate(self) -> float:
        return self.survivors / self.agents_total if self.agents_total else float("nan")

    @property
    def mean_death_age(self) -> float:
        return float(np.mean(self.death_ages)) if self.death_ages else float("nan")

    def summary(self) -> str:
        if self.episodes == 0:
            return "no finished episode"
        age = f"{self.mean_death_age:.1f}" if self.death_ages else "n/a"
        return (f"Episodes: {self.episodes} (died out: {self.died_out}, reached cap: {self.capped}) | "
                f"Survived to end: {self.survivors}/{self.agents_total} ({self.survival_rate:.0%}) | "
                f"Deaths: {self.deaths} (mean age {age}; hunger {self.causes['hunger']}, "
                f"cold {self.causes['cold']}, both {self.causes['both']})")
