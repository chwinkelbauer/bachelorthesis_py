"""
Configuration settings for the Multi-Agent Grid World Economic Simulation.
Ported from R config.R to Python.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class GridConfig:
    n_agents: int = 10
    n_grid: int = 10
    k_grid: int = 10
    n_food: int = 3
    n_wood: int = 2
    n_gold: int = 1
    decay_rate: Tuple[float, float, float] = (0.12, 0.20, 0.45)
    min_eff: Tuple[float, float, float] = (0.70, 0.50, 0.60)

@dataclass
class StartAgentConfig:
    q_start: Tuple[float, float, float] = (5.0, 5.0, 1.0)  # Food, Wood, Gold
    spec_start: Tuple[float, float] = (20.0, 15.0)          # Nutrition, Housing

@dataclass
class PreTrainingConfig:
    steps_each: int = 2000
    progression: float = 0.25

@dataclass
class EnvConfig:
    max_storage: float = 50.0
    sight: int = 5
    gath_eff: Tuple[float, float] = (0.5, 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "None": 0.0,
        "Food": 1.0,
        "Wood": 2.0,
        "Gold": 0.1
    })
    dead_time: Tuple[int, int] = (5, 8)            # (max_hunger_steps, max_cold_steps)
    utility: Tuple[float, float, float] = (1.0, 1.2, 0.1)  # weights for (Nutrition, Housing, Gold)
    death_penal: float = 35.0
    consume_yield: Tuple[float, float] = (3.0, 2.5)       # (Food -> N, Wood -> H)
    decrease: Tuple[float, float] = (0.4, 0.3)            # (N_drain, H_drain per step)
    grid: GridConfig = field(default_factory=GridConfig)
    start_agent: StartAgentConfig = field(default_factory=StartAgentConfig)
    pre_training: PreTrainingConfig = field(default_factory=PreTrainingConfig)

    @property
    def input_dim(self) -> int:
        """Dynamically compute input dimension based on sight size:
        vision (sight^2 * 3) + needs (2) + inventory (3) + efficiencies (3)
        For sight=5: 5*5*3 + 2 + 3 + 3 = 83.
        """
        return (self.sight * self.sight * 3) + 2 + 3 + 3


# Default global environment configuration instance
DEFAULT_CONFIG = EnvConfig()
