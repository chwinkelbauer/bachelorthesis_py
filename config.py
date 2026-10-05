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
    q_start: Tuple[float, float, float] = (4.0, 3.0, 1.0)  # Food, Wood, Gold (Starting weight = 4*0.5 + 3*1.0 + 1*2.0 = 7.0)
    spec_start: Tuple[float, float] = (20.0, 15.0)          # Nutrition, Housing

@dataclass
class PreTrainingConfig:
    steps_each: int = 2000
    progression: float = 0.25

@dataclass
class ProgressiveDrainConfig:
    """
    Configuration for progressive metabolic upkeep, differentiated inventory spoilage,
    and overburden / carrying capacity penalties (Anti-Hoarding).
    """
    enabled: bool = True
    exponent: Tuple[float, float] = (2.0, 2.0)               # Exponent for (Nutrition, Housing) progressive metabolic drain
    scale: Tuple[float, float] = (0.0015, 0.0015)            # Scaling factor for super-linear metabolic drain with high needs
    ref_level: Tuple[float, float] = (20.0, 15.0)            # Reference baseline level for needs (N, H)
    
    # Differentiated Spoilage Rates per step: Food=2.5% (high), Wood=0.5% (moderate), Gold=0.0% (permanent store of value)
    inventory_spoilage_rate: Tuple[float, float, float] = (0.025, 0.005, 0.0)
    inventory_spoilage_exp: Tuple[float, float, float] = (1.8, 1.5, 1.0)       # Super-linear spoilage exponent for large stockpiles
    spoilage_threshold: Tuple[float, float, float] = (3.0, 3.0, 0.0)          # Stockpile threshold above which super-linear spoilage accelerates

    # Carrying Load & Overburden Mechanics
    overburden_threshold: float = 0.70                       # Load ratio (weight / max_storage) where overburden begins
    overburden_metabolic_penalty: float = 0.30               # Extra metabolic drain factor when heavily carrying goods
    overburden_gathering_penalty: float = 0.2               # Gathering efficiency reduction when inventory is nearly full

@dataclass
class RewardConfig:
    """
    Configuration for reinforcement learning reward shaping and stability.
    """
    survival_bonus: float = 0.15                             # Positive reward per step survived
    death_penalty: float = 20.0                              # Scaled penalty on agent death
    danger_threshold: Tuple[float, float] = (10.0, 8.0)      # (N, H) threshold below which stress warning applies
    danger_penalty_scale: float = 0.04                       # Gradient stress penalty when approaching starvation/cold
    overburden_penalty: float = 0.06                         # Disutility penalty when inventory is overburdened (> 70%)
    entropy_coeff: float = 0.015                             # Entropy bonus for exploration in MAPPO policy gradients
    advantage_clip: float = 5.0                              # Maximum absolute advantage clipping for stability

@dataclass
class EnvConfig:
    max_storage: float = 15.0                                # Max inventory weight limit (sigma_max)
    sight: int = 5
    gath_eff: Tuple[float, float] = (0.5, 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "None": 0.0,
        "Food": 0.5,                                         # Food weight per unit
        "Wood": 1.0,                                         # Wood weight per unit
        "Gold": 2.0                                          # Gold weight per unit (heavy precious metal)
    })
    dead_time: Tuple[int, int] = (5, 8)            # (max_hunger_steps, max_cold_steps)
    utility: Tuple[float, float, float] = (1.0, 1.2, 0.1)  # weights for (Nutrition, Housing, Gold)
    death_penal: float = 20.0
    consume_yield: Tuple[float, float] = (3.0, 2.5)       # (Food -> N, Wood -> H)
    decrease: Tuple[float, float] = (0.4, 0.3)            # Baseline (N_drain, H_drain per step)
    grid: GridConfig = field(default_factory=GridConfig)
    start_agent: StartAgentConfig = field(default_factory=StartAgentConfig)
    pre_training: PreTrainingConfig = field(default_factory=PreTrainingConfig)
    progressive_drain: ProgressiveDrainConfig = field(default_factory=ProgressiveDrainConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @property
    def input_dim(self) -> int:
        """Dynamically compute input dimension based on sight size:
        vision (sight^2 * 3) + needs (2) + inventory (3) + efficiencies (3)
        For sight=5: 5*5*3 + 2 + 3 + 3 = 83.
        """
        return (self.sight * self.sight * 3) + 2 + 3 + 3


# Default global environment configuration instance
DEFAULT_CONFIG = EnvConfig()
