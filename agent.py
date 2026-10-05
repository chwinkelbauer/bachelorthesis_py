"""
Agent module defining economic agent attributes, initialization, movement, resource gathering,
consumption, and barter trade subloop mechanics.
Ported from R agent_init.R and simulation_loop.R to Python.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from config import EnvConfig, DEFAULT_CONFIG
from environment import calculate_inventory_weight


class Agent:
    def __init__(
        self,
        agent_id: int,
        pos: Tuple[int, int],
        max_storage: float,
        eff: np.ndarray,
        q_start: Tuple[float, float, float],
        spec_start: Tuple[float, float]
    ):
        self.id: int = agent_id
        self.pos: Tuple[int, int] = pos
        self.max_storage: float = max_storage
        self.eff: np.ndarray = eff  # np.ndarray shape (3,) -> [Food, Wood, Gold]

        self.n: float = float(spec_start[0])  # Nutrition
        self.h: float = float(spec_start[1])  # Housing

        self.inv: Dict[str, float] = {
            "f": float(q_start[0]),
            "w": float(q_start[1]),
            "g": float(q_start[2])
        }

        self.alive: bool = True
        self.hunger_steps: int = 0
        self.cold_steps: int = 0
        self.survival_steps: int = 0

        self.history: Dict[str, List[Any]] = {
            "states": [],
            "chosen_offers": [],
            "actions": [],
            "rewards": []
        }

    def copy_state(self) -> Dict[str, Any]:
        """Returns a serializable snapshot of agent attributes."""
        return {
            "id": self.id,
            "pos": self.pos,
            "n": self.n,
            "h": self.h,
            "inv": dict(self.inv),
            "alive": self.alive,
            "hunger_steps": self.hunger_steps,
            "cold_steps": self.cold_steps,
            "survival_steps": self.survival_steps
        }


def initialize_agents_isolated(
    num_agents: int,
    n: int,
    k: int,
    env_config: EnvConfig = DEFAULT_CONFIG
) -> List[Agent]:
    """
    Initializes a population of agents placed at distinct (non-overlapping) coordinates on an n x k grid.
    """
    if num_agents > (n * k):
        raise ValueError(f"Error: Number of agents ({num_agents}) exceeds available tiles ({n * k})!")

    max_storage = env_config.max_storage
    q_start = env_config.start_agent.q_start
    spec_start = env_config.start_agent.spec_start
    gath_eff_range = env_config.gath_eff

    agents: List[Agent] = []
    occupied_positions = set()

    for i in range(1, num_agents + 1):
        while True:
            potential_pos = (np.random.randint(0, n), np.random.randint(0, k))
            if potential_pos not in occupied_positions:
                occupied_positions.add(potential_pos)
                break

        # Generate individual gathering efficiencies for [Food, Wood, Gold]
        eff = np.random.uniform(gath_eff_range[0], gath_eff_range[1], size=3)

        agent = Agent(
            agent_id=i,
            pos=potential_pos,
            max_storage=max_storage,
            eff=eff,
            q_start=q_start,
            spec_start=spec_start
        )
        agents.append(agent)

    return agents


def execute_move(agent: Agent, action_idx: int, n_rows: int, n_cols: int) -> Agent:
    """
    Executes discrete spatial movement:
    action_idx:
      1: Move Up (r - 1)
      2: Move Down (r + 1)
      3: Move Left (c - 1)
      4: Move Right (c + 1)
      5-11: No movement (Stay)
    """
    r, c = agent.pos
    if action_idx == 1:
        r = max(0, r - 1)
    elif action_idx == 2:
        r = min(n_rows - 1, r + 1)
    elif action_idx == 3:
        c = max(0, c - 1)
    elif action_idx == 4:
        c = min(n_cols - 1, c + 1)

    agent.pos = (r, c)
    return agent


def execute_gather(
    agent: Agent,
    action_idx: int,
    environment_grid: np.ndarray,
    drop_fraction: float,
    config: EnvConfig = DEFAULT_CONFIG
) -> Agent:
    """
    Executes gathering or dropping mechanics based on action_idx:
      5: Gather Food
      6: Gather Wood
      7: Gather Gold
      8: Drop Food by drop_fraction
      9: Drop Wood by drop_fraction
     10: Drop Gold by drop_fraction
    """
    r, c = agent.pos
    current_weight = calculate_inventory_weight(agent.inv, config.weights)

    # Gathering Actions (5-7)
    if 5 <= action_idx <= 7:
        res_idx = action_idx - 5  # 0: Food, 1: Wood, 2: Gold
        tile_prob = environment_grid[r, c, res_idx]
        eff_factor = agent.eff[res_idx]

        harvest_amt = float(tile_prob * eff_factor)

        res_key = ["f", "w", "g"][res_idx]
        res_weight_unit = config.weights.get(["Food", "Wood", "Gold"][res_idx], 1.0)

        potential_add_weight = harvest_amt * res_weight_unit

        if (current_weight + potential_add_weight) <= agent.max_storage:
            agent.inv[res_key] += harvest_amt
        else:
            allowed_weight = max(0.0, agent.max_storage - current_weight)
            allowed_amt = allowed_weight / res_weight_unit if res_weight_unit > 0 else 0.0
            agent.inv[res_key] += allowed_amt

    # Dropping Actions (8-10)
    elif 8 <= action_idx <= 10:
        res_idx = action_idx - 8  # 0: Food, 1: Wood, 2: Gold
        res_key = ["f", "w", "g"][res_idx]

        drop_amount = agent.inv[res_key] * float(drop_fraction)
        agent.inv[res_key] = max(0.0, agent.inv[res_key] - drop_amount)

    return agent


def execute_consumption(
    agent: Agent,
    consume_type: int,
    consume_fraction: float,
    config: EnvConfig = DEFAULT_CONFIG
) -> Agent:
    """
    Executes physiological consumption of inventory goods and updates survival indicators:
    consume_type:
      1: Consume Nothing
      2: Consume Food (restores Nutrition N)
      3: Consume Wood (restores Housing H)
    """
    # Passive drain per step
    agent.n = max(0.0, agent.n - config.decrease[0])
    agent.h = max(0.0, agent.h - config.decrease[1])

    yield_f, yield_w = config.consume_yield

    if consume_type == 2:  # Food
        consumed_amt = agent.inv["f"] * float(consume_fraction)
        if consumed_amt > 0.0 and agent.inv["f"] >= consumed_amt:
            agent.inv["f"] = max(0.0, agent.inv["f"] - consumed_amt)
            agent.n += consumed_amt * yield_f

    elif consume_type == 3:  # Wood
        consumed_amt = agent.inv["w"] * float(consume_fraction)
        if consumed_amt > 0.0 and agent.inv["w"] >= consumed_amt:
            agent.inv["w"] = max(0.0, agent.inv["w"] - consumed_amt)
            agent.h += consumed_amt * yield_w

    # Update Hunger Steps Counter
    if agent.n > 1.0:
        agent.hunger_steps = 0  # Fully satisfied -> Reset
    elif agent.n > 0.0:
        pass  # Between 0 and 1 -> Starvation paused, keep counter as-is
    else:
        agent.n = 0.0
        agent.hunger_steps += 1  # Actively starving -> Increment

    # Update Cold Steps Counter
    if agent.h > 1.0:
        agent.cold_steps = 0  # Fully warm -> Reset
    elif agent.h > 0.0:
        pass  # Starvation paused
    else:
        agent.h = 0.0
        agent.cold_steps += 1  # Actively freezing -> Increment

    # Check Death Conditions
    if agent.hunger_steps >= config.dead_time[0] or agent.cold_steps >= config.dead_time[1]:
        agent.alive = False

    return agent


def execute_barter_subloop(
    agents: List[Agent],
    barter_decisions: List[bool],
    config: EnvConfig = DEFAULT_CONFIG
) -> List[Agent]:
    """
    Framework hook for synchronous barter trade matching loop among willing agents.
    Can be expanded with graph-based or pairwise exchange mechanics.
    """
    return agents
