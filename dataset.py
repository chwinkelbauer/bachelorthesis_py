"""
Dataset module for expert rule-based data collection, trajectory saving, and PyTorch Dataset loading.
Ported from R pre_learning/setup.R to Python.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any

from config import EnvConfig, DEFAULT_CONFIG
from environment import create_resource_map, extract_agent_observation
from agent import initialize_agents_isolated, execute_move, execute_gather, execute_consumption, Agent


def get_movement_towards(curr_r: int, curr_c: int, target_r: int, target_c: int) -> int:
    """
    Returns action index to move towards target coordinates:
      1: Up, 2: Down, 3: Left, 4: Right, 11: Stand
    """
    if target_r > curr_r:
        return 2  # Down
    if target_r < curr_r:
        return 1  # Up
    if target_c > curr_c:
        return 4  # Right
    if target_c < curr_c:
        return 3  # Left
    return 11  # Stand


def get_expert_decisions(
    agent: Agent,
    env_grid: np.ndarray,
    config: EnvConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Rule-based expert policy:
    1. Conservative consumption to satisfy hunger/cold.
    2. Priority sequence: Starvation (<30) -> Cold (<30) -> Wealth (Gold).
    3. Spatial navigation towards highest resource hubs within sight.
    """
    pos_r, pos_c = agent.pos
    sight = config.sight
    n_rows, n_cols, _ = env_grid.shape

    r_min = max(0, pos_r - sight)
    r_max = min(n_rows, pos_r + sight + 1)
    c_min = max(0, pos_c - sight)
    c_max = min(n_cols, pos_c + sight + 1)

    cur_food = env_grid[pos_r, pos_c, 0]
    cur_wood = env_grid[pos_r, pos_c, 1]
    cur_gold = env_grid[pos_r, pos_c, 2]

    # Consumption Logic
    consume_type = 1  # 1: None, 2: Food, 3: Wood
    consume_fraction = 0.0

    dec_f, dec_w = config.decrease
    yield_f, yield_w = config.consume_yield

    target_consume_f = (2.0 * dec_f) / yield_f
    target_consume_w = (2.0 * dec_w) / yield_w

    if agent.n < 35.0 and agent.inv["f"] > 0.1:
        consume_type = 2  # Food
        consume_fraction = min(1.0, float(target_consume_f / max(agent.inv["f"], 1e-5)))
    elif agent.h < 35.0 and agent.inv["w"] > 0.1:
        consume_type = 3  # Wood
        consume_fraction = min(1.0, float(target_consume_w / max(agent.inv["w"], 1e-5)))

    # Action Selection Logic
    action_idx = 11  # Default Stand

    if agent.n < 30.0:  # Priority 1: Hunger
        if cur_food > 0.3:
            action_idx = 5  # Gather Food
        else:
            sub_grid = env_grid[r_min:r_max, c_min:c_max, 0]
            max_pos = np.unravel_index(np.argmax(sub_grid), sub_grid.shape)
            target_r = r_min + max_pos[0]
            target_c = c_min + max_pos[1]
            action_idx = get_movement_towards(pos_r, pos_c, target_r, target_c)

    elif agent.h < 30.0:  # Priority 2: Cold
        if cur_wood > 0.3:
            action_idx = 6  # Gather Wood
        else:
            sub_grid = env_grid[r_min:r_max, c_min:c_max, 1]
            max_pos = np.unravel_index(np.argmax(sub_grid), sub_grid.shape)
            target_r = r_min + max_pos[0]
            target_c = c_min + max_pos[1]
            action_idx = get_movement_towards(pos_r, pos_c, target_r, target_c)

    else:  # Priority 3: Accumulate Gold
        if cur_gold > 0.3:
            action_idx = 7  # Gather Gold
        else:
            sub_grid = env_grid[r_min:r_max, c_min:c_max, 2]
            max_pos = np.unravel_index(np.argmax(sub_grid), sub_grid.shape)
            target_r = r_min + max_pos[0]
            target_c = c_min + max_pos[1]
            action_idx = get_movement_towards(pos_r, pos_c, target_r, target_c)

    return {
        "action_idx": action_idx,
        "consume_type": consume_type,
        "consume_fraction": consume_fraction,
        "drop_fraction": 0.0,
        "barter_flag": False
    }


def run_expert_collection(
    num_episodes: int = 25,
    config: EnvConfig = DEFAULT_CONFIG,
    save_path: str = "expert_dataset.pkl"
) -> str:
    """
    Runs multi-episode expert simulations and exports state-action pairs to pickle format.
    """
    print(f"-> Starting Expert Data Collection over {num_episodes} episodes...")

    n_rows = config.grid.n_grid
    n_cols = config.grid.k_grid
    n_agents = config.grid.n_agents

    all_states = []
    all_actions = []
    all_consume_types = []
    all_consume_amts = []

    for ep in range(1, num_episodes + 1):
        env_grid = create_resource_map(
            n=n_rows, k=n_cols,
            num_food=config.grid.n_food,
            num_wood=config.grid.n_wood,
            num_gold=config.grid.n_gold,
            decay_rate=config.grid.decay_rate,
            min_eff=config.grid.min_eff
        )
        agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)

        for step in range(200):
            if not any(a.alive for a in agents):
                break

            for i, agent in enumerate(agents):
                if not agent.alive:
                    continue

                state_vec = extract_agent_observation(agent, env_grid, config)
                decisions = get_expert_decisions(agent, env_grid, config)

                all_states.append(state_vec)
                all_actions.append(decisions["action_idx"])
                all_consume_types.append(decisions["consume_type"])
                all_consume_amts.append(decisions["consume_fraction"])

                # Execute expert action in environment
                agent = execute_move(agent, decisions["action_idx"], n_rows, n_cols)
                agent = execute_gather(agent, decisions["action_idx"], env_grid, decisions["drop_fraction"], config)
                agent = execute_consumption(agent, decisions["consume_type"], decisions["consume_fraction"], config)
                agents[i] = agent

        if ep % 5 == 0 or ep == num_episodes:
            print(f"   - Processed Episode {ep}/{num_episodes}")

    dataset_dict = {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int64),
        "consume_types": np.array(all_consume_types, dtype=np.int64),
        "consume_amts": np.array(all_consume_amts, dtype=np.float32)
    }

    with open(save_path, "wb") as f:
        pickle.dump(dataset_dict, f)

    print(f"-> Expert Dataset successfully saved to '{save_path}'! ({len(all_states)} samples)")
    return save_path


class ExpertDataset(Dataset):
    """
    PyTorch Dataset wrapper for Supervised Pre-Training.
    """
    def __init__(self, filepath: str = "expert_dataset.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file '{filepath}' not found. Run expert collection first.")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.states = torch.tensor(data["states"], dtype=torch.float32)
        # Convert 1-indexed R action indices (1..11) to 0-indexed PyTorch class indices (0..10)
        self.actions = torch.tensor(data["actions"] - 1, dtype=torch.long)
        # Convert 1-indexed R consume types (1..3) to 0-indexed PyTorch class indices (0..2)
        self.consume_types = torch.tensor(data["consume_types"] - 1, dtype=torch.long)
        self.consume_amts = torch.tensor(data["consume_amts"], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "consume_type": self.consume_types[idx],
            "consume_amt": self.consume_amts[idx]
        }
