"""
Environment module for Grid World generation, resource distributions, and spatial vision extractions.
Ported from R gen_map.R and help.R to Python using NumPy.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from config import EnvConfig, DEFAULT_CONFIG


def create_resource_map(
    n: int,
    k: int,
    num_food: int,
    num_wood: int,
    num_gold: int,
    decay_rate: Tuple[float, float, float] = (0.12, 0.20, 0.45),
    min_eff: Tuple[float, float, float] = (0.70, 0.50, 0.60),
    seed: int = None
) -> np.ndarray:
    """
    Generates an (n, k, 3) NumPy array representing probabilities for [Food, Wood, Gold]
    based on exponential distance decay from localized resource hubs.
    """
    if seed is not None:
        np.random.seed(seed)

    def generate_centers(num: int, max_r: int, max_c: int) -> np.ndarray:
        if num <= 0:
            return np.empty((0, 2), dtype=int)
        r_coords = np.random.randint(0, max_r, size=num)
        c_coords = np.random.randint(0, max_c, size=num)
        return np.column_stack((r_coords, c_coords))

    centers_f = generate_centers(num_food, n, k)
    centers_w = generate_centers(num_wood, n, k)
    centers_g = generate_centers(num_gold, n, k)

    resource_map = np.zeros((n, k, 3), dtype=np.float32)
    centers_list = [centers_f, centers_w, centers_g]

    for r in range(n):
        for c in range(k):
            for layer in range(3):
                centers = centers_list[layer]
                if len(centers) == 0:
                    resource_map[r, c, layer] = min_eff[layer]
                else:
                    dists = np.sqrt((centers[:, 0] - r) ** 2 + (centers[:, 1] - c) ** 2)
                    min_dist = np.min(dists)
                    prob = np.exp(-decay_rate[layer] * min_dist)
                    resource_map[r, c, layer] = max(min_eff[layer], float(prob))

    return resource_map


def get_nxn_array(n: int, pos: Tuple[int, int], grid: np.ndarray) -> np.ndarray:
    """
    Extracts an n x n x 3 vision window centered at `pos` (r, c).
    Out-of-bound cells are padded with 0.0.
    """
    vision = np.zeros((n, n, 3), dtype=np.float32)
    grid_r, grid_c, _ = grid.shape
    half = (n - 1) // 2

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            r = pos[0] + i
            c = pos[1] + j

            if 0 <= r < grid_r and 0 <= c < grid_c:
                vision[i + half, j + half, :] = grid[r, c, :]

    return vision


def get_agent_occupancy_map(agents: List[Any], grid_shape: Tuple[int, int]) -> np.ndarray:
    """
    Creates a 2D binary matrix of shape `grid_shape` indicating agent presence (1.0) or absence (0.0).
    """
    occ = np.zeros(grid_shape, dtype=np.float32)
    for agent in agents:
        if getattr(agent, 'alive', True):
            r, c = agent.pos
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                occ[r, c] = 1.0
    return occ


def calculate_inventory_weight(inv: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculates total weight of resources currently held in an inventory.
    """
    wf = weights.get("Food", 1.0)
    ww = weights.get("Wood", 2.0)
    wg = weights.get("Gold", 0.1)

    return (inv.get("f", 0.0) * wf) + (inv.get("w", 0.0) * ww) + (inv.get("g", 0.0) * wg)


def calculate_utility(agent: Any, config: EnvConfig = DEFAULT_CONFIG) -> float:
    """
    Calculates logarithmic utility:
    U = w_n * log(max(0, N) + 1) + w_h * log(max(0, H) + 1) + w_g * log(max(0, Gold) + 1)
    """
    w_n, w_h, w_g = config.utility

    u_n = w_n * np.log(max(0.0, float(agent.n)) + 1.0)
    u_h = w_h * np.log(max(0.0, float(agent.h)) + 1.0)
    u_g = w_g * np.log(max(0.0, float(agent.inv["g"])) + 1.0)

    return float(u_n + u_h + u_g)


def extract_agent_observation(agent: Any, environment_grid: np.ndarray, config: EnvConfig = DEFAULT_CONFIG) -> np.ndarray:
    """
    Extracts complete observation vector for an agent:
    [Vision Window Flattened (sight*sight*3), Nutrition, Housing, Food_inv, Wood_inv, Gold_inv, Food_eff, Wood_eff, Gold_eff]
    Total dimension = 83 (for sight=5).
    """
    sight_size = config.sight
    vision_array = get_nxn_array(sight_size, agent.pos, environment_grid)
    vision_vec = vision_array.flatten()

    inv_vec = np.array([agent.inv["f"], agent.inv["w"], agent.inv["g"]], dtype=np.float32)
    state_vec = np.concatenate([
        vision_vec,
        np.array([agent.n, agent.h], dtype=np.float32),
        inv_vec,
        agent.eff.astype(np.float32)
    ])
    return state_vec
