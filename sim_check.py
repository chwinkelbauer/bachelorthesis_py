"""
Simulation check module performing Monte Carlo balance checking and mathematical feasibility analysis.
Ported from R sim_check.R to Python using NumPy.
"""

import numpy as np
from config import EnvConfig, DEFAULT_CONFIG
from environment import create_resource_map


def sim_check_robust(config: EnvConfig = DEFAULT_CONFIG, num_maps: int = 200) -> None:
    """
    Executes Monte Carlo resource analysis over `num_maps` grid maps and prints
    detailed yield and survival metrics.
    """
    print("====================================================================")
    print("     DETAILED ROBUST SIMULATION BALANCE CHECK (MONTE CARLO)         ")
    print(f"     (Averaging over {num_maps} random maps)                                ")
    print("====================================================================\n")

    n_rows = config.grid.n_grid
    n_cols = config.grid.k_grid

    food_means = np.zeros(num_maps)
    wood_means = np.zeros(num_maps)
    gold_means = np.zeros(num_maps)

    for i in range(num_maps):
        env_grid = create_resource_map(
            n=n_rows, k=n_cols,
            num_food=config.grid.n_food,
            num_wood=config.grid.n_wood,
            num_gold=config.grid.n_gold,
            decay_rate=config.grid.decay_rate,
            min_eff=config.grid.min_eff,
            seed=123 + i
        )
        food_means[i] = np.mean(env_grid[:, :, 0])
        wood_means[i] = np.mean(env_grid[:, :, 1])
        gold_means[i] = np.mean(env_grid[:, :, 2])

    avg_tile_food = float(np.mean(food_means))
    avg_tile_wood = float(np.mean(wood_means))
    avg_tile_gold = float(np.mean(gold_means))

    print(f"1. ROBUST MAP AVERAGE (after {num_maps} maps):")
    print(f"   - Food (Layer 1): Mean Tile Value = {avg_tile_food:.3f} (± {np.std(food_means):.3f})")
    print(f"   - Wood (Layer 2): Mean Tile Value = {avg_tile_wood:.3f} (± {np.std(wood_means):.3f})")
    print(f"   - Gold (Layer 3): Mean Tile Value = {avg_tile_gold:.3f} (± {np.std(gold_means):.3f})\n")

    n_start = config.start_agent.spec_start[0]
    h_start = config.start_agent.spec_start[1]
    dec_n = config.decrease[0]
    dec_h = config.decrease[1]
    dead_t_n = config.dead_time[0]
    dead_t_h = config.dead_time[1]

    max_life_hunger = (n_start / dec_n) + dead_t_n
    max_life_cold = (h_start / dec_h) + dead_t_h

    print("2. PASSIVE LIFESPAN (Without Resource Intake):")
    print(f"   - Max Steps (Hunger / Food): {max_life_hunger:.1f} steps")
    print(f"   - Max Steps (Cold / Wood):   {max_life_cold:.1f} steps\n")

    min_eff_val = min(config.grid.min_eff)
    max_eff_val = 1.0
    avg_gath_eff = float(np.mean(config.gath_eff))
    max_gath_eff = max(config.gath_eff)

    consume_yield_f = config.consume_yield[0]
    consume_yield_w = config.consume_yield[1]

    # --- FOOD DETAILED BREAKDOWN ---
    f_worst_yield = 1.0 * min_eff_val * avg_gath_eff * consume_yield_f
    f_worst_net = f_worst_yield - dec_n

    f_avg_yield = 1.0 * avg_tile_food * avg_gath_eff * consume_yield_f
    f_avg_net = f_avg_yield - dec_n

    f_best_yield = 1.0 * max_eff_val * max_gath_eff * consume_yield_f
    f_best_net = f_best_yield - dec_n

    print("3. DETAILED FOOD (HUNGER) BALANCE:")
    print(f"   - Consumption Drain:        {dec_n:.2f} / step")
    print(f"   - Worst-Case (Border Tile): Yield = {f_worst_yield:.2f} | Net Change = {f_worst_net:.2f} / step")
    print(f"   - Map-Avg (Random Wandering): Yield = {f_avg_yield:.2f} | Net Change = {f_avg_net:.2f} / step")
    print(f"   - Best-Case  (Hub Center):    Yield = {f_best_yield:.2f} | Net Change = {f_best_net:.2f} / step\n")

    # --- WOOD DETAILED BREAKDOWN ---
    w_worst_yield = 1.0 * min_eff_val * avg_gath_eff * consume_yield_w
    w_worst_net = w_worst_yield - dec_h

    w_avg_yield = 1.0 * avg_tile_wood * avg_gath_eff * consume_yield_w
    w_avg_net = w_avg_yield - dec_h

    w_best_yield = 1.0 * max_eff_val * max_gath_eff * consume_yield_w
    w_best_net = w_best_yield - dec_h

    print("4. DETAILED WOOD (COLD) BALANCE:")
    print(f"   - Consumption Drain:        {dec_h:.2f} / step")
    print(f"   - Worst-Case (Border Tile): Yield = {w_worst_yield:.2f} | Net Change = {w_worst_net:.2f} / step")
    print(f"   - Map-Avg (Random Wandering): Yield = {w_avg_yield:.2f} | Net Change = {w_avg_net:.2f} / step")
    print(f"   - Best-Case  (Hub Center):    Yield = {f_best_yield:.2f} | Net Change = {w_best_net:.2f} / step\n")

    # --- PROGRESSIVE UPKEEP & INVENTORY MECHANICS ANALYSIS ---
    print("5. PROGRESSIVE UPKEEP & INVENTORY MECHANICS (Anti-Hoarding):")
    print(f"   - Max Inventory Storage Capacity: {config.max_storage:.1f} weight units")
    print(f"   - Resource Unit Weights: Food = {config.weights.get('Food', 0.5)} | Wood = {config.weights.get('Wood', 1.0)} | Gold = {config.weights.get('Gold', 2.0)}")
    if getattr(config, 'progressive_drain', None) and config.progressive_drain.enabled:
        pd = config.progressive_drain
        print(f"   - Progressive Metabolic Drain: Exponents = {pd.exponent} | Scale = {pd.scale}")
        # Test drain at baseline (20, 15) vs high needs (60, 50)
        baseline_drain_n = dec_n * (1.0 + pd.scale[0] * (1.0 ** pd.exponent[0]))
        high_drain_n = dec_n * (1.0 + pd.scale[0] * ((60.0 / pd.ref_level[0]) ** pd.exponent[0]))
        baseline_drain_h = dec_h * (1.0 + pd.scale[1] * (1.0 ** pd.exponent[1]))
        high_drain_h = dec_h * (1.0 + pd.scale[1] * ((50.0 / pd.ref_level[1]) ** pd.exponent[1]))

        print(f"   - Nutrition Drain: Baseline N={pd.ref_level[0]:.0f} -> {baseline_drain_n:.3f}/step | High N=60 -> {high_drain_n:.3f}/step (+{((high_drain_n/dec_n)-1)*100:.1f}%)")
        print(f"   - Housing Drain:   Baseline H={pd.ref_level[1]:.0f} -> {baseline_drain_h:.3f}/step | High H=50 -> {high_drain_h:.3f}/step (+{((high_drain_h/dec_h)-1)*100:.1f}%)")
        print(f"   - Differentiated Spoilage:")
        print(f"     * Food (Perishable): Base = {pd.inventory_spoilage_rate[0]*100:.1f}%/step | Super-linear Exponent = {pd.inventory_spoilage_exp[0]}")
        print(f"     * Wood (Weatherable): Base = {pd.inventory_spoilage_rate[1]*100:.1f}%/step | Super-linear Exponent = {pd.inventory_spoilage_exp[1]}")
        print(f"     * Gold (Store of Value): Base = {pd.inventory_spoilage_rate[2]*100:.1f}%/step (Permanent & Imperishable)")
        print(f"   - Overburden Mechanics: Threshold = {pd.overburden_threshold*100:.0f}% capacity ({pd.overburden_threshold*config.max_storage:.1f} wt) | Extra Metabolic Burn = +{pd.overburden_metabolic_penalty*100:.0f}%\n")
    else:
        print("   - Progressive Drain is disabled (flat consumption).\n")

    print("6. SUMMARY & FEASIBILITY CONCLUSION:")
    if f_best_net < 0 or w_best_net < 0:
        print("   [CRITICAL] Even under absolute optimal conditions (Hub Center), at least")
        print("   one resource has a negative net balance. Survival is mathematically impossible.")
    elif f_avg_net < 0 or w_avg_net < 0:
        print("   [CHALLENGING] Random wandering results in a net loss. Agents MUST learn")
        print("   to navigate directly to resource hubs to survive.")
    else:
        print("   [FAVORABLE] The average map yield is sufficient to sustain agents even with")
        print("   suboptimal movement.")
    print("====================================================================")


if __name__ == "__main__":
    sim_check_robust(DEFAULT_CONFIG, num_maps=200)
