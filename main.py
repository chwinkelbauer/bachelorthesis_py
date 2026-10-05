"""
Main Entry Point CLI for the Multi-Agent Grid World Economic Simulation in Python.
Supports balance checks, expert dataset collection, supervised pre-training, curriculum training,
and full MAPPO reinforcement learning experiments.
"""

import argparse 
import torch
import numpy as np

from config import EnvConfig, DEFAULT_CONFIG
from environment import create_resource_map, calculate_utility
from agent import initialize_agents_isolated
from network import MultiHeadGlobalBrain
from dataset import run_expert_collection
from training import (
    run_supervised_pretraining,
    run_curriculum_training,
    run_training_experiment,
    run_simulation_step
)
from sim_check import sim_check_robust


def run_test_head_function(steps: int = 10, encoder_type: str = "mlp") -> None:
    """
    Executes a short sanity check run to verify neural brain parameters and simulation steps.
    Ported from test_runner.R test_head_function().
    """
    config = DEFAULT_CONFIG
    n_rows = config.grid.n_grid
    n_cols = config.grid.k_grid
    n_agents = config.grid.n_agents

    print(f"-> Starting Sanity Check Run: {n_agents} Agents | Grid: {n_rows}x{n_cols} | Sight: {config.sight}x{config.sight} | Encoder: {encoder_type} | Steps: {steps}")

    env_grid = create_resource_map(
        n=n_rows, k=n_cols,
        num_food=config.grid.n_food,
        num_wood=config.grid.n_wood,
        num_gold=config.grid.n_gold,
        decay_rate=config.grid.decay_rate,
        min_eff=config.grid.min_eff
    )

    global_brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64, encoder_type=encoder_type)
    agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)

    optimizer = torch.optim.Adam(global_brain.parameters(), lr=0.01)

    for step in range(1, steps + 1):
        # Inspect weight snapshot
        if encoder_type == "mlp":
            w_before = global_brain.encoder.fc.weight[0, 0].item()
        else:
            w_before = global_brain.encoder.node_mlp[0].weight[0, 0].item()

        sim_output = run_simulation_step(agents, env_grid, global_brain, config, optimizer=optimizer, lr=0.01)

        if sim_output["reset_needed"]:
            print(f"Step {step:02d} | All agents died! Resetting episode...")
            agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)
            env_grid = create_resource_map(
                n=n_rows, k=n_cols,
                num_food=config.grid.n_food,
                num_wood=config.grid.n_wood,
                num_gold=config.grid.n_gold,
                decay_rate=config.grid.decay_rate,
                min_eff=config.grid.min_eff
            )
            continue

        agents = sim_output["agents"]
        global_brain = sim_output["global_brain"]

        if encoder_type == "mlp":
            w_after = global_brain.encoder.fc.weight[0, 0].item()
        else:
            w_after = global_brain.encoder.node_mlp[0].weight[0, 0].item()

        print(f"Step {step:02d} | Shared Encoder Weight W[0,0]: {w_before:.4f} -> {w_after:.4f}")

    print("-> Sanity Check Simulation completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Economic Simulation & MAPPO RL (Python)")
    parser.add_argument("--check", action="store_true", help="Run Monte Carlo balance check")
    parser.add_argument("--test", action="store_true", help="Run quick 10-step sanity check test run")
    parser.add_argument("--collect", action="store_true", help="Collect expert dataset")
    parser.add_argument("--episodes", type=int, default=25, help="Number of expert collection episodes")
    parser.add_argument("--pretrain", action="store_true", help="Run supervised pre-training")
    parser.add_argument("--epochs", "-epochs", "-e", type=int, default=10, help="Pre-training epochs")
    parser.add_argument("--curriculum", action="store_true", help="Run curriculum learning")
    parser.add_argument("--train", action="store_true", help="Run full MAPPO training experiment")
    parser.add_argument("--total-steps", "-total-steps", "-s", type=int, default=20000, help="Total steps for RL experiment")
    parser.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "gnn"], help="Encoder architecture (mlp or gnn)")
    parser.add_argument("--save-brain", type=str, default=None, help="File path to save the trained global brain weights (e.g. pretrained_brain.pt)")
    parser.add_argument("--load-brain", type=str, default=None, help="File path to load pre-trained global brain weights (e.g. pretrained_brain.pt)")

    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.check:
        sim_check_robust(config, num_maps=200)

    if args.test:
        run_test_head_function(steps=10, encoder_type=args.encoder)

    if args.collect:
        run_expert_collection(num_episodes=args.episodes, config=config)

    brain = None

    # Load brain weights from file if specified
    if args.load_brain is not None:
        load_path = args.load_brain
        brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64, encoder_type=args.encoder)
        brain.load_state_dict(torch.load(load_path))
        print(f"-> Successfully loaded pre-trained neural network weights from '{load_path}'!")

    if args.pretrain:
        if brain is None:
            brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64, encoder_type=args.encoder)
        brain = run_supervised_pretraining(brain, dataset_path="expert_dataset.pkl", epochs=args.epochs)
        if args.save_brain is None and not args.curriculum and not args.train:
            args.save_brain = "pretrained_brain.pt"

    if args.curriculum:
        if brain is None:
            brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64, encoder_type=args.encoder)
        brain = run_curriculum_training(brain, config=config)
        if args.save_brain is None and not args.train:
            args.save_brain = "curriculum_brain.pt"

    # Auto-save weights if save path is set or defaulted
    if args.save_brain and brain is not None:
        torch.save(brain.state_dict(), args.save_brain)
        print(f"-> Saved global brain weights to '{args.save_brain}'")

    if args.train:
        run_training_experiment(total_steps=args.total_steps, x_steps=2500, config=config, pre_trained_brain=brain)

    if not any([args.check, args.test, args.collect, args.pretrain, args.curriculum, args.train]):
        print("No specific task flag provided. Running quick test suite (--test)...")
        run_test_head_function(steps=10, encoder_type=args.encoder)


if __name__ == "__main__":
    main()
