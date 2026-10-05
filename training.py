"""
Training module featuring Supervised Pre-Training, Curriculum Learning, and MAPPO Reinforcement Learning.
Ported and modernized from R simulation_loop.R, pre_train.R, and curriculum.R to PyTorch autograd.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Any

from config import EnvConfig, DEFAULT_CONFIG
from environment import create_resource_map, extract_agent_observation, calculate_utility
from agent import initialize_agents_isolated, execute_move, execute_gather, execute_consumption, execute_barter_subloop, Agent
from network import MultiHeadGlobalBrain
from dataset import ExpertDataset


def run_supervised_pretraining(
    global_brain: MultiHeadGlobalBrain,
    dataset_path: str = "expert_dataset.pkl",
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 32
) -> MultiHeadGlobalBrain:
    """
    Supervised pre-training of MultiHeadGlobalBrain on expert trajectory dataset using PyTorch autograd.
    """
    print("-> Loading dataset for Supervised Pre-Training (Actions + Consumption)...")
    dataset = ExpertDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr)
    ce_loss_action = nn.CrossEntropyLoss()
    ce_loss_ctype = nn.CrossEntropyLoss()
    mse_loss_camt = nn.MSELoss()

    global_brain.train()
    print(f"-> Pre-training neural network on {len(dataset)} samples across operational heads...")

    for epoch in range(1, epochs + 1):
        total_epoch_loss = 0.0

        for batch in dataloader:
            states = batch["state"]
            actions = batch["action"]
            consume_types = batch["consume_type"]
            consume_amts = batch["consume_amt"]

            optimizer.zero_grad()
            out = global_brain(states)

            loss_act = ce_loss_action(out["logits_action"], actions)
            loss_ct = ce_loss_ctype(out["logits_consume_type"], consume_types)
            loss_camt = mse_loss_camt(out["consume_amt"], consume_amts)

            total_loss = loss_act + loss_ct + loss_camt
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item() * len(states)

        avg_loss = total_epoch_loss / len(dataset)
        print(f"   - Epoch {epoch:02d}/{epochs:02d} | Pre-Training Loss: {avg_loss:.4f}")

    print("-> Supervised Pre-Training completed successfully!")
    return global_brain


def run_simulation_step(
    agents: List[Agent],
    environment_grid: np.ndarray,
    global_brain: MultiHeadGlobalBrain,
    config: EnvConfig = DEFAULT_CONFIG,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.00025
) -> Dict[str, Any]:
    """
    Executes a single step of the multi-agent simulation loop and performs MAPPO Actor-Critic gradient updates.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr)

    n_rows, n_cols, _ = environment_grid.shape
    barter_decisions = [False] * len(agents)

    # Check if any agents are alive
    if not any(a.alive for a in agents):
        return {"agents": agents, "global_brain": global_brain, "reset_needed": True}

    global_brain.train()

    for i, agent in enumerate(agents):
        if not agent.alive:
            continue

        u_old = calculate_utility(agent, config)
        state_vec = extract_agent_observation(agent, environment_grid, config)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

        # Forward pass through neural brain
        out = global_brain(state_tensor)

        # Discrete Action Selection (Greedy / Argmax for policy execution)
        logits_action = out["logits_action"].squeeze(0)
        action_idx = int(torch.argmax(logits_action).item()) + 1  # 1-indexed

        logits_barter = out["logits_barter"].squeeze(0)
        barter_flag = (torch.argmax(logits_barter).item() == 1)

        logits_consume_type = out["logits_consume_type"].squeeze(0)
        consume_type = int(torch.argmax(logits_consume_type).item()) + 1  # 1-indexed

        drop_fraction = out["drop_amt"].squeeze(0).item()
        consume_fraction = out["consume_amt"].squeeze(0).item()
        v_pred = out["v_pred"].squeeze(0)

        barter_decisions[i] = barter_flag

        # Sequential Environment Execution
        agent = execute_move(agent, action_idx, n_rows, n_cols)
        agent = execute_gather(agent, action_idx, environment_grid, drop_fraction, config)
        agent = execute_consumption(agent, consume_type, consume_fraction, config)

        u_new = calculate_utility(agent, config)
        reward = u_new - u_old

        if not agent.alive:
            reward -= config.death_penal

        # MAPPO Advantage computation
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        advantage = (reward_tensor - v_pred).detach()

        # Critic MSE Loss
        loss_critic = F.mse_loss(v_pred, reward_tensor)

        # Policy Losses weighted by Advantage
        log_prob_act = F.log_softmax(logits_action, dim=-1)[action_idx - 1]
        loss_action = -log_prob_act * advantage

        log_prob_ct = F.log_softmax(logits_consume_type, dim=-1)[consume_type - 1]
        loss_ctype = -log_prob_ct * advantage

        # Continuous control penalty weighted by advantage
        loss_camt = -advantage * torch.log(out["consume_amt"].squeeze(0) + 1e-8)
        loss_damt = -advantage * torch.log(out["drop_amt"].squeeze(0) + 1e-8)

        total_step_loss = loss_critic + loss_action + loss_ctype + loss_camt + loss_damt

        optimizer.zero_grad()
        total_step_loss.backward()
        optimizer.step()

        agents[i] = agent

    # Execute Barter Subloop synchronously after all decisions
    agents = execute_barter_subloop(agents, barter_decisions, config)

    return {"agents": agents, "global_brain": global_brain, "reset_needed": False}


def run_curriculum_training(
    global_brain: MultiHeadGlobalBrain,
    config: EnvConfig = DEFAULT_CONFIG,
    lr: float = 0.001
) -> MultiHeadGlobalBrain:
    """
    Executes Curriculum Learning with progressive difficulty scaling on resource drain.
    """
    steps_each = config.pre_training.steps_each
    prog_step = config.pre_training.progression

    multipliers = np.arange(0.0, 1.0, prog_step)  # e.g., 0.0, 0.25, 0.50, 0.75
    num_stages = len(multipliers)
    total_steps = num_stages * steps_each

    print(f"-> Entering Curriculum Learning ({num_stages} stages, {steps_each} steps each, Total: {total_steps} steps)...")

    n_rows = config.grid.n_grid
    n_cols = config.grid.k_grid
    n_agents = config.grid.n_agents

    env_grid = create_resource_map(
        n=n_rows, k=n_cols,
        num_food=config.grid.n_food,
        num_wood=config.grid.n_wood,
        num_gold=config.grid.n_gold,
        decay_rate=config.grid.decay_rate,
        min_eff=config.grid.min_eff
    )
    agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)

    dead_lifespans = []
    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr)

    for step in range(1, total_steps + 1):
        stage_idx = min(int(np.ceil(step / steps_each)) - 1, num_stages - 1)
        current_multiplier = float(multipliers[stage_idx])

        current_config = copy.deepcopy(config)
        current_config.decrease = (
            config.decrease[0] * current_multiplier,
            config.decrease[1] * current_multiplier
        )
        stage_name = f"Stage {stage_idx + 1} (Drain: {int(current_multiplier * 100)}%)"

        for agent in agents:
            if agent.alive:
                agent.survival_steps += 1

        sim_output = run_simulation_step(agents, env_grid, global_brain, current_config, optimizer=optimizer, lr=lr)

        if sim_output["reset_needed"]:
            for a in agents:
                dead_lifespans.append(a.survival_steps)

            agents = initialize_agents_isolated(n_agents, n_rows, n_cols, current_config)
            continue

        agents = sim_output["agents"]
        global_brain = sim_output["global_brain"]

        if step % 500 == 0 or step == total_steps:
            utilities = [calculate_utility(a, current_config) for a in agents]
            mean_lifespan = float(np.mean(dead_lifespans)) if len(dead_lifespans) > 0 else float(np.mean([a.survival_steps for a in agents]))
            print(f"Step {step:05d} [{stage_name}] | Mean Utility: {np.mean(utilities):.2f} | Mean Lifespan: {mean_lifespan:.1f} steps")

    print("-> Curriculum Training successfully completed!")
    return global_brain


def run_training_experiment(
    total_steps: int = 20000,
    x_steps: int = 2500,
    config: EnvConfig = DEFAULT_CONFIG,
    pre_trained_brain: Optional[MultiHeadGlobalBrain] = None,
    lr: float = 0.0005
) -> MultiHeadGlobalBrain:
    """
    Runs main MAPPO multi-agent reinforcement learning experiment loop.
    """
    n_rows = config.grid.n_grid
    n_cols = config.grid.k_grid
    n_agents = config.grid.n_agents

    env_grid = create_resource_map(
        n=n_rows, k=n_cols,
        num_food=config.grid.n_food,
        num_wood=config.grid.n_wood,
        num_gold=config.grid.n_gold,
        decay_rate=config.grid.decay_rate,
        min_eff=config.grid.min_eff
    )

    if pre_trained_brain is not None:
        print("-> Using pre-trained Global Brain for training experiment...")
        global_brain = pre_trained_brain
    else:
        print("-> Initializing new random Global Brain...")
        global_brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64)

    agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)
    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr)

    episode = 1
    episode_start_step = 1
    episode_lengths = []
    interval_lifespans = []

    print(f"-> Starting RL Training with {total_steps} steps (chunk size {x_steps})...")

    for step in range(1, total_steps + 1):
        sim_output = run_simulation_step(agents, env_grid, global_brain, config, optimizer=optimizer, lr=lr)

        if sim_output["reset_needed"]:
            ep_length = step - episode_start_step + 1
            episode_lengths.append(ep_length)
            interval_lifespans.append(ep_length)

            print(f"Step {step} | Episode {episode} finished (all dead) after {ep_length} steps. Resetting...")

            episode += 1
            episode_start_step = step + 1
            agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)
            continue

        agents = sim_output["agents"]
        global_brain = sim_output["global_brain"]

        if step % x_steps == 0:
            avg_lifespan = float(np.mean(interval_lifespans)) if len(interval_lifespans) > 0 else 0.0
            utilities = [calculate_utility(a, config) for a in agents]
            alive_count = sum(1 for a in agents if a.alive)

            print("----------------------------------------------------")
            print(f">>> INTERVAL REPORT (Steps {step - x_steps + 1} - {step}) <<<")
            print(f"    - Average Episode Lifespan: {avg_lifespan:.2f} steps")
            print(f"    - Total Episodes in Interval: {len(interval_lifespans)}")
            print(f"    - Current Living Agents: {alive_count}/{n_agents}")
            print(f"    - Current Mean Utility: {np.mean(utilities):.2f}")
            print("----------------------------------------------------")

            interval_lifespans = []

        elif step % 100 == 0:
            utilities = [calculate_utility(a, config) for a in agents]
            alive_count = sum(1 for a in agents if a.alive)
            print(f"Step {step:04d} | Ep: {episode} | Living: {alive_count}/{n_agents} | Mean Utility: {np.mean(utilities):.2f}")

    print("-> RL Training completed!")
    if len(episode_lengths) > 0:
        print(f"-> Overall Average Episode Lifespan across {len(episode_lengths)} episodes: {np.mean(episode_lengths):.2f} steps")

    return global_brain
