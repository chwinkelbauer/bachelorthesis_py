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
from environment import create_resource_map, extract_agent_observation, calculate_utility, calculate_inventory_weight
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


class MAPPORolloutBuffer:
    """
    Trajectory Rollout Buffer for multi-agent trajectory collection and Generalized Advantage Estimation (GAE).
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_prob_acts: List[float] = []
        self.consume_types: List[int] = []
        self.log_prob_cts: List[float] = []
        self.consume_amts: List[float] = []
        self.drop_amts: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob_act: float,
        consume_type: int,
        log_prob_ct: float,
        consume_amt: float,
        drop_amt: float,
        reward: float,
        done: bool,
        value: float
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_prob_acts.append(log_prob_act)
        self.consume_types.append(consume_type)
        self.log_prob_cts.append(log_prob_ct)
        self.consume_amts.append(consume_amt)
        self.drop_amts.append(drop_amt)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, ...]:
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            non_terminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        # Normalize advantages over the batch for stable policy gradient steps
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return (
            torch.tensor(np.array(self.states), dtype=torch.float32),
            torch.tensor(np.array(self.actions), dtype=torch.long),
            torch.tensor(np.array(self.log_prob_acts), dtype=torch.float32),
            torch.tensor(np.array(self.consume_types), dtype=torch.long),
            torch.tensor(np.array(self.log_prob_cts), dtype=torch.float32),
            torch.tensor(np.array(self.consume_amts), dtype=torch.float32),
            torch.tensor(np.array(self.drop_amts), dtype=torch.float32),
            torch.tensor(norm_advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32)
        )


def update_mappo_policy(
    global_brain: MultiHeadGlobalBrain,
    optimizer: torch.optim.Optimizer,
    buffer: MAPPORolloutBuffer,
    last_value: float = 0.0,
    config: EnvConfig = DEFAULT_CONFIG,
    ref_brain: Optional[MultiHeadGlobalBrain] = None,
    bc_coeff: float = 0.0
) -> Dict[str, float]:
    """
    Performs PPO ratio-clipped mini-batch policy gradient and value function updates.
    """
    mp = getattr(config, 'mappo', None)
    gamma = mp.gamma if mp else 0.99
    gae_lambda = mp.gae_lambda if mp else 0.95
    ppo_epochs = mp.ppo_epochs if mp else 4
    mini_batch_size = mp.mini_batch_size if mp else 64
    clip_eps = mp.clip_epsilon if mp else 0.20
    val_coeff = mp.value_loss_coeff if mp else 0.50
    entropy_coeff = mp.entropy_coeff if mp else 0.012
    max_grad_norm = mp.max_grad_norm if mp else 0.50

    states, actions, old_lp_act, ctypes, old_lp_ct, camts, damts, advantages, returns = (
        buffer.compute_returns_and_advantages(last_value, gamma=gamma, gae_lambda=gae_lambda)
    )

    num_samples = len(states)
    if num_samples == 0:
        return {}

    global_brain.train()
    total_loss_sum = 0.0

    for _ in range(ppo_epochs):
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, mini_batch_size):
            batch_idx = indices[start : start + mini_batch_size]

            b_states = states[batch_idx]
            b_actions = actions[batch_idx]
            b_old_lp_act = old_lp_act[batch_idx]
            b_ctypes = ctypes[batch_idx]
            b_old_lp_ct = old_lp_ct[batch_idx]
            b_camts = camts[batch_idx]
            b_damts = damts[batch_idx]
            b_adv = advantages[batch_idx]
            b_ret = returns[batch_idx]

            eval_out = global_brain.evaluate_actions_batch(b_states, b_actions, b_ctypes, b_camts, b_damts)

            # PPO Ratio Clipping for Discrete Action Policy
            ratio_act = torch.exp(eval_out["log_prob_act"] - b_old_lp_act)
            surr1_act = ratio_act * b_adv
            surr2_act = torch.clamp(ratio_act, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            loss_act = -torch.min(surr1_act, surr2_act).mean()

            # PPO Ratio Clipping for Discrete Consume Type Policy
            ratio_ct = torch.exp(eval_out["log_prob_ct"] - b_old_lp_ct)
            surr1_ct = ratio_ct * b_adv
            surr2_ct = torch.clamp(ratio_ct, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            loss_ct = -torch.min(surr1_ct, surr2_ct).mean()

            # Critic Value Loss (MSE against discounted GAE returns)
            loss_val = val_coeff * F.mse_loss(eval_out["v_pred"], b_ret)

            # Entropy Exploration Bonus
            entropy_total = eval_out["entropy_act"].mean() + eval_out["entropy_ct"].mean()
            loss_entropy = -entropy_coeff * entropy_total

            # Continuous Control Guidance
            loss_camt = 0.5 * F.mse_loss(eval_out["consume_amt"], b_camts)
            loss_damt = 0.5 * F.mse_loss(eval_out["drop_amt"], b_damts)

            # Behavioral Cloning (BC) KL Regularization
            # KL[pi_current || pi_ref] — keeps the policy close to the pre-trained expert
            loss_bc = torch.tensor(0.0)
            if ref_brain is not None and bc_coeff > 0.0:
                with torch.no_grad():
                    ref_out = ref_brain(b_states)
                cur_out = global_brain(b_states)
                kl_act = F.kl_div(
                    F.log_softmax(cur_out["logits_action"], dim=-1),
                    F.softmax(ref_out["logits_action"], dim=-1),
                    reduction="batchmean"
                )
                kl_ct = F.kl_div(
                    F.log_softmax(cur_out["logits_consume_type"], dim=-1),
                    F.softmax(ref_out["logits_consume_type"], dim=-1),
                    reduction="batchmean"
                )
                loss_bc = bc_coeff * (kl_act + kl_ct)

            total_loss = loss_act + loss_ct + loss_val + loss_entropy + loss_camt + loss_damt + loss_bc

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(global_brain.parameters(), max_grad_norm)
            optimizer.step()

            total_loss_sum += total_loss.item()

    buffer.reset()
    return {"ppo_loss": total_loss_sum}


def run_simulation_step(
    agents: List[Agent],
    environment_grid: np.ndarray,
    global_brain: MultiHeadGlobalBrain,
    config: EnvConfig = DEFAULT_CONFIG,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.0003,
    deterministic: bool = False,
    buffer: Optional[MAPPORolloutBuffer] = None
) -> Dict[str, Any]:
    """
    Executes a single step of the multi-agent simulation loop.
    Buffers transitions for MAPPO GAE batch updates or performs fallback online updates.
    """
    n_rows, n_cols, _ = environment_grid.shape
    barter_decisions = [False] * len(agents)

    # Check if any agents are alive
    if not any(a.alive for a in agents):
        return {"agents": agents, "global_brain": global_brain, "reset_needed": True}

    rew_cfg = getattr(config, 'rewards', None)
    survival_bonus = rew_cfg.survival_bonus if rew_cfg else 0.20
    death_penal = rew_cfg.death_penalty if rew_cfg else config.death_penal
    danger_thresh = rew_cfg.danger_threshold if rew_cfg else (12.0, 10.0)
    danger_scale = rew_cfg.danger_penalty_scale if rew_cfg else 0.04
    pd = getattr(config, 'progressive_drain', None)

    for i, agent in enumerate(agents):
        if not agent.alive:
            continue

        u_old = calculate_utility(agent, config)
        state_vec = extract_agent_observation(agent, environment_grid, config)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

        # Forward pass through neural brain
        with torch.no_grad():
            out = global_brain(state_tensor)

        logits_action = out["logits_action"].squeeze(0)
        dist_action = torch.distributions.Categorical(logits=logits_action)

        logits_barter = out["logits_barter"].squeeze(0)
        dist_barter = torch.distributions.Categorical(logits=logits_barter)

        logits_consume_type = out["logits_consume_type"].squeeze(0)
        dist_consume_type = torch.distributions.Categorical(logits=logits_consume_type)

        if deterministic:
            action_sample = torch.argmax(logits_action)
            barter_sample = torch.argmax(logits_barter)
            ctype_sample = torch.argmax(logits_consume_type)
        else:
            action_sample = dist_action.sample()
            barter_sample = dist_barter.sample()
            ctype_sample = dist_consume_type.sample()

        action_idx = int(action_sample.item()) + 1
        barter_flag = (barter_sample.item() == 1)
        consume_type = int(ctype_sample.item()) + 1

        log_prob_act = float(dist_action.log_prob(action_sample).item())
        log_prob_ct = float(dist_consume_type.log_prob(ctype_sample).item())

        drop_fraction = float(out["drop_amt"].squeeze().item())
        consume_fraction = float(out["consume_amt"].squeeze().item())
        v_pred = float(out["v_pred"].squeeze().item())

        barter_decisions[i] = barter_flag

        # Sequential Environment Execution
        agent = execute_move(agent, action_idx, n_rows, n_cols)
        agent = execute_gather(agent, action_idx, environment_grid, drop_fraction, config)
        agent = execute_consumption(agent, consume_type, consume_fraction, config)

        u_new = calculate_utility(agent, config)
        delta_u = u_new - u_old

        # Reward Shaping: Delta Utility + Survival Bonus - Danger Penalty - Overburden Penalty - Death Penalty
        reward = delta_u + survival_bonus

        # Smooth danger penalty when approaching starvation/cold
        if agent.n < danger_thresh[0]:
            reward -= danger_scale * (danger_thresh[0] - agent.n)
        if agent.h < danger_thresh[1]:
            reward -= danger_scale * (danger_thresh[1] - agent.h)

        # Overburden penalty
        current_weight = calculate_inventory_weight(agent.inv, config.weights)
        load_ratio = current_weight / max(1e-5, agent.max_storage)
        if pd and pd.enabled and load_ratio > pd.overburden_threshold:
            excess_load = (load_ratio - pd.overburden_threshold) / (1.0 - pd.overburden_threshold + 1e-5)
            overburden_pen = rew_cfg.overburden_penalty if rew_cfg and hasattr(rew_cfg, 'overburden_penalty') else 0.05
            reward -= overburden_pen * (excess_load ** 2)

        if not agent.alive:
            reward -= death_penal

        done = not agent.alive

        # Buffer transition for MAPPO GAE batch update
        if buffer is not None:
            buffer.store(
                state=state_vec,
                action=action_idx,
                log_prob_act=log_prob_act,
                consume_type=consume_type,
                log_prob_ct=log_prob_ct,
                consume_amt=consume_fraction,
                drop_amt=drop_fraction,
                reward=float(reward),
                done=done,
                value=v_pred
            )
        elif optimizer is not None:
            # Fallback 1-step update if no rollout buffer was supplied
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            train_out = global_brain(state_tensor)
            adv = float(reward) - train_out["v_pred"].squeeze(0)
            t_dist_act = torch.distributions.Categorical(logits=train_out["logits_action"].squeeze(0))
            t_dist_ct = torch.distributions.Categorical(logits=train_out["logits_consume_type"].squeeze(0))
            loss = (
                -t_dist_act.log_prob(torch.tensor(action_idx - 1)) * adv.detach()
                - t_dist_ct.log_prob(torch.tensor(consume_type - 1)) * adv.detach()
                + 0.5 * (adv ** 2)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        agents[i] = agent

    # Execute Barter Subloop synchronously after all decisions
    agents = execute_barter_subloop(agents, barter_decisions, config)

    return {"agents": agents, "global_brain": global_brain, "reset_needed": False}


def run_curriculum_training(
    global_brain: MultiHeadGlobalBrain,
    config: EnvConfig = DEFAULT_CONFIG,
    lr: float = 0.0003
) -> MultiHeadGlobalBrain:
    """
    Executes Curriculum Learning with progressive difficulty scaling on resource drain,
    buffered trajectory collection, and MAPPO PPO updates.
    """
    steps_each = config.pre_training.steps_each
    prog_step = config.pre_training.progression

    multipliers = np.arange(0.0, 1.0, prog_step)
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

    mp = getattr(config, 'mappo', None)
    rollout_steps = mp.rollout_steps if mp else 64
    lr_use = mp.lr if mp else lr
    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr_use)
    buffer = MAPPORolloutBuffer(capacity=rollout_steps * n_agents * 2)

    dead_lifespans = []

    for step in range(1, total_steps + 1):
        stage_idx = min(int(np.ceil(step / steps_each)) - 1, num_stages - 1)
        current_multiplier = float(multipliers[stage_idx])

        current_config = copy.deepcopy(config)
        current_config.decrease = (
            config.decrease[0] * current_multiplier,
            config.decrease[1] * current_multiplier
        )
        if getattr(current_config, 'progressive_drain', None):
            current_config.progressive_drain.scale = (
                config.progressive_drain.scale[0] * current_multiplier,
                config.progressive_drain.scale[1] * current_multiplier
            )

        stage_name = f"Stage {stage_idx + 1} (Drain: {int(current_multiplier * 100)}%)"

        for agent in agents:
            if agent.alive:
                agent.survival_steps += 1

        sim_output = run_simulation_step(agents, env_grid, global_brain, current_config, buffer=buffer)

        # Periodic MAPPO PPO mini-batch update
        if step % rollout_steps == 0 and len(buffer.rewards) > 0:
            last_val = 0.0
            if any(a.alive for a in agents):
                alive_agent = next(a for a in agents if a.alive)
                s_vec = extract_agent_observation(alive_agent, env_grid, current_config)
                with torch.no_grad():
                    last_val = float(global_brain(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0))["v_pred"].squeeze().item())
            update_mappo_policy(global_brain, optimizer, buffer, last_val, current_config)

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
            alive_count = sum(1 for a in agents if a.alive)
            print(f"Step {step:05d} [{stage_name}] | Alive: {alive_count}/{n_agents} | Mean Utility: {np.mean(utilities):.2f} | Mean Lifespan: {mean_lifespan:.1f} steps")

    print("-> Curriculum Training successfully completed!")
    return global_brain


def run_training_experiment(
    total_steps: int = 20000,
    x_steps: int = 2500,
    config: EnvConfig = DEFAULT_CONFIG,
    pre_trained_brain: Optional[MultiHeadGlobalBrain] = None,
    lr: float = 0.0003
) -> MultiHeadGlobalBrain:
    """
    Runs main MAPPO multi-agent reinforcement learning experiment loop with GAE & PPO mini-batch updates.

    Training is partitioned into fixed-length *epochs* of ``config.mappo.epoch_max`` steps each.
    At every epoch boundary:
      1. Any remaining buffer transitions are flushed with a MAPPO update so the network
         learns from the entire epoch's experience before moving on.
      2. A qualitative performance assessment is printed (GOOD / OK / POOR) based on
         rolling mean utility and mean episode lifespan inside the epoch, giving clear
         per-epoch credit-assignment feedback.
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
        # Keep a frozen reference copy for Behavioral Cloning (BC) regularization
        ref_brain = copy.deepcopy(pre_trained_brain)
        ref_brain.eval()
        for p in ref_brain.parameters():
            p.requires_grad_(False)
    else:
        print("-> Initializing new random Global Brain...")
        global_brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64)
        ref_brain = None

    agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)

    mp = getattr(config, 'mappo', None)
    rollout_steps = mp.rollout_steps if mp else 64
    lr_use = mp.lr if mp else lr
    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr_use)
    buffer = MAPPORolloutBuffer(capacity=rollout_steps * n_agents * 2)

    # --- Epoch bookkeeping ---
    epoch_max   = mp.epoch_max if mp else 500        # Hard step cap per epoch
    eval_window = mp.epoch_eval_window if mp else 50 # Rolling window for epoch assessment

    # --- BC regularization setup ---
    current_bc_coeff = mp.bc_coeff if mp else 0.30
    bc_decay         = mp.bc_coeff_decay if mp else 0.80
    bc_min           = mp.bc_coeff_min if mp else 0.005
    if ref_brain is not None:
        print(f"-> Behavioral Cloning regularization enabled (bc_coeff={current_bc_coeff:.3f}, decay={bc_decay}, floor={bc_min})")
    else:
        current_bc_coeff = 0.0

    epoch      = 1
    epoch_step = 0   # Steps elapsed within the current epoch

    episode            = 1
    episode_start_step = 1
    episode_lengths: List[int]   = []
    interval_lifespans: List[int] = []

    # Rolling metrics collected within each epoch
    window_utilities: List[float] = []
    window_lifespans: List[int]   = []

    print(f"-> Starting MAPPO RL Training with {total_steps} steps "
          f"(Rollout: {rollout_steps}, epoch_max: {epoch_max}, "
          f"GAE gamma: {mp.gamma if mp else 0.99}, PPO clip: {mp.clip_epsilon if mp else 0.2})...")

    for step in range(1, total_steps + 1):
        epoch_step += 1

        for agent in agents:
            if agent.alive:
                agent.survival_steps += 1

        sim_output = run_simulation_step(agents, env_grid, global_brain, config, buffer=buffer)

        # --- Periodic MAPPO rollout update (every rollout_steps within the current epoch) ---
        if epoch_step % rollout_steps == 0 and len(buffer.rewards) > 0:
            last_val = 0.0
            if any(a.alive for a in agents):
                alive_agent = next(a for a in agents if a.alive)
                s_vec = extract_agent_observation(alive_agent, env_grid, config)
                with torch.no_grad():
                    last_val = float(global_brain(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0))["v_pred"].squeeze().item())
            update_mappo_policy(global_brain, optimizer, buffer, last_val, config,
                                ref_brain=ref_brain, bc_coeff=current_bc_coeff)

        # --- Collect rolling metrics for epoch-end assessment ---
        current_utilities = [calculate_utility(a, config) for a in agents]
        window_utilities.append(float(np.mean(current_utilities)))
        if len(window_utilities) > eval_window:
            window_utilities.pop(0)

        # --- Episode reset handling ---
        if sim_output["reset_needed"]:
            ep_length = step - episode_start_step + 1
            episode_lengths.append(ep_length)
            interval_lifespans.append(ep_length)
            window_lifespans.append(ep_length)
            if len(window_lifespans) > eval_window:
                window_lifespans.pop(0)

            print(f"Step {step} | Episode {episode} ended naturally (all agents dead) after {ep_length} steps.")

            episode += 1
            episode_start_step = step + 1
            agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)
        else:
            agents = sim_output["agents"]
            global_brain = sim_output["global_brain"]

        # =====================================================================
        # EPOCH BOUNDARY: fires every epoch_max steps
        # =====================================================================
        if epoch_step >= epoch_max:
            # 1. Flush any remaining buffer so network learns from the full epoch
            if len(buffer.rewards) > 0:
                last_val = 0.0
                if any(a.alive for a in agents):
                    alive_agent = next(a for a in agents if a.alive)
                    s_vec = extract_agent_observation(alive_agent, env_grid, config)
                    with torch.no_grad():
                        last_val = float(global_brain(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0))["v_pred"].squeeze().item())
                update_mappo_policy(global_brain, optimizer, buffer, last_val, config,
                                    ref_brain=ref_brain, bc_coeff=current_bc_coeff)

            # 2. Epoch-end performance assessment
            mean_utility  = float(np.mean(window_utilities)) if window_utilities else 0.0
            mean_lifespan = float(np.mean(window_lifespans)) if window_lifespans \
                            else float(np.mean([a.survival_steps for a in agents]))
            alive_count   = sum(1 for a in agents if a.alive)
            total_eps     = len(window_lifespans)

            if mean_utility >= 20.0 and mean_lifespan >= 150.0:
                verdict = "GOOD  -- agents thriving, policy improving"
            elif mean_utility >= 8.0 or mean_lifespan >= 80.0:
                verdict = "OK    -- moderate performance, keep training"
            else:
                verdict = "POOR  -- agents collapsing, policy needs revision"

            epoch_start = step - epoch_max + 1
            nat_eps = total_eps   # episodes that ended naturally within this epoch
            print("=" * 60)
            print(f"  EPOCH {epoch:03d} COMPLETE  (Steps {epoch_start} to {step})")
            print(f"  Verdict             : {verdict}")
            print(f"  Mean Utility        : {mean_utility:.2f}  (last {eval_window} steps)")
            print(f"  Avg Episode Length  : {mean_lifespan:.1f} steps  ({nat_eps} natural endings)")
            print(f"  Living Agents       : {alive_count}/{n_agents}")
            print(f"  BC Coeff            : {current_bc_coeff:.4f}")
            print("=" * 60)

            # Decay BC coefficient at every epoch boundary (fades the pre-training anchor)
            if ref_brain is not None:
                current_bc_coeff = max(bc_min, current_bc_coeff * bc_decay)

            # 3. Hard simulation reset at epoch boundary
            #    Any still-alive agents are forcibly ended; their lifespan up to this point
            #    is recorded so the metric is not lost, then a fresh cohort is spawned.
            surviving_steps = step - episode_start_step + 1
            if any(a.alive for a in agents):
                # Episode was cut short by the epoch boundary — record the partial lifespan
                episode_lengths.append(surviving_steps)
                interval_lifespans.append(surviving_steps)
                print(f"  --> Epoch boundary after {epoch_max} steps "
                      f"({alive_count}/{n_agents} agents alive). "
                      f"Current episode cut at step {surviving_steps} — starting fresh.")
                episode += 1
            else:
                print(f"  --> Epoch boundary after {epoch_max} steps — no survivors. Starting fresh.")

            agents = initialize_agents_isolated(n_agents, n_rows, n_cols, config)
            episode_start_step = step + 1

            # Reset epoch-local counters and rolling windows
            epoch      += 1
            epoch_step  = 0
            window_utilities.clear()
            window_lifespans.clear()
            interval_lifespans.clear()

        # --- Standard interval report (runs only when NOT at an epoch boundary) ---
        elif step % x_steps == 0:
            avg_lifespan = float(np.mean(interval_lifespans)) if len(interval_lifespans) > 0 else 0.0
            utilities    = [calculate_utility(a, config) for a in agents]
            alive_count  = sum(1 for a in agents if a.alive)

            print("----------------------------------------------------")
            print(f">>> INTERVAL REPORT (Steps {step - x_steps + 1} - {step}) <<<")
            print(f"    - Average Episode Lifespan: {avg_lifespan:.2f} steps")
            print(f"    - Total Episodes in Interval: {len(interval_lifespans)}")
            print(f"    - Current Living Agents: {alive_count}/{n_agents}")
            print(f"    - Current Mean Utility: {np.mean(utilities):.2f}")
            print("----------------------------------------------------")

            interval_lifespans = []

        elif step % 100 == 0:
            utilities   = [calculate_utility(a, config) for a in agents]
            alive_count = sum(1 for a in agents if a.alive)
            print(f"Step {step:04d} | Epoch {epoch} ({epoch_step}/{epoch_max}) | "
                  f"Ep {episode} | Living: {alive_count}/{n_agents} | "
                  f"Mean Utility: {np.mean(utilities):.2f}")

    print("-> RL Training completed!")
    if len(episode_lengths) > 0:
        print(f"-> Overall Average Episode Lifespan across {len(episode_lengths)} episodes: {np.mean(episode_lengths):.2f} steps")

    return global_brain
