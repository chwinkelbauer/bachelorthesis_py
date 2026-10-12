"""
Training module featuring Supervised Pre-Training, Curriculum Learning, and MAPPO Reinforcement Learning.
Ported and modernized from simulation_loop.R, pre_train.R, and curriculum.R to PyTorch autograd.
Cleaned: Behavioral Cloning anchors removed, curriculum multiplier fixed to 100%.
"""

import copy
from networkx import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Any

from config import EnvConfig, DEFAULT_CONFIG
from environment import create_resource_map, extract_agent_observation, calculate_utility, calculate_inventory_weight
from agent import initialize_agents_isolated, execute_move, execute_gather, execute_consumption, execute_barter_subloop, get_action_mask, Agent
from network import MultiHeadGlobalBrain, CONSUME_AMT_EPS
from dataset import ExpertDataset
from metrics import EpisodeStats


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

    global_brain.train()
    print(f"-> Pre-training neural network on {len(dataset)} samples across operational heads...")

    for epoch in range(1, epochs + 1):
        total_epoch_loss = 0.0

        for batch in dataloader:
            states = batch["state"]
            actions = batch["action"]
            consume_types = batch["consume_type"]
            consume_amts = batch["consume_amt"].clamp(CONSUME_AMT_EPS, 1.0 - CONSUME_AMT_EPS)
            action_masks = batch["action_mask"]

            optimizer.zero_grad()
            out = global_brain(states, action_mask=action_masks)

            loss_act = ce_loss_action(out["logits_action"], actions)
            loss_ct = ce_loss_ctype(out["logits_consume_type"], consume_types)
            # Consume level only matters when something is consumed (consume type != None)
            ca_weight = (consume_types != 0).float()
            nll_ca = -global_brain.consume_distribution(out).log_prob(consume_amts)
            loss_ca = (nll_ca * ca_weight).sum() / ca_weight.sum().clamp(min=1.0)

            total_loss = loss_act + loss_ct + loss_ca
            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item() * len(states)

        avg_loss = total_epoch_loss / len(dataset)
        print(f"   - Epoch {epoch:02d}/{epochs:02d} | Pre-Training Loss: {avg_loss:.4f}")

    print("-> Supervised Pre-Training completed successfully!")
    return global_brain


class MAPPORolloutBuffer:
    """
    Trajectory Rollout Buffer with agent-separated GAE computation to preserve strict temporal causality.
    """
    def __init__(self, num_agents: int, capacity: int = 1000):
        self.num_agents = num_agents
        self.capacity = capacity
        self.reset()

    def reset(self):
        # Separierte Puffer pro Agent, um Cross-Agent Value Bootstrapping zu verhindern
        self.agent_buffers = {i: {
            'states': [], 'actions': [], 'log_prob_acts': [],
            'consume_types': [], 'log_prob_cts': [], 'consume_amts': [], 'log_prob_cas': [],
            'drop_amts': [], 'rewards': [], 'dones': [], 'values': [], 'action_masks': []
        } for i in range(self.num_agents)}

    def store(
        self,
        agent_idx: int,
        state: np.ndarray,
        action: int,
        log_prob_act: float,
        consume_type: int,
        log_prob_ct: float,
        consume_amt: float,
        log_prob_ca: float,
        drop_amt: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: np.ndarray
    ):
        buf = self.agent_buffers[agent_idx]
        buf['states'].append(state)
        buf['actions'].append(action)
        buf['log_prob_acts'].append(log_prob_act)
        buf['consume_types'].append(consume_type)
        buf['log_prob_cts'].append(log_prob_ct)
        buf['consume_amts'].append(consume_amt)
        buf['log_prob_cas'].append(log_prob_ca)
        buf['drop_amts'].append(drop_amt)
        buf['rewards'].append(reward)
        buf['dones'].append(done)
        buf['values'].append(value)
        buf['action_masks'].append(action_mask)

    def compute_returns_and_advantages(
        self,
        last_values: Optional[Dict[int, float]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, ...]:
        if last_values is None:
            last_values = {}

        all_states, all_actions, all_lp_acts, all_ctypes, all_lp_cts, all_camts, all_lp_cas, all_damts, all_advs, all_rets, all_masks = [], [], [], [], [], [], [], [], [], [], []

        for agent_idx, buf in self.agent_buffers.items():
            n = len(buf['rewards'])
            if n == 0:
                continue
            
            rewards = np.array(buf['rewards'], dtype=np.float32)
            values = np.array(buf['values'], dtype=np.float32)
            dones = np.array(buf['dones'], dtype=bool)
            
            advantages = np.zeros(n, dtype=np.float32)
            returns = np.zeros(n, dtype=np.float32)
            last_gae = 0.0
            last_val = last_values.get(agent_idx, 0.0)

            # GAE Berechnung strikt innerhalb der Zeitschritte desselben Agenten
            for t in reversed(range(n)):
                next_val = last_val if t == n - 1 else values[t + 1]
                non_terminal = 0.0 if dones[t] else 1.0
                delta = rewards[t] + gamma * next_val * non_terminal - values[t]
                last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
                advantages[t] = last_gae
                returns[t] = advantages[t] + values[t]

            all_states.extend(buf['states'])
            all_actions.extend(buf['actions'])
            all_lp_acts.extend(buf['log_prob_acts'])
            all_ctypes.extend(buf['consume_types'])
            all_lp_cts.extend(buf['log_prob_cts'])
            all_camts.extend(buf['consume_amts'])
            all_lp_cas.extend(buf['log_prob_cas'])
            all_damts.extend(buf['drop_amts'])
            all_advs.extend(advantages)
            all_rets.extend(returns)
            all_masks.extend(buf['action_masks'])

        if len(all_advs) == 0:
            return tuple(torch.tensor([], dtype=torch.float32) for _ in range(11))

        advs_arr = np.array(all_advs, dtype=np.float32)
        adv_mean = np.mean(advs_arr)
        adv_std = np.std(advs_arr) + 1e-8
        norm_advantages = (advs_arr - adv_mean) / adv_std

        return (
            torch.tensor(np.array(all_states), dtype=torch.float32),
            torch.tensor(np.array(all_actions), dtype=torch.long),
            torch.tensor(np.array(all_lp_acts), dtype=torch.float32),
            torch.tensor(np.array(all_ctypes), dtype=torch.long),
            torch.tensor(np.array(all_lp_cts), dtype=torch.float32),
            torch.tensor(np.array(all_camts), dtype=torch.float32),
            torch.tensor(np.array(all_lp_cas), dtype=torch.float32),
            torch.tensor(np.array(all_damts), dtype=torch.float32),
            torch.tensor(norm_advantages, dtype=torch.float32),
            torch.tensor(np.array(all_rets), dtype=torch.float32),
            torch.tensor(np.array(all_masks), dtype=torch.bool)
        )


def update_mappo_policy(
global_brain: MultiHeadGlobalBrain,
    optimizer: torch.optim.Optimizer,
    buffer: MAPPORolloutBuffer,
    last_values: Optional[Dict[int, float]] = None,
    config: EnvConfig = DEFAULT_CONFIG
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
    consume_entropy_coeff = mp.consume_entropy_coeff if mp else 0.001
    max_grad_norm = mp.max_grad_norm if mp else 0.50

    states, actions, old_lp_act, ctypes, old_lp_ct, camts, old_lp_ca, damts, advantages, returns, action_masks = (
        buffer.compute_returns_and_advantages(last_values=last_values, gamma=gamma, gae_lambda=gae_lambda)
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
            b_old_lp_ca = old_lp_ca[batch_idx]
            b_damts = damts[batch_idx]
            b_adv = advantages[batch_idx]
            b_ret = returns[batch_idx]
            b_masks = action_masks[batch_idx]

            eval_out = global_brain.evaluate_actions_batch(b_states, b_actions, b_ctypes, b_camts, b_damts, b_masks)

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

            # PPO Ratio Clipping for Continuous Beta Consume Amount Policy (only where something was consumed)
            ca_weight = (b_ctypes != 1).float()
            ca_norm = ca_weight.sum().clamp(min=1.0)
            ratio_ca = torch.exp(eval_out["log_prob_ca"] - b_old_lp_ca)
            surr1_ca = ratio_ca * b_adv
            surr2_ca = torch.clamp(ratio_ca, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            loss_ca = -(torch.min(surr1_ca, surr2_ca) * ca_weight).sum() / ca_norm

            # Critic Value Loss (MSE against discounted GAE returns)
            loss_val = val_coeff * F.mse_loss(eval_out["v_pred"], b_ret)

            # Entropy Exploration Bonus
            entropy_ca = (eval_out["entropy_ca"] * ca_weight).sum() / ca_norm
            entropy_total = eval_out["entropy_act"].mean() + eval_out["entropy_ct"].mean()
            loss_entropy = -entropy_coeff * entropy_total - consume_entropy_coeff * entropy_ca

            # Continuous Control Guidance
            loss_damt = 0.5 * F.mse_loss(eval_out["drop_amt"], b_damts)

            total_loss = loss_act + loss_ct + loss_ca + loss_val + loss_entropy + loss_damt

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
        state_vec = extract_agent_observation(agent, environment_grid, config, agents)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        action_mask = get_action_mask(agent, n_rows, n_cols, agents, config.mask_occupied_tiles)
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool)

        with torch.no_grad():
            out = global_brain(state_tensor, action_mask=mask_tensor)

        logits_action = out["logits_action"].squeeze(0)
        dist_action = torch.distributions.Categorical(logits=logits_action)

        logits_barter = out["logits_barter"].squeeze(0)
        dist_barter = torch.distributions.Categorical(logits=logits_barter)

        logits_consume_type = out["logits_consume_type"].squeeze(0)
        dist_consume_type = torch.distributions.Categorical(logits=logits_consume_type)

        dist_consume_amt = global_brain.consume_distribution(out)

        if deterministic:
            action_sample = torch.argmax(logits_action)
            barter_sample = torch.argmax(logits_barter)
            ctype_sample = torch.argmax(logits_consume_type)
            camt_sample = dist_consume_amt.mean
        else:
            action_sample = dist_action.sample()
            barter_sample = dist_barter.sample()
            ctype_sample = dist_consume_type.sample()
            camt_sample = dist_consume_amt.sample()

        action_idx = int(action_sample.item()) + 1
        barter_flag = (barter_sample.item() == 1)
        consume_type = int(ctype_sample.item()) + 1

        log_prob_act = float(dist_action.log_prob(action_sample).item())
        log_prob_ct = float(dist_consume_type.log_prob(ctype_sample).item())
        camt_sample = camt_sample.clamp(CONSUME_AMT_EPS, 1.0 - CONSUME_AMT_EPS)
        log_prob_ca = float(dist_consume_amt.log_prob(camt_sample).item())

        drop_fraction = float(out["drop_amt"].squeeze().item())
        consume_fraction = float(camt_sample.item())
        v_pred = float(out["v_pred"].squeeze().item())

        barter_decisions[i] = barter_flag

        agent = execute_move(agent, action_idx, n_rows, n_cols)
        agent = execute_gather(agent, action_idx, environment_grid, drop_fraction, config)
        agent = execute_consumption(agent, consume_type, consume_fraction, config)

        u_new = calculate_utility(agent, config)
        delta_u = u_new - u_old

        reward = delta_u + survival_bonus

        if agent.n < danger_thresh[0]:
            reward -= danger_scale * (danger_thresh[0] - agent.n)
        if agent.h < danger_thresh[1]:
            reward -= danger_scale * (danger_thresh[1] - agent.h)

        current_weight = calculate_inventory_weight(agent.inv, config.weights)
        load_ratio = current_weight / max(1e-5, agent.max_storage)
        if pd and pd.enabled and load_ratio > pd.overburden_threshold:
            excess_load = (load_ratio - pd.overburden_threshold) / (1.0 - pd.overburden_threshold + 1e-5)
            overburden_pen = rew_cfg.overburden_penalty if rew_cfg and hasattr(rew_cfg, 'overburden_penalty') else 0.05
            reward -= overburden_pen * (excess_load ** 2)

        if not agent.alive:
            reward -= death_penal

        done = not agent.alive

        if buffer is not None:
            buffer.store(
                agent_idx=i,  # Übergabe des Agenten-Index zur Kausalitätstrennung
                state=state_vec,
                action=action_idx,
                log_prob_act=log_prob_act,
                consume_type=consume_type,
                log_prob_ct=log_prob_ct,
                consume_amt=consume_fraction,
                log_prob_ca=log_prob_ca,
                drop_amt=drop_fraction,
                reward=float(reward),
                done=done,
                value=v_pred,
                action_mask=action_mask
            )
        elif optimizer is not None:
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            train_out = global_brain(state_tensor, action_mask=mask_tensor)
            adv = float(reward) - train_out["v_pred"].squeeze(0)
            t_dist_act = torch.distributions.Categorical(logits=train_out["logits_action"].squeeze(0))
            t_dist_ct = torch.distributions.Categorical(logits=train_out["logits_consume_type"].squeeze(0))
            t_dist_ca = global_brain.consume_distribution(train_out)
            ca_term = t_dist_ca.log_prob(torch.tensor([consume_fraction])).squeeze(0) if consume_type != 1 else 0.0
            loss = (
                -t_dist_act.log_prob(torch.tensor(action_idx - 1)) * adv.detach()
                - t_dist_ct.log_prob(torch.tensor(consume_type - 1)) * adv.detach()
                - ca_term * adv.detach()
                + 0.5 * (adv ** 2)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        agents[i] = agent

    agents = execute_barter_subloop(agents, barter_decisions, config)

    return {"agents": agents, "global_brain": global_brain, "reset_needed": False}


def _new_episode(config: EnvConfig) -> Tuple[np.ndarray, List[Agent]]:
    """Generates a fresh resource map and a fresh agent population."""
    env_grid = create_resource_map(
        n=config.grid.n_grid, k=config.grid.k_grid,
        num_food=config.grid.n_food,
        num_wood=config.grid.n_wood,
        num_gold=config.grid.n_gold,
        decay_rate=config.grid.decay_rate,
        min_eff=config.grid.min_eff
    )
    agents = initialize_agents_isolated(config.grid.n_agents, config.grid.n_grid, config.grid.k_grid, config)
    return env_grid, agents


def _bootstrap_values(
    agents: List[Agent],
    env_grid: np.ndarray,
    global_brain: MultiHeadGlobalBrain,
    config: EnvConfig
) -> Dict[int, float]:
    """Critic values V(s) of the current agent states used to bootstrap GAE (0.0 for dead agents)."""
    last_values = {}
    for idx, a in enumerate(agents):
        if a.alive:
            s_vec = extract_agent_observation(a, env_grid, config, agents)
            with torch.no_grad():
                last_values[idx] = float(global_brain(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0))["v_pred"].squeeze().item())
        else:
            last_values[idx] = 0.0
    return last_values


def run_curriculum_training(
    global_brain: MultiHeadGlobalBrain,
    config: EnvConfig = DEFAULT_CONFIG,
    lr: float = 0.0003,
    full_drain_steps: int = 0,
    include_ramp: bool = True
) -> MultiHeadGlobalBrain:
    """
    Executes Curriculum Learning with progressive difficulty scaling on resource drain up to 100%.
    full_drain_steps: additional steps trained at 100% drain after the ramp.
    include_ramp: if False, the ramp is skipped and only the full_drain_steps at 100% are trained
                  (e.g. to continue training a loaded curriculum_brain.pt).
    """
    steps_each = config.pre_training.steps_each
    prog_step = config.pre_training.progression

    multipliers = np.linspace(0.0, 1.0, int(1.0 / prog_step) + 1)
    num_stages = len(multipliers)
    ramp_steps = num_stages * steps_each if include_ramp else 0
    total_steps = ramp_steps + full_drain_steps

    if include_ramp:
        print(f"-> Entering Curriculum Learning ({num_stages} stages, {steps_each} steps each, "
              f"+ {full_drain_steps} extra steps at 100% drain, Total: {total_steps} steps)...")
    else:
        print(f"-> Entering Full-Drain Training (100% drain, {full_drain_steps} steps, curriculum ramp skipped)...")

    n_agents = config.grid.n_agents
    episode_max = config.pre_training.episode_max

    env_grid, agents = _new_episode(config)

    mp = getattr(config, 'mappo', None)
    rollout_steps = mp.rollout_steps if mp else 64
    lr_use = mp.lr if mp else lr
    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr_use)

    # Korrekte Initialisierung mit num_agents
    buffer = MAPPORolloutBuffer(num_agents=n_agents, capacity=rollout_steps * n_agents * 2)

    def buffer_has_data() -> bool:
        return sum(len(b['rewards']) for b in buffer.agent_buffers.values()) > 0

    # Survival statistics since the last report (finished episodes only)
    episode_len = 0
    report = EpisodeStats(config.dead_time)
    prev_stage_key = None
    current_config = config

    def end_episode(truncated: bool, count: bool = True):
        """Ends the current episode. count=False: episode cut short by a stage change (not reported)."""
        nonlocal env_grid, agents, episode_len
        if count:
            report.record(agents, truncated)
        if truncated and buffer_has_data():
            # Bootstrap the truncated trajectories before the population is replaced
            update_mappo_policy(global_brain, optimizer, buffer, _bootstrap_values(agents, env_grid, global_brain, current_config), current_config)
        env_grid, agents = _new_episode(current_config)
        episode_len = 0

    for step in range(1, total_steps + 1):
        if step <= ramp_steps:
            stage_idx = min(int(np.ceil(step / steps_each)) - 1, num_stages - 1)
            current_multiplier = float(multipliers[stage_idx])
            stage_name = f"Stage {stage_idx + 1} (Drain: {int(current_multiplier * 100)}%)"
            stage_key = stage_idx
        else:
            current_multiplier = 1.0
            stage_name = f"Full Drain {step - ramp_steps}/{full_drain_steps} (Drain: 100%)"
            stage_key = "full"

        # Every stage starts from fresh agents so each drain level is trained from the true start state
        if prev_stage_key is not None and stage_key != prev_stage_key and episode_len > 0:
            end_episode(truncated=True, count=False)
        prev_stage_key = stage_key

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

        for agent in agents:
            if agent.alive:
                agent.survival_steps += 1

        sim_output = run_simulation_step(agents, env_grid, global_brain, current_config, buffer=buffer)
        episode_len += 1

        # Überprüfung über den agenten-getrennten Buffer
        if step % rollout_steps == 0 and buffer_has_data():
            update_mappo_policy(global_brain, optimizer, buffer, _bootstrap_values(agents, env_grid, global_brain, current_config), current_config)

        if sim_output["reset_needed"]:
            end_episode(truncated=False)
        else:
            agents = sim_output["agents"]
            global_brain = sim_output["global_brain"]

        if not sim_output["reset_needed"] and episode_len >= episode_max:
            end_episode(truncated=True)

        if step % 500 == 0 or step == total_steps:
            print(f"Step {step:05d} [{stage_name}] | {report.summary()}")
            report.reset()

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
    Runs main MAPPO multi-agent reinforcement learning experiment loop directly with pre-trained brain without BC anchors.
    """
    n_agents = config.grid.n_agents
    env_grid, agents = _new_episode(config)

    if pre_trained_brain is not None:
        print("-> Using pre-trained Global Brain for training experiment (Direct continuation)...")
        global_brain = pre_trained_brain
    else:
        print("-> Initializing new random Global Brain...")
        global_brain = MultiHeadGlobalBrain(input_dim=config.input_dim, hidden_dim=64)

    mp = getattr(config, 'mappo', None)
    rollout_steps = mp.rollout_steps if mp else 64
    lr_use = mp.lr if mp else lr
    optimizer = torch.optim.Adam(global_brain.parameters(), lr=lr_use)

    # Korrekte Initialisierung mit num_agents
    buffer = MAPPORolloutBuffer(num_agents=n_agents, capacity=rollout_steps * n_agents * 2)

    def buffer_has_data() -> bool:
        return sum(len(b['rewards']) for b in buffer.agent_buffers.values()) > 0

    epoch_max = mp.epoch_max if mp else 500
    eval_window = mp.epoch_eval_window if mp else 50

    epoch = 1
    epoch_step = 0
    episode = 1
    episode_len = 0

    epoch_stats = EpisodeStats(config.dead_time)      # Finished episodes of the current epoch
    interval_stats = EpisodeStats(config.dead_time)   # Finished episodes since the last interval report
    total_stats = EpisodeStats(config.dead_time)      # Finished episodes of the whole run
    window_utilities: List[float] = []

    def end_episode(truncated: bool):
        nonlocal env_grid, agents, episode, episode_len
        for stats in (epoch_stats, interval_stats, total_stats):
            stats.record(agents, truncated)
        episode += 1
        episode_len = 0
        env_grid, agents = _new_episode(config)

    print(f"-> Starting MAPPO RL Training with {total_steps} steps "
          f"(Rollout: {rollout_steps}, epoch_max: {epoch_max}, "
          f"GAE gamma: {mp.gamma if mp else 0.99}, PPO clip: {mp.clip_epsilon if mp else 0.2})...")

    for step in range(1, total_steps + 1):
        epoch_step += 1

        for agent in agents:
            if agent.alive:
                agent.survival_steps += 1

        sim_output = run_simulation_step(agents, env_grid, global_brain, config, buffer=buffer)
        episode_len += 1

        # Rollout Update mit agenten-getrennten last_values
        if epoch_step % rollout_steps == 0 and buffer_has_data():
            update_mappo_policy(global_brain, optimizer, buffer, _bootstrap_values(agents, env_grid, global_brain, config), config)

        if sim_output["reset_needed"]:
            print(f"Step {step} | Episode {episode} ended naturally (all agents dead) after {episode_len} steps.")
            end_episode(truncated=False)
        else:
            agents = sim_output["agents"]
            global_brain = sim_output["global_brain"]

        current_utilities = [calculate_utility(a, config) for a in agents]
        window_utilities.append(float(np.mean(current_utilities)))
        if len(window_utilities) > eval_window:
            window_utilities.pop(0)

        if epoch_step >= epoch_max:
            if buffer_has_data():
                update_mappo_policy(global_brain, optimizer, buffer, _bootstrap_values(agents, env_grid, global_brain, config), config)
            if episode_len > 0:
                end_episode(truncated=True)

            survival = epoch_stats.survival_rate
            if survival >= 0.9:
                verdict = "GOOD  -- (almost) all agents survive the episode"
            elif survival >= 0.5:
                verdict = "OK    -- most agents survive, keep training"
            else:
                verdict = "POOR  -- most agents die, policy needs revision"

            print("=" * 60)
            print(f"  EPOCH {epoch:03d} COMPLETE  (Steps {step - epoch_max + 1} to {step})")
            print(f"  Verdict             : {verdict}")
            print(f"  Mean Utility        : {np.mean(window_utilities):.2f}  (last {eval_window} steps)")
            print(f"  Survival            : {epoch_stats.summary()}")
            print("=" * 60)

            epoch += 1
            epoch_step = 0
            window_utilities.clear()
            epoch_stats.reset()

        elif step % x_steps == 0:
            print("----------------------------------------------------")
            print(f">>> INTERVAL REPORT (Steps {step - x_steps + 1} - {step}) <<<")
            print(f"    - {interval_stats.summary()}")
            print(f"    - Current Living Agents: {sum(1 for a in agents if a.alive)}/{n_agents}")
            print("----------------------------------------------------")
            interval_stats.reset()

        elif step % 100 == 0:
            utilities = [calculate_utility(a, config) for a in agents]
            alive_count = sum(1 for a in agents if a.alive)
            print(f"Step {step:04d} | Epoch {epoch} ({epoch_step}/{epoch_max}) | "
                  f"Ep {episode} | Living: {alive_count}/{n_agents} | "
                  f"Mean Utility: {np.mean(utilities):.2f}")

    print("-> RL Training completed!")
    print(f"-> Overall: {total_stats.summary()}")

    return global_brain
