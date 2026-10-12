"""
MAPPO update (Yu et al. 2022): shared actor, centralised critic, GAE, clipped surrogate.

The unit of an RL step is one agent-round (t, b, n). All decisions an agent takes in a round (labour, up to two trade
decisions, consumption) form one composite action whose log-probability is the sum of its parts. Agent-rounds
without decisions carry no policy gradient but still train the critic. Agents that are dead at the start of a round
are masked out. Exchange economies end after the last round (terminal); production economies are truncated and the
value of the state after the last round is bootstrapped (effectively infinite horizon).
Minibatches are formed over whole worlds so that every agent-round is complete.
"""

from typing import Dict

import torch
import torch.nn as nn

from agents.policy import PolicyNet, CriticNet, Decision, LabourDecision, ConsumeDecision
from configs.base import PPOConfig
from economy.world import EconomyBatch
from learning.rollout import Rollout


def snapshot_inputs(world: EconomyBatch, rollout: Rollout, worlds: torch.Tensor, rounds: int):
    """Public/private tokens, adjacency and alive mask of the round-start snapshots X[:rounds, worlds], flattened."""
    W, N = worlds.numel(), world.N
    xs = rollout.X[:rounds, worlds].reshape(rounds * W, N, -1)
    bw = worlds.repeat(rounds)
    t = torch.arange(rounds).repeat_interleave(W)
    dep = rollout.deprivation[:rounds, worlds].reshape(rounds * W, N)
    alive = rollout.alive[:rounds, worlds].reshape(rounds * W, N)
    adj = world.adj[bw] & alive.unsqueeze(1) & alive.unsqueeze(2)
    return world.public_tokens(xs, bw), world.private_tokens(xs, bw, t, dep), adj, alive


def critic_values(critic: CriticNet, world: EconomyBatch, rollout: Rollout, worlds: torch.Tensor, rounds: int) -> torch.Tensor:
    _, priv, adj, alive = snapshot_inputs(world, rollout, worlds, rounds)
    return critic(priv, adj, alive).view(rounds, worlds.numel(), -1)


def gae(rewards, values, dones, bootstrap_values, gamma: float, lam: float):
    """GAE over rounds with per-agent termination (death); after the last round the bootstrap value is used."""
    R = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(R)):
        nonterminal = 1.0 - dones[t].float()
        next_v = values[t + 1] if t + 1 < R else bootstrap_values
        delta = rewards[t] + gamma * nonterminal * next_v - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def _evaluate_kind(policy: PolicyNet, kind: str, st: dict, sel, z, h, pw):
    rr, ss = st["r"][sel], st["s"][sel]
    zs = z[rr, pw, ss]
    if kind == "trade":
        actions = Decision(st["response"][sel], st["direction"][sel], st["amount"][sel], None, None)
        return policy.decide(zs, h[rr, pw, st["p"][sel]], st["tokens"][sel], st["context"][sel],
                             st["dir_mask"][sel], st["resp_mask"][sel], actions=actions)
    if kind == "labour":
        actions = LabourDecision(st["choice"][sel], st["amount"][sel], None, None)
        return policy.labour(zs, h[rr, pw, ss], st["tokens"][sel], st["context"][sel], st["mask"][sel], actions=actions)
    actions = ConsumeDecision(st["choice"][sel], st["amount"][sel], None, None)
    return policy.consume(zs, h[rr, pw, ss], st["tokens"][sel], st["context"][sel], st["mask"][sel], actions=actions)


def ppo_update(rollout: Rollout, world: EconomyBatch, policy: PolicyNet, critic: CriticNet,
               optimizer: torch.optim.Optimizer, cfg: PPOConfig) -> Dict[str, float]:
    rewards, dones = rollout.rewards, rollout.dones
    R, B, N = rewards.shape
    kinds = {k: v for k, v in (("trade", rollout.trade), ("labour", rollout.labour), ("consume", rollout.consume)) if v}
    valid = rollout.alive[:R]                                             # alive at the start of the round

    with torch.no_grad():
        all_values = critic_values(critic, world, rollout, torch.arange(B), R + 1 if rollout.bootstrap else R)
    values = all_values[:R]
    bootstrap = all_values[R] * rollout.alive[R].float() if rollout.bootstrap else torch.zeros(B, N)
    adv, returns = gae(rewards, values, dones, bootstrap, cfg.gamma, cfg.gae_lambda)

    old_lp = torch.zeros(R, B, N)
    acted = torch.zeros(R, B, N, dtype=torch.bool)
    for st in kinds.values():
        idx = (st["r"], st["b"], st["s"])
        old_lp = old_lp.index_put(idx, st["log_prob"], accumulate=True)
        acted = acted.index_put(idx, torch.ones_like(st["r"], dtype=torch.bool))
    acted &= valid
    if acted.any():
        a = adv[acted]
        adv = (adv - a.mean()) / (a.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_updates = 0
    params = list(policy.parameters()) + list(critic.parameters())
    for _ in range(cfg.epochs):
        for worlds in torch.randperm(B).chunk(cfg.minibatches):
            W = worlds.numel()
            pos = torch.full((B,), -1, dtype=torch.long)
            pos[worlds] = torch.arange(W)
            mask_v = valid[:, worlds]

            v = critic_values(critic, world, rollout, worlds, R)
            value_loss = ((v - returns[:, worlds]) ** 2)[mask_v].mean() if mask_v.any() else torch.zeros(())

            policy_loss, entropy = torch.zeros(()), torch.zeros(())
            mask = acted[:, worlds]
            if mask.any():
                pub, priv, adj, _ = snapshot_inputs(world, rollout, worlds, R)
                z, h = policy.embed(pub, priv, adj)
                z, h = z.view(R, W, N, -1), h.view(R, W, N, -1)
                new_lp = torch.zeros(R, W, N)
                entropies = []
                for kind, st in kinds.items():
                    sel = torch.nonzero(pos[st["b"]] >= 0).flatten()
                    if sel.numel() == 0:
                        continue
                    pw = pos[st["b"][sel]]
                    out = _evaluate_kind(policy, kind, st, sel, z, h, pw)
                    new_lp = new_lp.index_put((st["r"][sel], pw, st["s"][sel]), out.log_prob, accumulate=True)
                    entropies.append(out.entropy)
                log_ratio = new_lp[mask] - old_lp[:, worlds][mask]
                ratio = torch.exp(log_ratio)
                A = adv[:, worlds][mask]
                surrogate = torch.min(ratio * A, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * A)
                policy_loss = -surrogate.mean()
                entropy = torch.cat(entropies).mean()
                with torch.no_grad():
                    stats["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                    stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in stats.items()}
