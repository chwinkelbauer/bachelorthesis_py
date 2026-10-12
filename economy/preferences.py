"""
Agent-specific preferences: sparse Dirichlet log utility with optional Stone-Geary subsistence.

    U_i(x) = sum_k alpha_ik * ln(1 + max(x_k - gamma_ik, 0))

alpha_i is non-zero only on a random support S_i of m consumable goods, so "agent i has no use for good k"
(alpha_ik = 0) is exact. Accepting such a good can only be motivated by passing it on (indirect exchange).
"""

from typing import Tuple

import torch

from configs.base import PreferenceConfig
from economy.goods import GoodsSpace


def sample_preferences(cfg: PreferenceConfig, goods: GoodsSpace, n_agents: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns alpha (B, N, G) with rows summing to 1 and gamma (B, N, G) subsistence levels."""
    B, G = goods.tier.shape
    m = cfg.goods_valued
    alpha = torch.zeros(B, n_agents, G)
    gamma = torch.zeros(B, n_agents, G)
    dirichlet = torch.distributions.Dirichlet(torch.full((m,), float(cfg.dirichlet_beta)))

    for b in range(B):
        candidates = torch.nonzero(goods.consumable[b]).flatten()
        basic = candidates[goods.tier[b, candidates] <= cfg.subsistence_max_tier]
        if candidates.numel() < m:
            raise ValueError(f"World {b} has {candidates.numel()} consumable goods, fewer than goods_valued={m}")
        if cfg.subsistence and basic.numel() < cfg.subsistence_goods:
            raise ValueError(f"World {b} has too few basic consumable goods for subsistence")
        for i in range(n_agents):
            if cfg.subsistence:
                # subsistence goods are basic goods (tier <= subsistence_max_tier) and part of the support
                sub = basic[torch.randperm(basic.numel())[: cfg.subsistence_goods]]
                rest = candidates[~torch.isin(candidates, sub)]
                support = torch.cat([sub, rest[torch.randperm(rest.numel())[: m - sub.numel()]]])
                gamma[b, i, sub] = torch.empty(sub.numel()).uniform_(*cfg.subsistence_range)
            else:
                support = candidates[torch.randperm(candidates.numel())[:m]]
            alpha[b, i, support] = dirichlet.sample()
    return alpha, gamma


def utility(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """U(x) summed over the last (goods) dimension."""
    return (alpha * torch.log1p(torch.clamp(x - gamma, min=0.0))).sum(-1)


def marginal_utility(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """dU/dx_k (right derivative at the subsistence kink)."""
    return alpha / (1.0 + torch.clamp(x - gamma, min=0.0))
