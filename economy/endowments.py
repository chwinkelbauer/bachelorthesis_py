"""
Stage A endowments: each agent receives a few random goods once at the start of the episode.
"""

import torch

from configs.base import EndowmentConfig
from economy.goods import GoodsSpace


def sample_endowments(cfg: EndowmentConfig, goods: GoodsSpace, n_agents: int, capacity: float) -> torch.Tensor:
    """Returns initial holdings x0 (B, N, G); each agent's endowment weighs at most 80% of capacity."""
    B, G = goods.tier.shape
    x = torch.zeros(B, n_agents, G)
    for b in range(B):
        pool = torch.nonzero(goods.consumable[b]).flatten() if cfg.consumable_only else torch.arange(G)
        k = min(cfg.goods_per_agent, pool.numel())
        for i in range(n_agents):
            chosen = pool[torch.randperm(pool.numel())[:k]]
            x[b, i, chosen] = torch.empty(k).uniform_(*cfg.amount_range)
    load = (x * goods.weight[:, None, :]).sum(-1, keepdim=True)
    scale = torch.clamp(0.8 * capacity / load.clamp(min=1e-9), max=1.0)
    return x * scale
