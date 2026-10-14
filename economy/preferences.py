"""
Agent-specific preferences: sparse Dirichlet log utility with optional Stone-Geary subsistence.

    U_i(x) = sum_k alpha_ik * ln(1 + max(x_k - gamma_ik, 0))                 (default)
    U_i(x) = sum_k alpha_ik * ln(1 + x_k - gamma_ik)                         (continuous_subsistence, gamma < 1)

With the default, consumption below the need gives the same utility as consumption at the need (0), so skipping meals
is free until death. The continuous form extends the log below gamma: hunger costs utility, and by concavity steady
eating beats eating in lumps (no feast-and-fast). Both forms agree for goods without a need (gamma = 0).

alpha_i is non-zero only on a random support S_i of m_i consumable goods, so "agent i has no use for good k"
(alpha_ik = 0) is exact. Accepting such a good can only be motivated by passing it on (indirect exchange).
m_i is fixed (goods_valued) or drawn per agent from U{lo..hi} (goods_valued_range).

Optional tier weights: alpha_ik = d_ik * w_ik / sum_j d_ij * w_ij with d_i ~ Dirichlet(beta) on S_i and
w_ik = w_t * exp(s_t z_ik - s_t^2 / 2), z_ik ~ N(0, 1), t = tier(k): an agent- and good-specific multiplier with mean w_t
and log-spread s_t (s_t = 0: fixed). Basic goods can carry little utility while still being needed for subsistence
(gamma); higher tiers can be high-variance, high-utility goods. With min_processed_valued > 0
every agent values at least that many processed goods (tier >= 1), so higher-tier goods are always worth acquiring.
"""

from typing import Tuple

import torch

MIN_EXCESS = -0.99   # continuous subsistence: x - gamma is floored here (gamma < 1 keeps x = 0 above it)

from configs.base import PreferenceConfig
from economy.goods import GoodsSpace


def _setup(cfg: PreferenceConfig):
    lo, hi = cfg.goods_valued_range if cfg.goods_valued_range[1] > 0 else (cfg.goods_valued, cfg.goods_valued)
    if cfg.continuous_subsistence and cfg.subsistence_range[1] >= 1.0:
        raise ValueError("continuous_subsistence needs gamma < 1 (ln(1 + c - gamma) must be finite at c = 0)")
    if lo < (cfg.subsistence_goods if cfg.subsistence else 0) + cfg.min_processed_valued:
        raise ValueError("Every agent must value at least its subsistence goods and min_processed_valued goods")
    dirichlet = {m: torch.distributions.Dirichlet(torch.full((m,), float(cfg.dirichlet_beta))) for m in range(lo, hi + 1)}
    tier_weight = torch.tensor(cfg.tier_weights, dtype=torch.float) if cfg.tier_weights else None
    tier_sigma = torch.tensor(cfg.tier_weight_sigma, dtype=torch.float) if cfg.tier_weight_sigma else None
    return lo, hi, dirichlet, tier_weight, tier_sigma


def _candidates(cfg: PreferenceConfig, goods: GoodsSpace, b: int, hi: int):
    candidates = torch.nonzero(goods.consumable[b]).flatten()
    basic = candidates[goods.tier[b, candidates] <= cfg.subsistence_max_tier]
    if candidates.numel() < hi:
        raise ValueError(f"World {b} has {candidates.numel()} consumable goods, fewer than {hi} valued goods")
    if cfg.subsistence and basic.numel() < cfg.subsistence_goods:
        raise ValueError(f"World {b} has too few basic consumable goods for subsistence")
    return candidates, basic


def sample_preferences(cfg: PreferenceConfig, goods: GoodsSpace, n_agents: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns alpha (B, N, G) with rows summing to 1 and gamma (B, N, G) subsistence levels."""
    B, G = goods.tier.shape
    lo, hi, dirichlet, tier_weight, tier_sigma = _setup(cfg)
    alpha = torch.zeros(B, n_agents, G)
    gamma = torch.zeros(B, n_agents, G)
    for b in range(B):
        candidates, basic = _candidates(cfg, goods, b, hi)
        for i in range(n_agents):
            _draw_agent(cfg, goods, b, candidates, basic, lo, hi, dirichlet, tier_weight, tier_sigma, alpha[b, i], gamma[b, i])
    return alpha, gamma


def sample_agent_preferences(cfg: PreferenceConfig, goods: GoodsSpace, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fresh preferences (alpha (G,), gamma (G,)) for one newcomer in world b, drawn like everybody else's."""
    lo, hi, dirichlet, tier_weight, tier_sigma = _setup(cfg)
    candidates, basic = _candidates(cfg, goods, b, hi)
    G = goods.tier.shape[1]
    alpha, gamma = torch.zeros(G), torch.zeros(G)
    _draw_agent(cfg, goods, b, candidates, basic, lo, hi, dirichlet, tier_weight, tier_sigma, alpha, gamma)
    return alpha, gamma


def _draw_agent(cfg, goods, b, candidates, basic, lo, hi, dirichlet, tier_weight, tier_sigma, alpha_i, gamma_i) -> None:
    """Draws one agent's support, weights and needs into alpha_i / gamma_i (G,), in place."""
    m = lo if lo == hi else int(torch.randint(lo, hi + 1, (1,)))
    if cfg.subsistence:
        # subsistence goods are basic goods (tier <= subsistence_max_tier) and part of the support
        sub = basic[torch.randperm(basic.numel())[: cfg.subsistence_goods]]
        if cfg.min_processed_valued > 0:
            processed = candidates[(goods.tier[b, candidates] >= 1) & ~torch.isin(candidates, sub)]
            sub = torch.cat([sub, processed[torch.randperm(processed.numel())[: cfg.min_processed_valued]]])
        rest = candidates[~torch.isin(candidates, sub)]
        support = torch.cat([sub, rest[torch.randperm(rest.numel())[: m - sub.numel()]]])
        needs = sub[: cfg.subsistence_goods]
        gamma_i[needs] = torch.empty(needs.numel()).uniform_(*cfg.subsistence_range)
    else:
        support = candidates[torch.randperm(candidates.numel())[:m]]
    weights = dirichlet[m].sample()
    if tier_weight is not None:
        w = tier_weight[goods.tier[b, support].clamp(max=tier_weight.numel() - 1)]
        if tier_sigma is not None:
            sig = tier_sigma[goods.tier[b, support].clamp(max=tier_sigma.numel() - 1)]
            w = w * torch.exp(sig * torch.randn(m) - sig ** 2 / 2)   # mean-preserving log-normal noise
        weights = weights * w
        weights = weights / weights.sum()
    alpha_i[support] = weights


def excess(x: torch.Tensor, gamma: torch.Tensor, continuous: bool = False) -> torch.Tensor:
    """x - gamma, floored at 0 (default) or at MIN_EXCESS (continuous subsistence)."""
    return torch.clamp(x - gamma, min=MIN_EXCESS if continuous else 0.0)


def utility(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, continuous: bool = False) -> torch.Tensor:
    """U(x) summed over the last (goods) dimension."""
    return (alpha * torch.log1p(excess(x, gamma, continuous))).sum(-1)


def marginal_utility(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, continuous: bool = False) -> torch.Tensor:
    """dU/dx_k (default: right derivative at the subsistence kink)."""
    return alpha / (1.0 + excess(x, gamma, continuous))
