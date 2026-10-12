"""
Production side of stages B-D.

Income (stage B): agent i earns raw goods at mean flow f_ik per round, paid as a lump of f_ik * P_i every P_i rounds
(arrival when (t - phase_i) mod P_i = 0). Every raw good has at least one earner, so every good is producible.

Processing (Leontief): one unit of good k uses a_kj units of every input j (recipe row a_k.). With holdings x the
maximal output is q_max,k = min_{j: a_kj > 0} x_j / a_kj; raw goods cannot be processed.

Feasibility: raw_requirements gives the raw units embodied in one unit of every good (recursive Leontief). Incomes
are scaled so that, per raw good, the income flow covers `subsistence_surplus` times the raw units needed for
everyone's subsistence (otherwise deaths would reflect built-in scarcity rather than behaviour).

Harvesting (stage C): agent i can harvest raw good r if it has access (access_goods_per_agent raw goods, every raw
good accessible to someone); one round of harvesting r yields y_ir = base_r * skill_ir with skill_ir ~ LogNormal(0, s).
base_r >= 1 is set so that, if every agent split its labour evenly over its accessible goods, the harvest of r would
cover `subsistence_surplus` times the aggregate subsistence requirement of r.

Usefulness: good j is useful to agent i if alpha_ij > 0 or j is (possibly indirectly) an input of a useful good.
Input value: v_ij = max over goods k using j directly of alpha_ik (direct downstream preference).
"""

from typing import Tuple

import torch

from configs.base import ProductionConfig
from economy.goods import GoodsSpace


def sample_income(cfg: ProductionConfig, goods: GoodsSpace, n_agents: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns lump amount per arrival (B, N, G), period (B, N) long, phase (B, N) long."""
    B, G = goods.tier.shape
    lo, hi = cfg.income_period_range
    period = torch.randint(lo, hi + 1, (B, n_agents))
    phase = (torch.rand(B, n_agents) * period).long()
    amount = torch.zeros(B, n_agents, G)
    for b in range(B):
        raw = torch.nonzero(goods.tier[b] == 0).flatten()
        k = min(cfg.income_goods_per_agent, raw.numel())
        earns = torch.zeros(n_agents, G, dtype=torch.bool)
        order = torch.randperm(n_agents)
        for t, r in enumerate(raw[torch.randperm(raw.numel())]):      # coverage: every raw good has an earner
            earns[order[t % n_agents], r] = True
        for i in range(n_agents):
            missing = k - int(earns[i].sum())
            if missing > 0:
                free = raw[~earns[i, raw]]
                earns[i, free[torch.randperm(free.numel())[:missing]]] = True
        flow = torch.empty(n_agents, G).uniform_(*cfg.income_flow_range)
        amount[b] = flow * earns * period[b].unsqueeze(-1)
    return amount, period, phase


def raw_requirements(recipe: torch.Tensor, tier: torch.Tensor) -> torch.Tensor:
    """(B, G, G): req[b, k, r] = units of raw good r embodied in one unit of good k (tier is identical across worlds)."""
    B, G, _ = recipe.shape
    req = torch.zeros(B, G, G)
    t0 = tier[0]
    for k in torch.nonzero(t0 == 0).flatten():
        req[:, k, k] = 1.0
    for t in range(1, int(t0.max()) + 1):
        for k in torch.nonzero(t0 == t).flatten():
            req[:, k] = (recipe[:, k].unsqueeze(-1) * req).sum(1)
    return req


def ensure_subsistence_feasible(income: torch.Tensor, period: torch.Tensor, gamma: torch.Tensor,
                                recipe: torch.Tensor, tier: torch.Tensor, surplus: float) -> torch.Tensor:
    """Scales income per world and raw good so that flow >= surplus * aggregate subsistence raw requirement."""
    req = raw_requirements(recipe, tier)
    need = (gamma.unsqueeze(-1) * req.unsqueeze(1)).sum((1, 2))          # (B, G) raw units per round
    flow = (income / period.unsqueeze(-1).float()).sum(1)                  # (B, G) raw units per round
    scale = torch.where(flow > 0, torch.clamp(surplus * need / flow.clamp(min=1e-12), min=1.0), torch.ones_like(flow))
    return income * scale.unsqueeze(1)


def sample_harvest(cfg: ProductionConfig, goods: GoodsSpace, gamma: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
    """Harvest yield per round (B, N, G); zero for goods an agent cannot harvest and for all non-raw goods."""
    B, N, G = gamma.shape
    access = torch.zeros(B, N, G, dtype=torch.bool)
    for b in range(B):
        raw = torch.nonzero(goods.tier[b] == 0).flatten()
        k = min(cfg.access_goods_per_agent, raw.numel())
        order = torch.randperm(N)
        for t, r in enumerate(raw[torch.randperm(raw.numel())]):      # coverage: every raw good is accessible
            access[b, order[t % N], r] = True
        for i in range(N):
            missing = k - int(access[b, i].sum())
            if missing > 0:
                free = raw[~access[b, i, raw]]
                access[b, i, free[torch.randperm(free.numel())[:missing]]] = True
    skill = torch.exp(cfg.skill_sigma * torch.randn(B, N, G)) * access
    req = raw_requirements(recipe, goods.tier)
    need = (gamma.unsqueeze(-1) * req.unsqueeze(1)).sum((1, 2))          # (B, G) raw units per round
    even_split = (skill / access.sum(-1, keepdim=True).clamp(min=1)).sum(1)   # harvest per round if labour is split evenly
    base = torch.clamp(cfg.subsistence_surplus * need / even_split.clamp(min=1e-12), min=1.0)
    return base.unsqueeze(1) * skill


def income_arrives(period: torch.Tensor, phase: torch.Tensor, t: int) -> torch.Tensor:
    """(B, N) bool: does the lump arrive in round t?"""
    return torch.remainder(t - phase, period) == 0


def rounds_to_next_income(period: torch.Tensor, phase: torch.Tensor, t) -> torch.Tensor:
    """Rounds from round t until the next arrival after t, in [1, P]."""
    return torch.remainder(phase - t - 1, period) + 1


def max_output(x: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
    """x (S, N, G), recipe (S, G, G) -> q_max (S, N, G); 0 for goods without recipe."""
    uses = recipe > 0
    ratio = x.unsqueeze(2) / recipe.unsqueeze(1).clamp(min=1e-12)
    ratio = torch.where(uses.unsqueeze(1), ratio, torch.full_like(ratio, float("inf")))
    q = ratio.amin(-1)
    return torch.where(uses.any(-1).unsqueeze(1), q, torch.zeros_like(q))


def usefulness(alpha: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
    """(B, N, G) bool: consumed by the agent or (transitively) needed for a good it consumes."""
    useful = alpha > 0
    uses = (recipe > 0).unsqueeze(1)                       # (B, 1, K, J): k uses j
    for _ in range(recipe.shape[-1]):
        via = (useful.unsqueeze(-1) & uses).any(2)        # useful output k that uses j
        updated = useful | via
        if torch.equal(updated, useful):
            break
        useful = updated
    return useful


def input_value(alpha: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
    """(B, N, G): v_ij = max_k alpha_ik over goods k that use j directly."""
    uses = (recipe > 0).unsqueeze(1).float()               # (B, 1, K, J)
    return (alpha.unsqueeze(-1) * uses).amax(2)


def capacity_limited_output(q: torch.Tensor, out_weight: torch.Tensor, input_weight: torch.Tensor,
                            load: torch.Tensor, capacity: float) -> torch.Tensor:
    """Caps output q so that the load after processing (output in, inputs out) stays within capacity."""
    net = out_weight - input_weight                        # weight change per unit of output
    room = torch.clamp(capacity - load, min=0.0)
    cap = torch.where(net > 0, room / net.clamp(min=1e-12), torch.full_like(q, float("inf")))
    return torch.minimum(q, cap)
