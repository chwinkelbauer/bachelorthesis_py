"""
Theoretical benchmarks for the pure exchange economy with U_i(x) = sum_k alpha_ik ln(1 + x_k) (gamma = 0).

Walrasian equilibrium: individual demand solves alpha_k / (1 + x_k) = lambda p_k for x_k > 0, i.e.
    x_k(lambda) = max(0, alpha_k / (lambda p_k) - 1),   lambda chosen so that p . x = p . e (bisection).
The utility is additively separable with relative curvature x/(1+x) < 1, so aggregate demand satisfies gross
substitutes; tatonnement therefore converges to the unique equilibrium.

Utilitarian planner: maximise sum_i U_i subject to sum_i x_ik = X_k. Separable per good (water filling):
    x_ik = max(0, alpha_ik / mu_k - 1),   mu_k chosen so that the allocation sums to X_k.

Since spoilage is multiplicative and good-specific, aggregate supply at the end of an episode is
X_k (1 - delta_k)^R regardless of who held the good. Benchmarks are therefore computed on the spoiled
endowments e_i (1 - delta)^R, which is exactly the set of final allocations trade could reach.
"""

from typing import Dict

import torch

from economy.preferences import utility

BISECTION_STEPS = 60


def _bisect_log(fn, lo: torch.Tensor, hi: torch.Tensor, steps: int = BISECTION_STEPS) -> torch.Tensor:
    """Finds y in [lo, hi] (log space) with fn(exp(y)) = 0 for fn decreasing."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        positive = fn(torch.exp(mid)) > 0
        lo = torch.where(positive, mid, lo)
        hi = torch.where(positive, hi, mid)
    return torch.exp(0.5 * (lo + hi))


def demand(alpha: torch.Tensor, prices: torch.Tensor, wealth: torch.Tensor) -> torch.Tensor:
    """
    Exact demand, alpha (B, N, G), prices (B, G), wealth (B, N) -> (B, N, G). Goods with price 0 must have alpha 0.
    Sort goods by r_k = alpha_k / p_k (descending). On the active set A = top-M goods, the budget gives
    lambda = sum_A alpha / (m + sum_A p); M is the largest prefix with r_M > lambda_M.
    """
    p = prices.unsqueeze(1).expand_as(alpha)
    ratio = torch.where(p > 0, alpha / p.clamp(min=1e-30), torch.zeros_like(alpha))
    r_sorted, order = ratio.sort(-1, descending=True)
    lam_m = alpha.gather(-1, order).cumsum(-1) / (wealth.unsqueeze(-1) + p.gather(-1, order).cumsum(-1)).clamp(min=1e-30)
    active = (r_sorted > lam_m) & (r_sorted > 0)
    M = active.sum(-1, keepdim=True)
    lam = lam_m.gather(-1, (M - 1).clamp(min=0))
    x = torch.clamp(ratio / lam.clamp(min=1e-30) - 1.0, min=0.0)
    return torch.where((M > 0) & (wealth.unsqueeze(-1) > 0), x, torch.zeros_like(x))


def walras_equilibrium(alpha: torch.Tensor, endowment: torch.Tensor, iterations: int = 300, step: float = 0.3) -> Dict[str, torch.Tensor]:
    """Tatonnement with multiplicative price updates. Returns prices (mean 1 over traded goods), allocation, residual."""
    supply = endowment.sum(1)                                     # (B, G)
    traded = supply > 1e-9
    alpha = alpha * traded.unsqueeze(1)                           # goods with no supply cannot be demanded
    prices = traded.float().clone()
    for _ in range(iterations):
        wealth = (endowment * prices.unsqueeze(1)).sum(-1)
        z = (demand(alpha, prices, wealth).sum(1) - supply) / supply.clamp(min=1e-9)
        prices = prices * torch.exp(step * z.clamp(-1.0, 1.0)) * traded
        prices = prices / (prices.sum(-1, keepdim=True) / traded.sum(-1, keepdim=True).clamp(min=1)).clamp(min=1e-12)
        prices = prices.clamp(min=1e-8) * traded
    wealth = (endowment * prices.unsqueeze(1)).sum(-1)
    alloc = demand(alpha, prices.clamp(min=1e-8), wealth)
    residual = ((alloc.sum(1) - supply).abs() / supply.clamp(min=1e-9) * traded).amax(-1)
    return {"prices": prices, "allocation": alloc, "residual": residual}


def planner_allocation(alpha: torch.Tensor, endowment: torch.Tensor) -> torch.Tensor:
    """Utilitarian optimum (B, N, G)."""
    supply = endowment.sum(1)                                     # (B, G)
    hi = torch.log(alpha.amax(1).clamp(min=1e-12)) + 1e-6

    def excess(mu):
        return torch.clamp(alpha / mu.unsqueeze(1) - 1.0, min=0.0).sum(1) - supply

    mu = _bisect_log(excess, torch.full_like(hi, -30.0), hi)
    alloc = torch.clamp(alpha / mu.unsqueeze(1) - 1.0, min=0.0)
    return torch.where(supply.unsqueeze(1) > 1e-9, alloc, torch.zeros_like(alloc))


def welfare_benchmarks(alpha, gamma, x0, spoilage, rounds) -> Dict[str, torch.Tensor]:
    """Per-world utilitarian welfare (B,) of: no trade, Walrasian equilibrium, planner; plus Walras prices."""
    e = x0 * ((1.0 - spoilage) ** rounds).unsqueeze(1)
    we = walras_equilibrium(alpha, e)
    plan = planner_allocation(alpha, e)
    return {
        "endowment_end": e,
        "W_no_trade": utility(e, alpha, gamma).sum(-1),
        "W_walras": utility(we["allocation"], alpha, gamma).sum(-1),
        "W_planner": utility(plan, alpha, gamma).sum(-1),
        "prices": we["prices"],
        "walras_residual": we["residual"],
    }
