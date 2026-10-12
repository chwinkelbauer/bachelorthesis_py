"""
Random goods space: a production DAG of goods described only by property vectors.

No good carries a label such as "money" or "gold": monetary roles must emerge from properties
(weight, spoilage, tier, use value) and from the agents' learned behaviour.
"""

from dataclasses import dataclass

import torch

from configs.base import GoodsConfig

PROPERTY_FEATURES = 5  # log weight, spoilage, normalised tier, is_raw, consumable


@dataclass
class GoodsSpace:
    """Goods of B independent worlds (G goods each)."""
    tier: torch.Tensor        # (B, G) long, 0 = raw resource
    weight: torch.Tensor      # (B, G) weight per unit
    spoilage: torch.Tensor    # (B, G) fraction lost per round
    consumable: torch.Tensor  # (B, G) bool, good can carry use value
    recipe: torch.Tensor      # (B, G, G) recipe[b, out, in] = input units per output unit (0 for raw goods)

    @property
    def n_goods(self) -> int:
        return self.tier.shape[1]

    def features(self) -> torch.Tensor:
        """(B, G, PROPERTY_FEATURES) property vector of every good."""
        max_tier = max(int(self.tier.max().item()), 1)
        return torch.stack([
            torch.log(self.weight),
            self.spoilage,
            self.tier.float() / max_tier,
            (self.tier == 0).float(),
            self.consumable.float(),
        ], dim=-1)


def generate_goods(cfg: GoodsConfig, worlds: int) -> GoodsSpace:
    """Samples an independent random goods space for each world."""
    tiers = torch.cat([torch.full((n,), t, dtype=torch.long) for t, n in enumerate(cfg.goods_per_tier)])
    G = tiers.numel()
    tier = tiers.unsqueeze(0).expand(worlds, G).clone()

    lo, hi = cfg.weight_range
    weight = torch.exp(torch.empty(worlds, G).uniform_(float(torch.log(torch.tensor(lo))), float(torch.log(torch.tensor(hi)))))
    spoilage = torch.empty(worlds, G).uniform_(*cfg.spoilage_range)

    consumable = torch.ones(worlds, G, dtype=torch.bool)
    raw = tier == 0
    consumable[raw] = torch.rand(int(raw.sum())) < cfg.raw_consumable_prob

    recipe = torch.zeros(worlds, G, G)
    for out in range(G):
        t = int(tiers[out])
        if t == 0:
            continue
        lower = torch.nonzero(tiers < t).flatten()
        previous = torch.nonzero(tiers == t - 1).flatten()
        for b in range(worlds):
            k = int(torch.randint(cfg.recipe_inputs[0], cfg.recipe_inputs[1] + 1, (1,)))
            k = min(k, lower.numel())
            first = previous[torch.randint(previous.numel(), (1,))]          # at least one input from tier t-1
            rest = lower[lower != first]
            others = rest[torch.randperm(rest.numel())[: k - 1]]
            inputs = torch.cat([first, others])
            recipe[b, out, inputs] = torch.empty(inputs.numel()).uniform_(*cfg.recipe_coef_range)

    return GoodsSpace(tier=tier, weight=weight, spoilage=spoilage, consumable=consumable, recipe=recipe)
