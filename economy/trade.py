"""
Offer representation and decoding.

An offer made by agent o to partner p is, per good k, a direction d_k in {NONE, GIVE, WANT} and an amount
a_k in (0, 1):
    give_k = a_k * x_o,k   if d_k = GIVE      (fraction of the offerer's own stock)
    want_k = a_k * x_p,k   if d_k = WANT      (fraction of the partner's stock)
Giving and wanting the same good is impossible by construction. The trade is scaled down uniformly
(preserving its exchange ratios) if it would push either side above its carrying capacity.
"""

from typing import Tuple

import torch

NONE, GIVE, WANT = 0, 1, 2
ACCEPT, COUNTER, REJECT = 0, 1, 2
N_RESPONSES = 3
N_DIRECTIONS = 3
EPS_STOCK = 1e-6


def direction_mask(x_self: torch.Tensor, x_partner: torch.Tensor) -> torch.Tensor:
    """(D, G, 3) bool: NONE always allowed, GIVE needs own stock, WANT needs partner stock."""
    mask = torch.ones(*x_self.shape, N_DIRECTIONS, dtype=torch.bool)
    mask[..., GIVE] = x_self > EPS_STOCK
    mask[..., WANT] = x_partner > EPS_STOCK
    return mask


def response_mask(turn: torch.Tensor) -> torch.Tensor:
    """
    (D, 3) bool. Turn 0 (opening): the only option is to make an offer (COUNTER).
    Turn 1 (responder): accept, counter-offer or reject. Turn 2 (final): accept or reject.
    """
    mask = torch.ones(turn.numel(), N_RESPONSES, dtype=torch.bool)
    mask[turn == 0, ACCEPT] = False
    mask[turn == 0, REJECT] = False
    mask[turn == 2, COUNTER] = False
    return mask


def decode_offer(
    direction: torch.Tensor, amount: torch.Tensor,
    x_self: torch.Tensor, x_partner: torch.Tensor,
    weight: torch.Tensor, capacity: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (give, want) in absolute units from the offerer's perspective, capacity-feasible."""
    give = torch.where(direction == GIVE, amount * x_self, torch.zeros_like(x_self))
    want = torch.where(direction == WANT, amount * x_partner, torch.zeros_like(x_partner))
    scale = capacity_scale(x_self, x_partner, give, want, weight, capacity)
    return give * scale[:, None], want * scale[:, None]


def capacity_scale(x_self, x_partner, give, want, weight, capacity) -> torch.Tensor:
    """Largest s in [0, 1] such that both sides stay within capacity after trading s * (give, want)."""
    scale = torch.ones(x_self.shape[0])
    for x, inflow, outflow in ((x_self, want, give), (x_partner, give, want)):
        load = (x * weight).sum(-1)
        net = ((inflow - outflow) * weight).sum(-1)
        over = net > 0
        room = torch.clamp(capacity - load, min=0.0)
        s = torch.where(over, room / net.clamp(min=1e-12), torch.ones_like(net))
        scale = torch.minimum(scale, s.clamp(max=1.0))
    return scale
