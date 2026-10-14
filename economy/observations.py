"""
Observation features. Goods are represented as tokens (one feature vector per good), so every network is
permutation-equivariant over goods and works for any number of goods.

The feature layout is identical in all stages (features a stage does not use are 0), so that weights can be carried
from stage to stage (curriculum warm start). Two token slots are reserved for the stage-D ledger.

Information structure:
  * public  : what neighbours can observe about an agent (holdings; preferences only under full information)
  * private : what an agent knows about itself (holdings, preferences, subsistence, income, production options)
  * decision: own private view + partner's public view (trade mode) + the standing offer in absolute units
The critic (centralised training) sees every agent's private view.
"""

from typing import Dict

import torch

from economy.goods import PROPERTY_FEATURES
from economy.preferences import utility, marginal_utility, excess

EXPERIENCE_FEATURES = 4                    # last round, per good: received, given, asked for by a partner, offered by a partner
MEMORY_INPUT_FEATURES = EXPERIENCE_FEATURES + PROPERTY_FEATURES + 1
PUBLIC_FEATURES = PROPERTY_FEATURES + 3
PRIVATE_FEATURES = PROPERTY_FEATURES + 10
LEDGER_FEATURES = 2
DECISION_TOKEN_FEATURES = PROPERTY_FEATURES + 8 + 3 + 3 + LEDGER_FEATURES
DECISION_CONTEXT_FEATURES = 16
DU_CONTEXT_INDEX = 9                       # position of the immediate utility change in the decision context

MODE_TRADE, MODE_LABOUR, MODE_CONSUME = 0, 1, 2


def public_tokens(x, alpha, gamma, goods_features, full_information: bool, continuous: bool = False) -> torch.Tensor:
    """(S, N, G, PUBLIC_FEATURES)."""
    fi = 1.0 if full_information else 0.0
    props = goods_features.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    return torch.cat([
        props,
        torch.log1p(x).unsqueeze(-1),
        (fi * alpha).unsqueeze(-1),
        (fi * marginal_utility(x, alpha, gamma, continuous)).unsqueeze(-1),
    ], dim=-1)


def private_tokens(x, alpha, gamma, goods_features, extras: Dict[str, torch.Tensor], continuous: bool = False) -> torch.Tensor:
    """
    (S, N, G, PRIVATE_FEATURES). extras: income (S,N,G), next_income (S,N), deprivation (S,N),
    feasible (S,N,G) maximal processing output, input_value (S,N,G), harvest (S,N,G) harvest yield.
    """
    G = x.shape[-1]
    props = goods_features.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    return torch.cat([
        props,
        torch.log1p(x).unsqueeze(-1),
        alpha.unsqueeze(-1),
        marginal_utility(x, alpha, gamma, continuous).unsqueeze(-1),
        gamma.unsqueeze(-1),
        torch.log1p(extras["income"]).unsqueeze(-1),
        extras["next_income"].unsqueeze(-1).unsqueeze(-1).expand(*x.shape[:2], G, 1),
        extras["deprivation"].unsqueeze(-1).unsqueeze(-1).expand(*x.shape[:2], G, 1),
        torch.log1p(extras["feasible"]).unsqueeze(-1),
        extras["input_value"].unsqueeze(-1),
        torch.log1p(extras["harvest"]).unsqueeze(-1),
    ], dim=-1)


def decision_inputs(
    mode: int, x_s, alpha_s, gamma_s, extras_s: Dict[str, torch.Tensor], x_p, alpha_p, gamma_p,
    goods_features, weight, recv, give, has_offer, turn, is_proposer, round_frac,
    capacity: float, full_information: bool, ledger_partner=None, ledger_bilateral=None, ledger_on: bool = False,
    continuous: bool = False,
):
    """
    Tokens (D, G, DECISION_TOKEN_FEATURES) and context (D, DECISION_CONTEXT_FEATURES) for D decisions of one mode.
    In labour/consume mode the partner and offer fields are zero. recv/give: standing offer seen by the decider.
    extras_s: income, feasible, input_value, harvest (D, G); next_income, deprivation (D,).
    ledger_partner: partner's public cumulative net contribution per good (given - received), (D, G).
    ledger_bilateral: net units the decider has received from this partner so far, (D, G). Both enter only in trade
    mode and only while the ledger is on, on a signed log scale sign(v) ln(1 + |v|).
    """
    D, G = x_s.shape
    fi = 1.0 if full_information else 0.0
    trade = 1.0 if mode == MODE_TRADE else 0.0
    x_after = x_s + recv - give
    du_k = alpha_s * (torch.log1p(excess(x_after, gamma_s, continuous)) - torch.log1p(excess(x_s, gamma_s, continuous)))
    tokens = torch.cat([
        goods_features,
        torch.log1p(x_s).unsqueeze(-1),
        alpha_s.unsqueeze(-1),
        marginal_utility(x_s, alpha_s, gamma_s, continuous).unsqueeze(-1),
        gamma_s.unsqueeze(-1),
        torch.log1p(extras_s["income"]).unsqueeze(-1),
        torch.log1p(extras_s["feasible"]).unsqueeze(-1),
        extras_s["input_value"].unsqueeze(-1),
        torch.log1p(extras_s["harvest"]).unsqueeze(-1),
        trade * torch.log1p(x_p).unsqueeze(-1),
        (trade * fi * alpha_p).unsqueeze(-1),
        (trade * fi * marginal_utility(x_p, alpha_p, gamma_p, continuous)).unsqueeze(-1),
        torch.log1p(recv).unsqueeze(-1),
        torch.log1p(give).unsqueeze(-1),
        du_k.unsqueeze(-1),
        _ledger_tokens(D, G, bool(trade) and ledger_on, ledger_partner, ledger_bilateral),
    ], dim=-1)

    du_total = utility(x_after, alpha_s, gamma_s, continuous) - utility(x_s, alpha_s, gamma_s, continuous)
    mode_onehot = torch.zeros(D, 3)
    mode_onehot[:, mode] = 1.0
    turn_onehot = trade * torch.nn.functional.one_hot(turn, 3).float()
    context = torch.cat([
        round_frac.unsqueeze(-1),
        mode_onehot,
        turn_onehot,
        (trade * is_proposer.float()).unsqueeze(-1),
        (trade * has_offer.float()).unsqueeze(-1),
        du_total.unsqueeze(-1),
        ((x_s * weight).sum(-1) / capacity).unsqueeze(-1),
        (trade * (x_p * weight).sum(-1) / capacity).unsqueeze(-1),
        torch.full((D, 1), fi),
        extras_s["next_income"].unsqueeze(-1),
        extras_s["deprivation"].unsqueeze(-1),
        torch.full((D, 1), 1.0 if ledger_on else 0.0),
    ], dim=-1)
    return tokens, context


def memory_inputs(experience: torch.Tensor, goods_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    (S, N, G, MEMORY_INPUT_FEATURES) input of the per-good memory update: the agent's own trade experience of the last
    round (log units received, given, asked for by partners, offered by partners), the good's properties and the
    agent's current holdings. Private: only the agent's own trades and the offers made to it.
    """
    props = goods_features.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    return torch.cat([experience, props, torch.log1p(x).unsqueeze(-1)], dim=-1)


def signed_log(v: torch.Tensor) -> torch.Tensor:
    return torch.sign(v) * torch.log1p(v.abs())


def _ledger_tokens(D, G, visible, partner, bilateral) -> torch.Tensor:
    if not visible or partner is None:
        return torch.zeros(D, G, LEDGER_FEATURES)
    return torch.stack([signed_log(partner), signed_log(bilateral)], dim=-1)
