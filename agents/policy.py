"""
Preference-conditioned shared policy and centralised critic.

Policy (decentralised execution):
    agent embedding  z_i = MLP([Enc_priv(own goods tokens), GATv2^L(Enc_pub(public goods tokens))_i]),
                     computed from the state at the start of the round (market context)
    decision trunk   goods tokens (current state, partner, standing offer) + context [mode, z_self, h_other]
                     -> Transformer over goods with a leading decision token
    heads            trade:   response {accept, counter, reject}; per good direction {none, give, want} + Beta amount
                     labour:  {idle} + per good {harvest, process}; Beta amount (fraction of maximal processing)
                     consume: per good {keep, consume}; Beta amount (fraction of holdings)
Critic (centralised training): V_i = MLP([h_i, mean_j h_j]) with h = GATv2^L(Enc(all private tokens)).

Optional memory (cfg.memory): every agent keeps a state mu_ik per good k, updated once per round by a GRU cell from the
agent's own trade experience of the last round, the good's properties and its holdings:
    mu_ik,t = GRU(mu_ik,t-1, [experience_ik,t, properties_k, ln(1 + x_ik,t)]),   mu_ik,0 = 0
The state is appended to the agent's own goods tokens (agent embedding) and to its decision tokens (per-good heads).
It is shared over goods (permutation-equivariant) and trained end to end through all rounds of an episode. The critic
additionally sees the raw experience features. With memory off the model is exactly the memory-free model.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.layers import GoodsSetEncoder, DenseGATv2, mlp, MASK_VALUE
from configs.base import ModelConfig
from economy.observations import (
    PUBLIC_FEATURES, PRIVATE_FEATURES, DECISION_TOKEN_FEATURES, DECISION_CONTEXT_FEATURES,
    EXPERIENCE_FEATURES, MEMORY_INPUT_FEATURES,
)
from economy.trade import COUNTER, NONE

AMOUNT_EPS = 1e-3


@dataclass
class Decision:
    """Trade decision."""
    response: torch.Tensor   # (D,) long
    direction: torch.Tensor  # (D, G) long
    amount: torch.Tensor     # (D, G) in (0, 1)
    log_prob: torch.Tensor   # (D,)
    entropy: torch.Tensor    # (D,)


@dataclass
class LabourDecision:
    choice: torch.Tensor     # (D,) long: 0 idle, 1..G harvest, G+1..2G process
    amount: torch.Tensor     # (D,) fraction of the maximal processing output
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass
class ConsumeDecision:
    choice: torch.Tensor     # (D, G) long: 0 keep, 1 consume
    amount: torch.Tensor     # (D, G) fraction of holdings consumed
    log_prob: torch.Tensor
    entropy: torch.Tensor


def _beta(params: torch.Tensor) -> torch.distributions.Beta:
    ab = F.softplus(params) + 1.0
    return torch.distributions.Beta(ab[..., 0], ab[..., 1])


def _sample_amount(dist, greedy: bool) -> torch.Tensor:
    return (dist.mean if greedy else dist.sample()).clamp(AMOUNT_EPS, 1 - AMOUNT_EPS)


class PolicyNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden
        M = cfg.memory_size if cfg.memory else 0
        self.memory_size = M
        self.memory = nn.GRUCell(MEMORY_INPUT_FEATURES, M) if M else None
        self.pub_enc = GoodsSetEncoder(PUBLIC_FEATURES, H)
        self.priv_enc = GoodsSetEncoder(PRIVATE_FEATURES + M, H)
        self.gnn = nn.ModuleList([DenseGATv2(H, cfg.attn_heads) for _ in range(cfg.gnn_layers)])
        self.agent_mlp = nn.Sequential(mlp(2 * H, H, H), nn.LayerNorm(H))
        self.token_in = nn.Linear(DECISION_TOKEN_FEATURES + M, H)
        self.ctx_in = mlp(DECISION_CONTEXT_FEATURES + 2 * H, H, H)
        layer = nn.TransformerEncoderLayer(H, cfg.attn_heads, 2 * H, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.transformer_layers, enable_nested_tensor=False)
        # trade heads
        self.response_head = mlp(H, H, 3)
        self.direction_head = nn.Linear(H, 3)
        self.amount_head = nn.Linear(H, 2)
        # labour heads
        self.idle_head = mlp(H, H, 1)
        self.labour_head = nn.Linear(H, 2)
        self.labour_amount_head = nn.Linear(H, 2)
        # consumption heads
        self.consume_head = nn.Linear(H, 2)
        self.consume_amount_head = nn.Linear(H, 2)

    # ---------------------------------------------------------------- memory
    def initial_memory(self, S: int, N: int, G: int) -> torch.Tensor:
        return torch.zeros(S, N, G, self.memory_size)

    def remember(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """One memory update: inputs (S, N, G, MEMORY_INPUT_FEATURES), memory (S, N, G, M) -> (S, N, G, M)."""
        shape = memory.shape
        return self.memory(inputs.reshape(-1, inputs.shape[-1]), memory.reshape(-1, shape[-1])).view(shape)

    # ---------------------------------------------------------------- embeddings
    def embed(self, public_tokens: torch.Tensor, private_tokens: torch.Tensor, adj: torch.Tensor):
        """Returns (z, h_pub): private agent embeddings (S, N, H) and public neighbourhood embeddings (S, N, H)."""
        h = self.pub_enc(public_tokens)
        for layer in self.gnn:
            h = layer(h, adj)
        z = self.agent_mlp(torch.cat([self.priv_enc(private_tokens), h], dim=-1))
        return z, h

    def _trunk(self, z_self, h_other, tokens, context):
        c = self.ctx_in(torch.cat([context, z_self, h_other], dim=-1))
        seq = torch.cat([c.unsqueeze(1), self.token_in(tokens) + c.unsqueeze(1)], dim=1)
        out = self.transformer(seq)
        return out[:, 0], out[:, 1:]

    # ---------------------------------------------------------------- trade
    def decide(self, z_self, h_partner, tokens, context, dir_mask, resp_mask,
               actions: Optional[Decision] = None, greedy: bool = False) -> Decision:
        cls, goods = self._trunk(z_self, h_partner, tokens, context)
        resp_logits = self.response_head(cls).masked_fill(~resp_mask, MASK_VALUE)
        dir_logits = self.direction_head(goods).masked_fill(~dir_mask, MASK_VALUE)
        d_resp = torch.distributions.Categorical(logits=resp_logits)
        d_dir = torch.distributions.Categorical(logits=dir_logits)
        d_amt = _beta(self.amount_head(goods))

        if actions is None:
            if greedy:
                response, direction = resp_logits.argmax(-1), dir_logits.argmax(-1)
            else:
                response, direction = d_resp.sample(), d_dir.sample()
            amount = _sample_amount(d_amt, greedy)
        else:
            response, direction, amount = actions.response, actions.direction, actions.amount

        counter = (response == COUNTER).float()
        active = (direction != NONE).float()
        lp_offer = d_dir.log_prob(direction).sum(-1) + (d_amt.log_prob(amount) * active).sum(-1)
        log_prob = d_resp.log_prob(response) + counter * lp_offer
        ent_offer = d_dir.entropy().sum(-1) + (d_amt.entropy() * active).sum(-1)
        entropy = d_resp.entropy() + counter * ent_offer
        return Decision(response, direction, amount, log_prob, entropy)

    # ---------------------------------------------------------------- labour
    def labour(self, z_self, h_self, tokens, context, mask,
               actions: Optional[LabourDecision] = None, greedy: bool = False) -> LabourDecision:
        """mask (D, 1 + 2G): idle, harvest per good, process per good."""
        cls, goods = self._trunk(z_self, h_self, tokens, context)
        D, G = goods.shape[:2]
        per_good = self.labour_head(goods)                                  # (D, G, 2)
        logits = torch.cat([self.idle_head(cls), per_good[..., 0], per_good[..., 1]], dim=-1)
        logits = logits.masked_fill(~mask, MASK_VALUE)
        d_choice = torch.distributions.Categorical(logits=logits)
        d_amt = _beta(self.labour_amount_head(goods))                       # (D, G)

        if actions is None:
            choice = logits.argmax(-1) if greedy else d_choice.sample()
            good = torch.remainder(choice - 1, G).clamp(min=0)
            amount = _sample_amount(d_amt, greedy).gather(-1, good.unsqueeze(-1)).squeeze(-1)
        else:
            choice, amount = actions.choice, actions.amount
            good = torch.remainder(choice - 1, G).clamp(min=0)
        processing = (choice > G).float()
        lp_amt = torch.distributions.Beta(d_amt.concentration1.gather(-1, good.unsqueeze(-1)).squeeze(-1),
                                          d_amt.concentration0.gather(-1, good.unsqueeze(-1)).squeeze(-1)).log_prob(amount)
        log_prob = d_choice.log_prob(choice) + processing * lp_amt
        entropy = d_choice.entropy()
        return LabourDecision(choice, amount, log_prob, entropy)

    # ---------------------------------------------------------------- consumption
    def consume(self, z_self, h_self, tokens, context, mask,
                actions: Optional[ConsumeDecision] = None, greedy: bool = False) -> ConsumeDecision:
        """mask (D, G, 2): keep always allowed; consume only goods held and valued."""
        cls, goods = self._trunk(z_self, h_self, tokens, context)
        logits = self.consume_head(goods).masked_fill(~mask, MASK_VALUE)
        d_choice = torch.distributions.Categorical(logits=logits)
        d_amt = _beta(self.consume_amount_head(goods))
        if actions is None:
            choice = logits.argmax(-1) if greedy else d_choice.sample()
            amount = _sample_amount(d_amt, greedy)
        else:
            choice, amount = actions.choice, actions.amount
        active = (choice == 1).float()
        log_prob = d_choice.log_prob(choice).sum(-1) + (d_amt.log_prob(amount) * active).sum(-1)
        entropy = d_choice.entropy().sum(-1) + (d_amt.entropy() * active).sum(-1)
        return ConsumeDecision(choice, amount, log_prob, entropy)


class CriticNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden
        self.extra_features = EXPERIENCE_FEATURES if cfg.memory else 0   # raw trade experience (memory runs)
        self.enc = GoodsSetEncoder(PRIVATE_FEATURES + self.extra_features, H)
        self.gnn = nn.ModuleList([DenseGATv2(H, cfg.attn_heads) for _ in range(cfg.gnn_layers)])
        self.head = mlp(2 * H, H, 1)

    def forward(self, private_tokens: torch.Tensor, adj: torch.Tensor, alive: torch.Tensor = None) -> torch.Tensor:
        h = self.enc(private_tokens)
        for layer in self.gnn:
            h = layer(h, adj)
        if alive is None:
            g = h.mean(1, keepdim=True)
        else:
            w = alive.float().unsqueeze(-1)
            g = (h * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(torch.cat([h, g.expand_as(h)], dim=-1)).squeeze(-1)
