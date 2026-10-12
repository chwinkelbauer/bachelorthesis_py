"""
Neural building blocks (pure PyTorch, dense adjacency; networks are small so dense attention is fastest).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1e9


def mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, out))


class GoodsSetEncoder(nn.Module):
    """DeepSets over goods tokens: (S, N, G, F) -> (S, N, H) via token MLP and mean+max pooling."""

    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.token = mlp(features, hidden, hidden)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.token(tokens)
        return self.out(torch.cat([h.mean(-2), h.amax(-2)], dim=-1))


class DenseGATv2(nn.Module):
    """
    GATv2 attention (Brody, Alon & Yahav 2022) over a dense adjacency, pre-LayerNorm residual block:
        e_ij = a^T LeakyReLU(W_dst h_i + W_src h_j [+ W_e e_ij]),  h_i' = h_i + W_o sum_j softmax_j(e_ij) W_src h_j
    Nodes without neighbours receive a zero message.
    """

    def __init__(self, hidden: int, heads: int, edge_features: int = 0):
        super().__init__()
        assert hidden % heads == 0
        self.heads, self.dh = heads, hidden // heads
        self.norm = nn.LayerNorm(hidden)
        self.w_src = nn.Linear(hidden, hidden)
        self.w_dst = nn.Linear(hidden, hidden)
        self.w_edge = nn.Linear(edge_features, hidden, bias=False) if edge_features else None
        self.att = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.w_out = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.LayerNorm(hidden), mlp(hidden, 2 * hidden, hidden))

    def forward(self, h: torch.Tensor, adj: torch.Tensor, edges: torch.Tensor = None) -> torch.Tensor:
        S, N, D = h.shape
        x = self.norm(h)
        src = self.w_src(x).view(S, 1, N, self.heads, self.dh)
        dst = self.w_dst(x).view(S, N, 1, self.heads, self.dh)
        e = src + dst
        if self.w_edge is not None and edges is not None:
            e = e + self.w_edge(edges).view(S, N, N, self.heads, self.dh)
        logits = (F.leaky_relu(e, 0.2) * self.att).sum(-1)                  # (S, N, N, H)
        logits = logits.masked_fill(~adj.unsqueeze(-1), MASK_VALUE)
        attn = torch.softmax(logits, dim=2) * adj.unsqueeze(-1).float()
        msg = (attn.unsqueeze(-1) * src).sum(2).reshape(S, N, D)
        h = h + self.w_out(msg)
        return h + self.ff(h)
