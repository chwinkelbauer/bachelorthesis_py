"""
Trade network generation and random bilateral matching (search friction).
"""

import networkx as nx
import numpy as np
import torch

from configs.base import NetworkConfig


def build_networks(cfg: NetworkConfig, worlds: int, n_agents: int) -> torch.Tensor:
    """Returns a symmetric boolean adjacency (B, N, N) without self-loops."""
    adj = torch.zeros(worlds, n_agents, n_agents, dtype=torch.bool)
    for b in range(worlds):
        seed = int(torch.randint(0, 2**31 - 1, (1,)))
        if cfg.kind == "complete":
            graph = nx.complete_graph(n_agents)
        elif cfg.kind == "ring":
            graph = nx.watts_strogatz_graph(n_agents, cfg.degree, 0.0, seed=seed)
        elif cfg.kind == "small_world":
            graph = nx.connected_watts_strogatz_graph(n_agents, cfg.degree, cfg.rewire_prob, seed=seed)
        else:
            raise ValueError(f"Unknown network kind '{cfg.kind}'")
        a = nx.to_numpy_array(graph, nodelist=range(n_agents), dtype=np.float32) > 0
        adj[b] = torch.from_numpy(a)
    adj &= ~torch.eye(n_agents, dtype=torch.bool).unsqueeze(0)
    return adj


def random_maximal_matching(adj: torch.Tensor) -> torch.Tensor:
    """
    Uniformly random edge priorities, then repeatedly match locally dominant edges
    (both endpoints rank the edge highest among their free neighbours). Ends in a maximal matching.
    Returns partner (B, N) long, -1 for unmatched agents.
    """
    B, N, _ = adj.shape
    w = torch.triu(torch.rand(B, N, N), diagonal=1)
    w = w + w.transpose(1, 2)
    partner = torch.full((B, N), -1, dtype=torch.long)
    free = torch.ones(B, N, dtype=torch.bool)
    self_idx = torch.arange(N).unsqueeze(0).expand(B, N)
    for _ in range(N):
        valid = adj & free[:, :, None] & free[:, None, :]
        if not valid.any():
            break
        score = torch.where(valid, w, torch.full_like(w, -1.0))
        best_val, best = score.max(dim=-1)
        mutual = (best_val >= 0) & (best.gather(1, best) == self_idx)
        partner = torch.where(mutual, best, partner)
        free &= ~mutual
    return partner


def matched_pairs(partner: torch.Tensor):
    """Converts a partner array into pair index tensors (b, proposer, responder) with random roles."""
    B, N = partner.shape
    b, i = torch.nonzero((partner >= 0) & (partner > torch.arange(N).unsqueeze(0)), as_tuple=True)
    j = partner[b, i]
    swap = torch.rand(b.numel()) < 0.5
    proposer = torch.where(swap, j, i)
    responder = torch.where(swap, i, j)
    return b, proposer, responder
