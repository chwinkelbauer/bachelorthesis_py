"""
Transaction ledger: every executed trade of an episode batch, plus the static facts needed to analyse it.
All money metrics are computed from this record (analysis/money_metrics.py).

Production economies (stages B-D) additionally carry a FlowTracker that follows every unit acquired through trade
until it leaves the agent (traded on, consumed, used as a production input, spoiled, or lost at death).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class EpisodeRecord:
    alpha: torch.Tensor          # (B, N, G)
    gamma: torch.Tensor          # (B, N, G)
    x0: torch.Tensor             # (B, N, G) holdings at the start
    x_final: torch.Tensor        # (B, N, G) holdings at the end
    weight: torch.Tensor         # (B, G)
    spoilage: torch.Tensor       # (B, G) effective per-round spoilage (0 if disabled)
    goods_features: torch.Tensor # (B, G, F)
    rounds: int
    meetings: int = 0            # matched pairs over the episode
    meetings_by_round: List[int] = field(default_factory=list)
    ledger_switch_round: int = -1
    flows: Optional["FlowTracker"] = None   # production economies only
    # One row per executed trade; a = the pair's proposer (opening offer), p = the responder.
    b: List[torch.Tensor] = field(default_factory=list)
    r: List[torch.Tensor] = field(default_factory=list)
    a: List[torch.Tensor] = field(default_factory=list)
    p: List[torch.Tensor] = field(default_factory=list)
    recv_a: List[torch.Tensor] = field(default_factory=list)   # units a receives (= units p gives)
    give_a: List[torch.Tensor] = field(default_factory=list)   # units a gives (= units p receives)
    du_a: List[torch.Tensor] = field(default_factory=list)     # immediate utility change of a
    du_p: List[torch.Tensor] = field(default_factory=list)
    turn: List[torch.Tensor] = field(default_factory=list)     # turn at which the trade was accepted

    def add(self, b, r, a, p, recv_a, give_a, du_a, du_p, turn) -> None:
        if b.numel() == 0:
            return
        self.b.append(b); self.r.append(torch.full_like(b, r)); self.a.append(a); self.p.append(p)
        self.recv_a.append(recv_a); self.give_a.append(give_a)
        self.du_a.append(du_a); self.du_p.append(du_p); self.turn.append(turn)

    def trades(self) -> dict:
        """Concatenated trade table (empty tensors if no trade happened)."""
        G = self.x0.shape[-1]
        if not self.b:
            empty = torch.zeros(0, dtype=torch.long)
            return dict(b=empty, r=empty, a=empty, p=empty, recv_a=torch.zeros(0, G), give_a=torch.zeros(0, G),
                        du_a=torch.zeros(0), du_p=torch.zeros(0), turn=empty)
        return dict(
            b=torch.cat(self.b), r=torch.cat(self.r), a=torch.cat(self.a), p=torch.cat(self.p),
            recv_a=torch.cat(self.recv_a), give_a=torch.cat(self.give_a),
            du_a=torch.cat(self.du_a), du_p=torch.cat(self.du_p), turn=torch.cat(self.turn),
        )


class FlowTracker:
    """
    Fate of traded units in a production economy, with proportional attribution: an agent's holding x_ik contains a
    pool P_ik <= x_ik of units acquired through trade; every outflow of k removes the fraction outflow / x_ik from the
    pool. Units passed on by trade count as *completed* pass-through only if the same trade brings the agent a good
    that is useful to it (consumed, or transitively needed as a production input).
    Volumes per world and good (B, G):
        supply    initial holdings + income + production
        acquired  units received through trade
        passed    completed pass-through volume
        consumed  units consumed
        produced / harvested  units made by processing / harvesting
    Per agent (B, N): utility (summed flow utility), alive_rounds, deaths.
    """

    def __init__(self, B: int, N: int, G: int, R: int):
        self.round = 0                                    # set by the world; used for per-round volumes
        self.passed_by_round = torch.zeros(R, B, G)
        self.acquired_by_round = torch.zeros(R, B, G)
        self.pool = torch.zeros(B, N, G)
        self.supply = torch.zeros(B, G)
        self.acquired = torch.zeros(B, G)
        self.passed = torch.zeros(B, G)
        self.consumed = torch.zeros(B, G)
        self.produced = torch.zeros(B, G)
        self.harvested = torch.zeros(B, G)
        self.labour_counts = torch.zeros(B, N, 1 + 2 * G)   # how often each labour option was chosen
        self.utility = torch.zeros(B, N)
        self.alive_rounds = torch.zeros(B, N)
        self.deaths = torch.zeros(B, N)

    @staticmethod
    def _keep_fraction(outflow: torch.Tensor, holding: torch.Tensor) -> torch.Tensor:
        return torch.where(holding > 1e-12, 1.0 - outflow / holding.clamp(min=1e-12), torch.zeros_like(holding)).clamp(0.0, 1.0)

    def remove(self, b, i, outflow, holding_before) -> None:
        """Non-trade outflow (consumption, production input) from agents (b, i)."""
        self.pool[b, i] = self.pool[b, i] * self._keep_fraction(outflow, holding_before)

    def trade(self, b, i, inflow, outflow, holding_before, receives_useful) -> None:
        """One side of executed trades: pass-through from the pool, then the inflow joins the pool."""
        share = torch.where(holding_before > 1e-12, self.pool[b, i] / holding_before.clamp(min=1e-12), torch.zeros_like(holding_before))
        passed = outflow * share.clamp(0.0, 1.0)
        completed = passed * receives_useful.unsqueeze(-1).float()
        self.passed.index_put_((b,), completed, accumulate=True)
        self.acquired.index_put_((b,), inflow, accumulate=True)
        self.passed_by_round[self.round].index_put_((b,), completed, accumulate=True)
        self.acquired_by_round[self.round].index_put_((b,), inflow, accumulate=True)
        self.pool[b, i] = self.pool[b, i] - passed + inflow
