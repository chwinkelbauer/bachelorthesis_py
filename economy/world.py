"""
Batched economy: B independent worlds with N agents and G goods, stepped in lockstep.

Exchange economy (stage A), one round t:
    1. random maximal matching on the trade network      2. bilateral negotiation (learning/rollout.py)
    3. execution of accepted trades                        4. spoilage x_k <- (1 - delta_k) x_k
    reward r_t = U_i(x_{t+1}) - U_i(x_t) (utility of holdings, telescoping to final utility)

Production economy (stages B-D), one round t:
    1. income lumps arrive (stage B)                       2. labour: process inputs into a higher-tier good, or idle
    3. matching + negotiation + execution                  4. consumption c_t -> flow utility u_i(c_t)
    5. spoilage; deprivation counter (any subsistence good consumed below gamma); death after `death_after` rounds
    reward r_t = u_i(c_t) = sum_k alpha_ik ln(1 + max(c_tk - gamma_ik, 0)), minus `death_penalty` in the round an
    agent dies (default 0); dead agents leave the market, their holdings perish. Rollouts are truncated after R rounds and bootstrapped (effectively infinite horizon).
"""

from typing import Dict

import torch

from configs.base import ExperimentConfig
from economy.goods import generate_goods
from economy.preferences import sample_preferences, utility
from economy.endowments import sample_endowments
from economy.network import build_networks, random_maximal_matching, matched_pairs
from economy.observations import public_tokens, private_tokens, decision_inputs
from economy.production import (
    sample_income, income_arrives, rounds_to_next_income, max_output, usefulness, input_value,
    capacity_limited_output, ensure_subsistence_feasible, sample_harvest,
)
from economy.ledger import EpisodeRecord, FlowTracker


class EconomyBatch:
    def __init__(self, cfg: ExperimentConfig, worlds: int):
        if cfg.market.economy not in ("exchange", "production"):
            raise ValueError(f"Unknown economy '{cfg.market.economy}'")
        self.cfg = cfg
        self.production = cfg.market.economy == "production"
        self.B = worlds
        self.N = cfg.market.n_agents
        self.G = cfg.goods.n_goods
        self.R = cfg.market.rounds
        self.capacity = cfg.market.capacity
        self.full_information = cfg.market.full_information

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        cfg, B, N, G = self.cfg, self.B, self.N, self.G
        self.goods = generate_goods(cfg.goods, B)
        self.goods_features = self.goods.features()
        self.weight = self.goods.weight
        self.spoilage = self.goods.spoilage if cfg.market.spoilage else torch.zeros_like(self.goods.spoilage)
        self.alpha, self.gamma = sample_preferences(cfg.preferences, self.goods, N)
        if self.production:
            pc = cfg.production
            if pc.income:
                self.income, self.period, self.phase = sample_income(pc, self.goods, N)
            else:
                self.income = torch.zeros(B, N, G)
                self.period = torch.ones(B, N, dtype=torch.long)
                self.phase = torch.zeros(B, N, dtype=torch.long)
            self.recipe = self.goods.recipe if pc.processing else torch.zeros_like(self.goods.recipe)
            if pc.max_tier >= 0:   # technology level (learning sessions): higher tiers cannot be produced yet
                self.recipe = self.recipe * (self.goods.tier <= pc.max_tier).unsqueeze(-1)
            if pc.income:
                self.income = ensure_subsistence_feasible(self.income, self.period, self.gamma, self.goods.recipe,
                                                          self.goods.tier, pc.subsistence_surplus)
            self.harvest = (sample_harvest(pc, self.goods, self.gamma, self.goods.recipe) if pc.harvesting
                            else torch.zeros(B, N, G))
            # start with one income lump and one harvest of every accessible good
            self.x = self._fit_capacity(torch.zeros(B, N, G), self.income + self.harvest)
            # starting reserve of the subsistence good: time to learn to trade before deprivation kills
            self.x = self._fit_capacity(self.x, self.gamma * pc.subsistence_buffer_rounds)
        else:
            self.income = torch.zeros(B, N, G)
            self.period = torch.ones(B, N, dtype=torch.long)
            self.phase = torch.zeros(B, N, dtype=torch.long)
            self.recipe = torch.zeros_like(self.goods.recipe)
            self.harvest = torch.zeros(B, N, G)
            self.x = sample_endowments(cfg.endowments, self.goods, N, self.capacity)
        self.useful = usefulness(self.alpha, self.recipe)
        self.input_value = input_value(self.alpha, self.recipe)
        self.adj = build_networks(cfg.network, B, N)
        self.alive = torch.ones(B, N, dtype=torch.bool)
        self.deprivation = torch.zeros(B, N)
        self._deprived = torch.zeros(B, N, dtype=torch.bool)
        self.round = 0
        self.contribution = torch.zeros(B, N, G)               # public record: units given minus received
        self.bilateral = torch.zeros(B, N, N, G)               # bilateral[b, i, j]: net units i received from j
        flows = None
        if self.production:
            flows = FlowTracker(B, N, G, self.R)
            flows.supply += self.x.sum(1)
        self.record = EpisodeRecord(
            alpha=self.alpha, gamma=self.gamma, x0=self.x.clone(), x_final=self.x.clone(),
            weight=self.weight, spoilage=self.spoilage, goods_features=self.goods_features, rounds=self.R,
            flows=flows, ledger_switch_round=cfg.ledger.switch_round,
        )

    def _fit_capacity(self, x: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
        """Adds `add` to holdings, scaled down per agent so that the load stays within capacity."""
        load = (x * self.weight.unsqueeze(1)).sum(-1)
        extra = (add * self.weight.unsqueeze(1)).sum(-1)
        scale = torch.clamp((self.capacity - load).clamp(min=0.0) / extra.clamp(min=1e-12), max=1.0)
        return x + add * scale.unsqueeze(-1)

    # ------------------------------------------------------------------ economics
    def utility(self, x: torch.Tensor = None) -> torch.Tensor:
        return utility(self.x if x is None else x, self.alpha, self.gamma)

    def ledger_on(self) -> bool:
        """Ledger state in the current round: `enabled`, flipped from `switch_round` on."""
        sw = self.cfg.ledger.switch_round
        flipped = sw >= 0 and self.round >= sw
        return self.cfg.ledger.enabled != flipped

    def adjacency(self) -> torch.Tensor:
        """Trade network restricted to living agents."""
        return self.adj & self.alive.unsqueeze(1) & self.alive.unsqueeze(2)

    def begin_round(self) -> None:
        """Production economy: income lumps arrive."""
        if not self.production:
            return
        self.record.flows.round = self.round
        arrive = income_arrives(self.period, self.phase, self.round) & self.alive
        before = self.x
        self.x = self._fit_capacity(self.x, self.income * arrive.unsqueeze(-1))
        self.record.flows.supply += (self.x - before).sum(1)
        self.record.flows.alive_rounds += self.alive.float()

    def labour(self, b, i, choice, amount) -> None:
        """
        Labour choice per agent: 0 = idle, 1..G = harvest good k-1 (stage C, yield y_ik, capacity-limited),
        G+1..2G = process good k-G-1 (output = amount * q_max, capacity-limited; inputs are used up).
        """
        G = self.G
        self.record.flows.labour_counts.index_put_((b, i, choice), torch.ones_like(choice, dtype=torch.float), accumulate=True)
        is_harvest = (choice >= 1) & (choice <= G)
        if is_harvest.any():
            hb, hi, k = b[is_harvest], i[is_harvest], choice[is_harvest] - 1
            x = self.x[hb, hi]
            w = self.weight[hb].gather(-1, k.unsqueeze(-1)).squeeze(-1)
            room = torch.clamp(self.capacity - (x * self.weight[hb]).sum(-1), min=0.0)
            q = torch.minimum(self.harvest[hb, hi, k], room / w)
            self.x[hb, hi, k] += q
            self.record.flows.harvested.index_put_((hb, k), q, accumulate=True)
            self.record.flows.supply.index_put_((hb, k), q, accumulate=True)
        is_process = choice > G
        if is_process.any():
            b, i, k, a = b[is_process], i[is_process], choice[is_process] - G - 1, amount[is_process]
            x = self.x[b, i]
            rec = self.recipe[b, k]                                          # (D, G) inputs per unit of output
            q_max = max_output(x.unsqueeze(1), self.recipe[b]).squeeze(1)       # (D, G)
            q = a * q_max.gather(-1, k.unsqueeze(-1)).squeeze(-1)
            w = self.weight[b]
            load = (x * w).sum(-1)
            q = capacity_limited_output(q, w.gather(-1, k.unsqueeze(-1)).squeeze(-1), (rec * w).sum(-1), load, self.capacity)
            used = q.unsqueeze(-1) * rec
            self.record.flows.remove(b, i, used, x)
            new = x - used
            new[torch.arange(k.numel()), k] += q
            self.x[b, i] = new.clamp(min=0.0)
            self.record.flows.produced.index_put_((b, k), q, accumulate=True)
            self.record.flows.supply.index_put_((b, k), q, accumulate=True)

    def match(self):
        partner = random_maximal_matching(self.adjacency())
        b, proposer, responder = matched_pairs(partner)
        self.record.meetings += b.numel()
        self.record.meetings_by_round.append(b.numel())
        return b, proposer, responder

    def execute(self, b, a, p, recv_a, give_a, turn) -> None:
        """Executes trades (a receives recv_a and gives give_a to p) and writes them to the ledger."""
        if b.numel() == 0:
            return
        xa, xp = self.x[b, a].clone(), self.x[b, p].clone()
        u_a = utility(xa, self.alpha[b, a], self.gamma[b, a])
        u_p = utility(xp, self.alpha[b, p], self.gamma[b, p])
        if self.production:
            flows = self.record.flows
            useful_a = (recv_a * self.useful[b, a]).sum(-1) > 0
            useful_p = (give_a * self.useful[b, p]).sum(-1) > 0
            flows.trade(b, a, recv_a, give_a, xa, useful_a)
            flows.trade(b, p, give_a, recv_a, xp, useful_p)
        self.x[b, a] = (xa + recv_a - give_a).clamp(min=0.0)
        self.x[b, p] = (xp + give_a - recv_a).clamp(min=0.0)
        net_a = recv_a - give_a
        self.contribution[b, a] -= net_a
        self.contribution[b, p] += net_a
        self.bilateral[b, a, p] += net_a
        self.bilateral[b, p, a] -= net_a
        du_a = utility(self.x[b, a], self.alpha[b, a], self.gamma[b, a]) - u_a
        du_p = utility(self.x[b, p], self.alpha[b, p], self.gamma[b, p]) - u_p
        self.record.add(b, self.round, a, p, recv_a, give_a, du_a, du_p, turn)

    def consume(self, b, i, choice, amount) -> torch.Tensor:
        """Consumes amount * x_k of every good with choice 1; returns the flow utility (D,)."""
        x = self.x[b, i]
        c = (choice == 1).float() * amount * x
        flows = self.record.flows
        flows.remove(b, i, c, x)
        flows.consumed.index_put_((b,), c, accumulate=True)
        self.x[b, i] = (x - c).clamp(min=0.0)
        u = utility(c, self.alpha[b, i], self.gamma[b, i])
        flows.utility.index_put_((b, i), u, accumulate=True)
        gamma = self.gamma[b, i]
        self._deprived[b, i] = ((gamma > 0) & (c < gamma)).any(-1)
        return u

    def end_round(self) -> torch.Tensor:
        """Spoilage, deprivation and death. Returns agents that died this round (B, N)."""
        keep = (1.0 - self.spoilage).unsqueeze(1)
        self.x *= keep
        died = torch.zeros(self.B, self.N, dtype=torch.bool)
        if self.production:
            flows = self.record.flows
            flows.pool *= keep
            self.deprivation = torch.where(self._deprived & self.alive, self.deprivation + 1, torch.zeros_like(self.deprivation))
            died = self.alive & (self.deprivation >= self.cfg.market.death_after)
            self.alive &= ~died
            self.x[died] = 0.0
            flows.pool[died] = 0.0
            flows.deaths += died.float()
            self._deprived.zero_()
        self.round += 1
        if self.round >= self.R:
            self.record.x_final = self.x.clone()
        return died

    # ------------------------------------------------------------------ observations
    def _extras(self, x: torch.Tensor, b: torch.Tensor, t, deprivation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stage-dependent private features for holdings x (S, N, G) of worlds b at round(s) t."""
        if self.production:
            t = torch.as_tensor(t)
            t = t.view(-1, 1) if t.dim() > 0 else t
            nxt = rounds_to_next_income(self.period[b], self.phase[b], t).float() / self.period[b].float()
            feasible = max_output(x, self.recipe[b])
        else:
            nxt = torch.zeros(x.shape[:2])
            feasible = torch.zeros_like(x)
        return {
            "income": self.income[b], "next_income": nxt,
            "deprivation": deprivation / self.cfg.market.death_after,
            "feasible": feasible, "input_value": self.input_value[b], "harvest": self.harvest[b],
        }

    def public_tokens(self, x: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
        b = torch.arange(self.B) if b is None else b
        return public_tokens(x, self.alpha[b], self.gamma[b], self.goods_features[b], self.full_information)

    def private_tokens(self, x: torch.Tensor, b: torch.Tensor = None, t=None, deprivation: torch.Tensor = None) -> torch.Tensor:
        b = torch.arange(self.B) if b is None else b
        t = self.round if t is None else t
        deprivation = self.deprivation[b] if deprivation is None else deprivation
        extras = self._extras(x, b, t, deprivation)
        return private_tokens(x, self.alpha[b], self.gamma[b], self.goods_features[b], extras)

    def decision_tensors(self, mode, b, s, p, x_s, x_p, recv, give, has_offer, turn, is_proposer):
        """Decision inputs for agents s (partners p) in worlds b at the current round."""
        D = b.numel()
        if self.production:
            nxt = rounds_to_next_income(self.period[b, s], self.phase[b, s], self.round).float() / self.period[b, s].float()
            feasible = max_output(x_s.unsqueeze(1), self.recipe[b]).squeeze(1)
        else:
            nxt, feasible = torch.zeros(D), torch.zeros_like(x_s)
        extras_s = {
            "income": self.income[b, s], "next_income": nxt,
            "deprivation": self.deprivation[b, s] / self.cfg.market.death_after,
            "feasible": feasible, "input_value": self.input_value[b, s], "harvest": self.harvest[b, s],
        }
        round_frac = torch.full((D,), 0.0 if self.production else self.round / max(self.R - 1, 1))
        return decision_inputs(
            mode, x_s, self.alpha[b, s], self.gamma[b, s], extras_s, x_p, self.alpha[b, p], self.gamma[b, p],
            self.goods_features[b], self.weight[b], recv, give, has_offer, turn, is_proposer, round_frac,
            self.capacity, self.full_information,
            ledger_partner=self.contribution[b, p], ledger_bilateral=self.bilateral[b, s, p], ledger_on=self.ledger_on(),
        )

    def labour_mask(self, b, s, x_s) -> torch.Tensor:
        """(D, 1 + 2G) bool: idle always; harvest goods the agent can harvest (stage C); process feasible goods."""
        D, G = x_s.shape
        mask = torch.zeros(D, 1 + 2 * G, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 1:G + 1] = self.harvest[b, s] > 0
        mask[:, G + 1:] = max_output(x_s.unsqueeze(1), self.recipe[b]).squeeze(1) > 1e-6
        return mask

    def consume_mask(self, b, s, x_s) -> torch.Tensor:
        """(D, G, 2) bool: keeping is always allowed; consuming only goods held and valued (alpha > 0)."""
        mask = torch.ones(*x_s.shape, 2, dtype=torch.bool)
        mask[..., 1] = (x_s > 1e-6) & (self.alpha[b, s] > 0)
        return mask

    def peek_next_observation(self):
        """Holdings at the start of the next round (after its income), without changing the world: bootstrap state."""
        if not self.production:
            return self.x.clone()
        arrive = income_arrives(self.period, self.phase, self.round) & self.alive
        return self._fit_capacity(self.x, self.income * arrive.unsqueeze(-1))
