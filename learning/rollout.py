"""
Episode rollout over all worlds in parallel.

Trade protocol per matched pair (proposer i, responder j), K = 2 offers at most:
    turn 0  i makes an opening offer o_i
    turn 1  j accepts o_i, rejects, or makes a counter-offer o_j
    turn 2  (only after a counter-offer) i accepts o_j or rejects
Production economies add a labour phase before and a consumption phase after trading (see economy/world.py).

Every decision is stored with its exact inputs (tokens, context, masks) and the indices needed to recompute the
differentiable agent embeddings from the round's state snapshot during the PPO update.
"""

from dataclasses import dataclass
from typing import Dict

import torch

from economy.observations import MODE_TRADE, MODE_LABOUR, MODE_CONSUME
from economy.trade import decode_offer, direction_mask, response_mask, ACCEPT, COUNTER
from economy.world import EconomyBatch

TRADE_FIELDS = ("r", "b", "s", "p", "tokens", "context", "dir_mask", "resp_mask", "response", "direction", "amount", "log_prob")
SELF_FIELDS = ("r", "b", "s", "tokens", "context", "mask", "choice", "amount", "log_prob")


@dataclass
class Rollout:
    X: torch.Tensor            # (S, B, N, G) holdings at the start of each round (after income); S = R (+1 bootstrap)
    deprivation: torch.Tensor  # (S, B, N)
    alive: torch.Tensor        # (S, B, N) bool
    rewards: torch.Tensor      # (R, B, N)
    dones: torch.Tensor        # (R, B, N) bool: agent died in this round
    bootstrap: bool            # production: value of the state after the last round is bootstrapped
    trade: Dict[str, torch.Tensor]
    labour: Dict[str, torch.Tensor]
    consume: Dict[str, torch.Tensor]


def _store(store, **row):
    for k, v in row.items():
        store[k].append(v)


def _cat(store) -> Dict[str, torch.Tensor]:
    return {k: torch.cat(v) for k, v in store.items()} if store["r"] else {}


def _negotiate(world, policy, z, h, r, b, s, p, recv, give, has_offer, turn_id, is_proposer, greedy, store):
    D = b.numel()
    turn = torch.full((D,), turn_id, dtype=torch.long)
    has = torch.full((D,), has_offer, dtype=torch.bool)
    prop = torch.full((D,), is_proposer, dtype=torch.bool)
    x_s, x_p = world.x[b, s], world.x[b, p]
    tokens, context = world.decision_tensors(MODE_TRADE, b, s, p, x_s, x_p, recv, give, has, turn, prop)
    dmask, rmask = direction_mask(x_s, x_p), response_mask(turn)
    d = policy.decide(z[b, s], h[b, p], tokens, context, dmask, rmask, greedy=greedy)
    _store(store, r=torch.full((D,), r, dtype=torch.long), b=b, s=s, p=p, tokens=tokens, context=context,
           dir_mask=dmask, resp_mask=rmask, response=d.response, direction=d.direction, amount=d.amount,
           log_prob=d.log_prob)
    return d


def _self_decision(world, policy, z, h, r, mode, greedy, store):
    b, s = torch.nonzero(world.alive, as_tuple=True)
    D, G = b.numel(), world.G
    x_s = world.x[b, s]
    zeros = torch.zeros(D, G)
    off = torch.zeros(D, dtype=torch.bool)
    tokens, context = world.decision_tensors(mode, b, s, s, x_s, zeros, zeros, zeros, off,
                                             torch.zeros(D, dtype=torch.long), off)
    if mode == MODE_LABOUR:
        mask = world.labour_mask(b, s, x_s)
        d = policy.labour(z[b, s], h[b, s], tokens, context, mask, greedy=greedy)
    else:
        mask = world.consume_mask(b, s, x_s)
        d = policy.consume(z[b, s], h[b, s], tokens, context, mask, greedy=greedy)
    _store(store, r=torch.full((D,), r, dtype=torch.long), b=b, s=s, tokens=tokens, context=context, mask=mask,
           choice=d.choice, amount=d.amount, log_prob=d.log_prob)
    return b, s, d


def _trade_phase(world, policy, z, h, r, greedy, store) -> None:
    b, i, j = world.match()
    if b.numel() == 0:
        return
    G = world.G
    x = world.x
    w = world.weight[b]
    zeros = torch.zeros(b.numel(), G)

    d0 = _negotiate(world, policy, z, h, r, b, i, j, zeros, zeros, False, 0, True, greedy, store)
    give_i, want_i = decode_offer(d0.direction, d0.amount, x[b, i], x[b, j], w, world.capacity)

    d1 = _negotiate(world, policy, z, h, r, b, j, i, give_i, want_i, True, 1, False, greedy, store)
    give_j, want_j = decode_offer(d1.direction, d1.amount, x[b, j], x[b, i], w, world.capacity)
    accept1 = d1.response == ACCEPT
    countered = torch.nonzero(d1.response == COUNTER).flatten()

    accept2 = torch.zeros_like(accept1)
    if countered.numel() > 0:
        c = countered
        d2 = _negotiate(world, policy, z, h, r, b[c], i[c], j[c], give_j[c], want_j[c], True, 2, True, greedy, store)
        accept2[c] = d2.response == ACCEPT

    # holdings change from the proposer's point of view
    recv_i = torch.where(accept1[:, None], want_i, torch.where(accept2[:, None], give_j, zeros))
    out_i = torch.where(accept1[:, None], give_i, torch.where(accept2[:, None], want_j, zeros))
    done = torch.nonzero(accept1 | accept2).flatten()
    turn = torch.where(accept1, 1, 2)[done]
    world.execute(b[done], i[done], j[done], recv_i[done], out_i[done], turn)


@torch.no_grad()
def collect_rollout(world: EconomyBatch, policy, greedy: bool = False) -> Rollout:
    world.reset()
    B, N, G, R = world.B, world.N, world.G, world.R
    production = world.production
    S = R + 1 if production else R
    X = torch.zeros(S, B, N, G)
    DEP = torch.zeros(S, B, N)
    ALIVE = torch.zeros(S, B, N, dtype=torch.bool)
    rewards = torch.zeros(R, B, N)
    dones = torch.zeros(R, B, N, dtype=torch.bool)
    stores = {"trade": {k: [] for k in TRADE_FIELDS},
              "labour": {k: [] for k in SELF_FIELDS},
              "consume": {k: [] for k in SELF_FIELDS}}

    for r in range(R):
        world.begin_round()
        X[r], DEP[r], ALIVE[r] = world.x, world.deprivation, world.alive
        z, h = policy.embed(world.public_tokens(world.x), world.private_tokens(world.x), world.adjacency())
        u_before = world.utility()

        if production:
            b, s, d = _self_decision(world, policy, z, h, r, MODE_LABOUR, greedy, stores["labour"])
            world.labour(b, s, d.choice, d.amount)

        _trade_phase(world, policy, z, h, r, greedy, stores["trade"])

        if production:
            b, s, d = _self_decision(world, policy, z, h, r, MODE_CONSUME, greedy, stores["consume"])
            rewards[r, b, s] = world.consume(b, s, d.choice, d.amount)

        dones[r] = world.end_round()
        if not production:
            rewards[r] = world.utility() - u_before

    if production:
        X[R], DEP[R], ALIVE[R] = world.peek_next_observation(), world.deprivation, world.alive

    return Rollout(X=X, deprivation=DEP, alive=ALIVE, rewards=rewards, dones=dones, bootstrap=production,
                   trade=_cat(stores["trade"]), labour=_cat(stores["labour"]), consume=_cat(stores["consume"]))
