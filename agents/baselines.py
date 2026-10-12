"""
Non-learning baselines and counterfactual wrappers with the same interface as PolicyNet.

RandomPolicy (null model): uniformly random allowed offers, responses, labour and consumption choices. It accepts
goods regardless of their use, so it produces no-use acceptances and pass-through by chance only.

ZICPolicy (zero intelligence, constrained; Gode & Sunder 1993): random offers, but a standing offer is accepted only
if it does not lower the agent's immediate utility (myopic individual rationality).

RandomTradeNull(policy): the policy's own labour and consumption, but random trading. Null model for production
economies: it isolates trade behaviour while keeping production and consumption realistic.

ZICTradeNull(policy): the policy's own labour and consumption, ZI-C trading. Second null for production economies.

NoTradePolicy(policy): the policy's own labour and consumption, trading switched off (empty offers, rejection).
Counterfactual for the gains from trade.

Note: ZI-C does accept goods it has no use for whenever the trade costs it nothing (e.g. swapping one useless good
for another) and may pass them on later by chance. It is therefore used as a second, stronger null (see
analysis/money_metrics.py), not as a strict "no indirect exchange" reference.
"""

import torch

from agents.policy import Decision, LabourDecision, ConsumeDecision, AMOUNT_EPS
from economy.observations import DU_CONTEXT_INDEX
from economy.trade import ACCEPT, COUNTER, REJECT, NONE


def _zeros_embedding(public_tokens):
    S, N = public_tokens.shape[:2]
    return torch.zeros(S, N, 1), torch.zeros(S, N, 1)


class RandomPolicy:
    def embed(self, public_tokens, private_tokens, adj):
        return _zeros_embedding(public_tokens)

    def decide(self, z_self, h_partner, tokens, context, dir_mask, resp_mask, actions=None, greedy=False) -> Decision:
        D, G = dir_mask.shape[:2]
        direction = torch.distributions.Categorical(probs=dir_mask.float()).sample()
        amount = torch.rand(D, G).clamp(AMOUNT_EPS, 1 - AMOUNT_EPS)
        response = torch.distributions.Categorical(probs=resp_mask.float()).sample()
        zeros = torch.zeros(D)
        return Decision(response, direction, amount, zeros, zeros)

    def labour(self, z_self, h_self, tokens, context, mask, actions=None, greedy=False) -> LabourDecision:
        D = mask.shape[0]
        choice = torch.distributions.Categorical(probs=mask.float()).sample()
        zeros = torch.zeros(D)
        return LabourDecision(choice, torch.rand(D).clamp(AMOUNT_EPS, 1 - AMOUNT_EPS), zeros, zeros)

    def consume(self, z_self, h_self, tokens, context, mask, actions=None, greedy=False) -> ConsumeDecision:
        D, G = mask.shape[:2]
        choice = torch.distributions.Categorical(probs=mask.float()).sample()
        zeros = torch.zeros(D)
        return ConsumeDecision(choice, torch.rand(D, G).clamp(AMOUNT_EPS, 1 - AMOUNT_EPS), zeros, zeros)


class ZICPolicy(RandomPolicy):
    def __init__(self, counter_prob: float = 0.5):
        self.counter_prob = counter_prob

    def decide(self, z_self, h_partner, tokens, context, dir_mask, resp_mask, actions=None, greedy=False) -> Decision:
        D, G = dir_mask.shape[:2]
        direction = torch.distributions.Categorical(probs=dir_mask.float()).sample()
        amount = torch.rand(D, G).clamp(AMOUNT_EPS, 1 - AMOUNT_EPS)
        acceptable = context[:, DU_CONTEXT_INDEX] >= 0.0
        counter = torch.rand(D) < self.counter_prob
        response = torch.where(acceptable, torch.full((D,), ACCEPT), torch.where(counter, torch.full((D,), COUNTER), torch.full((D,), REJECT)))
        # respect the turn mask: forced opening offer, no counter at the final turn
        response = torch.where(~resp_mask[torch.arange(D), response], torch.where(resp_mask[:, COUNTER], COUNTER, REJECT), response)
        zeros = torch.zeros(D)
        return Decision(response, direction, amount, zeros, zeros)


class _Wrapper:
    def __init__(self, policy):
        self.policy = policy

    def embed(self, public_tokens, private_tokens, adj):
        return self.policy.embed(public_tokens, private_tokens, adj)

    def labour(self, *args, **kwargs):
        return self.policy.labour(*args, **kwargs)

    def consume(self, *args, **kwargs):
        return self.policy.consume(*args, **kwargs)

    def eval(self):
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self


class RandomTradeNull(_Wrapper):
    def decide(self, z_self, h_partner, tokens, context, dir_mask, resp_mask, actions=None, greedy=False) -> Decision:
        return RandomPolicy().decide(z_self, h_partner, tokens, context, dir_mask, resp_mask)


class ZICTradeNull(_Wrapper):
    def decide(self, z_self, h_partner, tokens, context, dir_mask, resp_mask, actions=None, greedy=False) -> Decision:
        return ZICPolicy().decide(z_self, h_partner, tokens, context, dir_mask, resp_mask)


class NoTradePolicy(_Wrapper):
    def decide(self, z_self, h_partner, tokens, context, dir_mask, resp_mask, actions=None, greedy=False) -> Decision:
        D, G = dir_mask.shape[:2]
        response = torch.where(resp_mask[:, REJECT], torch.full((D,), REJECT), torch.full((D,), COUNTER))
        zeros = torch.zeros(D)
        return Decision(response, torch.full((D, G), NONE), torch.full((D, G), 0.5), zeros, zeros)
