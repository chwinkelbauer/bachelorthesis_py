"""Invariants of the economy: goods graph, preferences, network/matching, offers, conservation of goods."""

import unittest

import torch

from agents.baselines import RandomPolicy
from configs.presets import get_preset
from economy.goods import generate_goods
from economy.network import build_networks, random_maximal_matching
from economy.preferences import sample_preferences
from economy.trade import decode_offer, GIVE, WANT
from economy.world import EconomyBatch
from learning.rollout import collect_rollout
from learning.seeding import set_seed


class TestGoods(unittest.TestCase):
    def test_recipes_use_lower_tiers_only(self):
        set_seed(0)
        cfg = get_preset("A").goods
        goods = generate_goods(cfg, 8)
        tiers = goods.tier[0]
        for out in range(goods.n_goods):
            inputs = torch.nonzero(goods.recipe[:, out].sum(0)).flatten()
            if tiers[out] == 0:
                self.assertEqual(inputs.numel(), 0)
            else:
                self.assertTrue((tiers[inputs] < tiers[out]).all())
                for b in range(8):
                    used = torch.nonzero(goods.recipe[b, out]).flatten()
                    self.assertTrue((tiers[used] == tiers[out] - 1).any(), "needs an input from the tier below")
        self.assertTrue(goods.consumable[goods.tier > 0].all())


class TestPreferences(unittest.TestCase):
    def test_sparse_dirichlet_support(self):
        set_seed(1)
        cfg = get_preset("A")
        goods = generate_goods(cfg.goods, 6)
        alpha, gamma = sample_preferences(cfg.preferences, goods, 10)
        torch.testing.assert_close(alpha.sum(-1), torch.ones(6, 10))
        self.assertTrue(((alpha > 0).sum(-1) == cfg.preferences.goods_valued).all())
        self.assertFalse((alpha > 0)[~goods.consumable.unsqueeze(1).expand_as(alpha)].any())
        self.assertTrue((gamma == 0).all())


class TestNetwork(unittest.TestCase):
    def test_matching_is_valid_and_maximal(self):
        set_seed(2)
        for kind in ("small_world", "complete", "ring"):
            cfg = get_preset("A").network
            cfg.kind = kind
            adj = build_networks(cfg, 16, 11)
            self.assertTrue((adj == adj.transpose(1, 2)).all())
            self.assertFalse(adj.diagonal(dim1=1, dim2=2).any())
            partner = random_maximal_matching(adj)
            for b in range(16):
                free = []
                for i in range(11):
                    j = int(partner[b, i])
                    if j >= 0:
                        self.assertEqual(int(partner[b, j]), i)
                        self.assertTrue(bool(adj[b, i, j]))
                    else:
                        free.append(i)
                for i in free:
                    for j in free:
                        self.assertFalse(bool(adj[b, i, j]), "matching must be maximal")


class TestOffers(unittest.TestCase):
    def test_decode_respects_stocks_and_capacity(self):
        set_seed(3)
        D, G = 200, 6
        x_s, x_p = torch.rand(D, G) * 10, torch.rand(D, G) * 10
        weight = torch.rand(G) + 0.2
        direction = torch.randint(0, 3, (D, G))
        amount = torch.rand(D, G)
        capacity = 30.0
        give, want = decode_offer(direction, amount, x_s, x_p, weight, capacity)
        self.assertTrue((give <= x_s + 1e-6).all() and (want <= x_p + 1e-6).all())
        self.assertTrue(((give > 0) & (want > 0)).sum() == 0, "a good is never given and wanted at once")
        self.assertTrue((give[direction != GIVE] == 0).all() and (want[direction != WANT] == 0).all())
        for x, inflow, outflow in ((x_s, want, give), (x_p, give, want)):
            before = (x * weight).sum(-1)
            after = ((x + inflow - outflow) * weight).sum(-1)
            self.assertTrue(((after <= capacity + 1e-4) | (after <= before + 1e-6)).all())


class TestConservation(unittest.TestCase):
    def test_ledger_replay_reproduces_final_holdings(self):
        set_seed(4)
        cfg = get_preset("A")
        world = EconomyBatch(cfg, 8)
        collect_rollout(world, RandomPolicy())
        rec = world.record
        t = rec.trades()
        self.assertGreater(t["b"].numel(), 0)
        x = rec.x0.clone()
        for r in range(rec.rounds):
            sel = t["r"] == r
            b, a, p = t["b"][sel], t["a"][sel], t["p"][sel]
            x[b, a] += t["recv_a"][sel] - t["give_a"][sel]
            x[b, p] += t["give_a"][sel] - t["recv_a"][sel]
            x *= (1 - rec.spoilage).unsqueeze(1)
        torch.testing.assert_close(x, rec.x_final, atol=1e-4, rtol=1e-4)
        self.assertTrue((rec.x_final >= 0).all())
        decay = ((1 - rec.spoilage) ** rec.rounds)
        torch.testing.assert_close(rec.x_final.sum(1), rec.x0.sum(1) * decay, atol=1e-3, rtol=1e-4)
        load = (rec.x_final * rec.weight.unsqueeze(1)).sum(-1)
        self.assertTrue((load <= cfg.market.capacity + 1e-3).all())


if __name__ == "__main__":
    unittest.main()
