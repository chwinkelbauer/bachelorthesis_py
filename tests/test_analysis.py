"""Correctness of benchmarks, money metrics and change-point detection."""

import unittest

import numpy as np
import torch

from analysis.benchmarks import demand, walras_equilibrium, planner_allocation
from analysis.changepoints import binary_segmentation
from analysis.money_metrics import compute_money_metrics, classify_stage
from configs.base import MetricsConfig
from configs.presets import get_preset
from economy.ledger import EpisodeRecord
from economy.preferences import utility
from economy.world import EconomyBatch
from learning.seeding import set_seed


def _economy(seed=0, worlds=16):
    set_seed(seed)
    world = EconomyBatch(get_preset("A"), worlds)
    world.reset()
    return world


class TestBenchmarks(unittest.TestCase):
    def test_demand_kkt_and_budget(self):
        set_seed(0)
        alpha = torch.distributions.Dirichlet(torch.ones(5)).sample((3, 4))
        prices = torch.rand(3, 5) + 0.1
        wealth = torch.rand(3, 4) * 10
        x = demand(alpha, prices, wealth)
        torch.testing.assert_close((x * prices.unsqueeze(1)).sum(-1), wealth, atol=1e-4, rtol=1e-4)
        bang = alpha / (1 + x) / prices.unsqueeze(1)              # marginal utility per unit of money
        lam = (bang * (x > 1e-6)).amax(-1, keepdim=True)
        self.assertTrue(((bang - lam).abs()[x > 1e-6] < 1e-4).all(), "equal bang-for-buck on purchased goods")
        self.assertTrue((bang[x <= 1e-6] <= lam.expand_as(bang)[x <= 1e-6] + 1e-4).all())

    def test_walras_clears_markets_and_planner_dominates(self):
        world = _economy()
        e = world.record.x0
        we = walras_equilibrium(world.alpha, e)
        self.assertLess(float(we["residual"].max()), 1e-3)
        wealth = (e * we["prices"].unsqueeze(1)).sum(-1)
        spend = (we["allocation"] * we["prices"].unsqueeze(1)).sum(-1)
        torch.testing.assert_close(spend, wealth, atol=1e-3, rtol=1e-3)
        plan = planner_allocation(world.alpha, e)
        torch.testing.assert_close(plan.sum(1), e.sum(1), atol=1e-3, rtol=1e-3)
        W_w = utility(we["allocation"], world.alpha, world.gamma).sum(-1)
        W_p = utility(plan, world.alpha, world.gamma).sum(-1)
        W_0 = utility(e, world.alpha, world.gamma).sum(-1)
        self.assertTrue((W_p >= W_w - 1e-4).all())
        self.assertTrue((W_w >= W_0 - 1e-4).all())
        u_w, u_0 = utility(we["allocation"], world.alpha, world.gamma), utility(e, world.alpha, world.gamma)
        self.assertTrue((u_w >= u_0 - 1e-4).all(), "Walras allocation is individually rational")


class TestMoneyMetrics(unittest.TestCase):
    def _record(self, recv_goods_valued: bool):
        """3 agents, 3 goods, no double coincidence: agent 0 values good 0 and holds good 1, agent 1 values good 1
        and holds good 2, agent 2 values good 2 and holds good 0. The only route for agent 0 is indirect:
        accept good 2 (no use) from agent 1, then pass it on to agent 2 for good 0."""
        alpha = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        x0 = torch.tensor([[[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]]])
        rec = EpisodeRecord(alpha=alpha, gamma=torch.zeros_like(alpha), x0=x0, x_final=x0.clone(),
                            weight=torch.ones(1, 3), spoilage=torch.zeros(1, 3),
                            goods_features=torch.zeros(1, 3, 5), rounds=2, meetings=2)
        world, agent0, agent1, agent2 = torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), torch.tensor([2])
        # round 0: agent 0 gives good 1 to agent 1 and receives good 2, which it has no use for
        rec.add(world, 0, agent0, agent1, torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.0, 2.0, 0.0]]),
                torch.tensor([-0.1]), torch.tensor([0.5]), torch.tensor([1]))
        # round 1: agent 0 passes good 2 to agent 2, receiving good 0 (valued) or nothing (a gift)
        received = torch.tensor([[2.0, 0.0, 0.0]]) if recv_goods_valued else torch.tensor([[0.0, 0.0, 0.0]])
        rec.add(world, 1, agent0, agent2, received, torch.tensor([[0.0, 0.0, 2.0]]),
                torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([1]))
        return rec

    def test_completed_indirect_exchange_is_resale(self):
        m = compute_money_metrics(self._record(True), MetricsConfig())
        self.assertAlmostEqual(m["resale_rate"], 1.0, places=5)
        self.assertGreater(m["no_use_share"], 0.0)
        self.assertAlmostEqual(m["money_hhi"], 1.0, places=5)

    def test_free_swap_is_resale_but_not_costly(self):
        # agent 0 gave up good 1, which it does not value: Kiyotaki-Wright indirect exchange, but not costly
        m = compute_money_metrics(self._record(True), MetricsConfig())
        self.assertAlmostEqual(m["resale_rate"], 1.0, places=5)
        self.assertAlmostEqual(m["costly_resale_rate"], 0.0, places=5)
        self.assertAlmostEqual(m["costly_no_use_share"], 0.0, places=5)

    def test_stronger_null_is_elementwise_max(self):
        from learning.evaluate import stronger_null
        s = stronger_null({"random": {"a": 0.1, "b": 0.5, "stage": "x"}, "zic": {"a": 0.3, "b": 0.2, "stage": "y"}})
        self.assertEqual(s, {"a": 0.3, "b": 0.5})

    def test_gift_is_not_resale(self):
        m = compute_money_metrics(self._record(False), MetricsConfig())
        self.assertAlmostEqual(m["resale_rate"], 0.0, places=5)

    def test_stage_classification_is_null_referenced(self):
        cfg = MetricsConfig()
        base = dict(trade_rate=0.5, no_use_share=0.3, resale_rate=0.4, money_hhi=0.8)
        self.assertEqual(classify_stage(base, dict(resale_rate=0.1, money_hhi=0.3), cfg), "money")
        self.assertEqual(classify_stage(base, dict(resale_rate=0.1, money_hhi=0.7), cfg), "indirect_exchange")
        self.assertEqual(classify_stage(base, dict(resale_rate=0.4, money_hhi=0.3), cfg), "barter")
        self.assertEqual(classify_stage(dict(base, trade_rate=0.0), dict(resale_rate=0.0, money_hhi=0.0), cfg), "autarky")


class TestChangePoints(unittest.TestCase):
    def test_detects_level_shift(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([rng.normal(0.0, 0.05, 40), rng.normal(1.0, 0.05, 40)])
        self.assertEqual(binary_segmentation(y), [40])

    def test_no_change_in_noise(self):
        rng = np.random.default_rng(1)
        self.assertEqual(binary_segmentation(rng.normal(0.0, 1.0, 80)), [])


if __name__ == "__main__":
    unittest.main()


class TestSummarize(unittest.TestCase):
    def test_cross_run_summary(self):
        import json, os, tempfile
        from scripts.summarize import load_runs, run_table, aggregate_table
        def write(path, text):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        with tempfile.TemporaryDirectory() as root:
            for seed, hhi in ((0, 0.30), (1, 0.38)):
                d = os.path.join(root, f"A_s{seed}")
                os.makedirs(d)
                write(os.path.join(d, "config.json"),
                      json.dumps({"name": "A", "market": {"economy": "exchange"}, "ppo": {"iterations": 300}}))
                write(os.path.join(d, "metrics.csv"), "iteration,seconds\n1,1.0\n300,9.0\n")
                m = {"efficiency_vs_walras": 0.5, "resale_rate": 0.4, "null_resale_rate": 0.2, "costly_resale_rate": 0.4,
                     "money_hhi": hhi, "null_money_hhi": 0.25, "stage": "indirect_exchange", "stage_counts": {"indirect_exchange": 4}}
                write(os.path.join(d, "evaluation.json"), json.dumps({"learned_stochastic": m}))
            os.makedirs(os.path.join(root, "B_s0"))
            write(os.path.join(root, "B_s0", "config.json"),
                  json.dumps({"name": "B", "market": {"economy": "production"}, "ppo": {"iterations": 300}}))
            runs = load_runs(root)
            self.assertEqual([r["run"] for r in runs], ["A_s0", "A_s1", "B_s0"])
            self.assertIn("partial", run_table(runs, "production"))
            agg = aggregate_table(runs)
            self.assertIn("| A | 0, 1 | HHI |", agg)
            self.assertIn("+0.050 / +0.130", agg)
