"""Stage B production economy: income, processing, consumption/death, flow tracking, invariants, GAE."""

import tempfile
import unittest

import torch

from agents.baselines import RandomPolicy, NoTradePolicy
from configs.presets import get_preset
from economy.goods import generate_goods
from economy.ledger import FlowTracker
from economy.production import sample_income, income_arrives, rounds_to_next_income, usefulness, max_output, raw_requirements
from economy.world import EconomyBatch
from learning.evaluate import evaluate_all
from learning.ppo import gae
from learning.rollout import collect_rollout
from learning.seeding import set_seed
from learning.trainer import train, load_checkpoint


def _world(seed=0, worlds=4):
    set_seed(seed)
    world = EconomyBatch(get_preset("B"), worlds)
    world.reset()
    return world


class TestIncome(unittest.TestCase):
    def test_schedule_and_coverage(self):
        set_seed(0)
        cfg = get_preset("B")
        goods = generate_goods(cfg.goods, 8)
        amount, period, phase = sample_income(cfg.production, goods, 12)
        raw = goods.tier == 0
        self.assertFalse((amount[~raw.unsqueeze(1).expand_as(amount)] > 0).any(), "income only in raw goods")
        self.assertTrue(((amount > 0).sum(1) > 0)[raw].all(), "every raw good has an earner")
        self.assertTrue(((period >= 1) & (period <= 4)).all() and (phase < period).all())
        arrivals = torch.stack([income_arrives(period, phase, t) for t in range(24)])
        torch.testing.assert_close(arrivals.float().sum(0), 24.0 / period.float(), atol=1.0, rtol=0.0)
        nxt = rounds_to_next_income(period, phase, 5)
        self.assertTrue(income_arrives(period, phase, 5 + nxt).all())

    def test_subsistence_is_feasible_with_surplus(self):
        world = _world(seed=11, worlds=64)
        req = raw_requirements(world.recipe, world.goods.tier)
        need = (world.gamma.unsqueeze(-1) * req.unsqueeze(1)).sum((1, 2))
        flow = (world.income / world.period.unsqueeze(-1).float()).sum(1)
        raw = world.goods.tier[0] == 0
        surplus = world.cfg.production.subsistence_surplus
        self.assertTrue((surplus * need[:, raw] <= flow[:, raw] + 1e-4).all())


class TestProcessing(unittest.TestCase):
    def test_leontief_output_and_inputs(self):
        world = _world()
        G = world.G
        k = int(torch.nonzero(world.goods.tier[0] == 1)[0])
        rec = world.recipe[0, k]
        world.x[0, 0] = 0.0
        world.x[0, 0, rec > 0] = 5.0
        x_before = world.x[0, 0].clone()
        q_max = float(max_output(x_before.view(1, 1, G), world.recipe[:1])[0, 0, k])
        world.labour(torch.tensor([0]), torch.tensor([0]), torch.tensor([G + 1 + k]), torch.tensor([0.5]))
        q = 0.5 * q_max
        expected = x_before - q * rec
        expected[k] += q
        torch.testing.assert_close(world.x[0, 0], expected, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(float(world.record.flows.produced[0, k]), q, places=5)

    def test_usefulness_is_transitive(self):
        alpha = torch.tensor([[[0.0, 0.0, 1.0]]])
        recipe = torch.zeros(1, 3, 3)
        recipe[0, 1, 0] = 1.0   # good 1 uses good 0
        recipe[0, 2, 1] = 1.0   # good 2 uses good 1
        self.assertTrue(usefulness(alpha, recipe).all())


class TestConsumptionAndDeath(unittest.TestCase):
    def test_death_after_deprivation(self):
        world = _world()
        D = world.cfg.market.death_after
        sub = (world.gamma[0, 0] > 0)
        self.assertTrue(sub.any())
        b, i = torch.tensor([0]), torch.tensor([0])
        for t in range(D):
            self.assertTrue(bool(world.alive[0, 0]))
            world.consume(b, i, torch.zeros(1, world.G, dtype=torch.long), torch.full((1, world.G), 0.5))
            world.end_round()
        self.assertFalse(bool(world.alive[0, 0]))
        self.assertTrue((world.x[0, 0] == 0).all())

    def test_consumption_utility(self):
        world = _world()
        b, i = torch.tensor([0]), torch.tensor([1])
        world.x[0, 1] = 4.0
        choice = (world.alpha[0, 1] > 0).long().unsqueeze(0)
        u = world.consume(b, i, choice, torch.full((1, world.G), 0.5))
        c = 2.0 * (world.alpha[0, 1] > 0).float()
        expected = (world.alpha[0, 1] * torch.log1p(torch.clamp(c - world.gamma[0, 1], min=0.0))).sum()
        self.assertAlmostEqual(float(u), float(expected), places=5)


class TestFlowTracker(unittest.TestCase):
    def test_pass_through_requires_useful_receipt(self):
        f = FlowTracker(1, 2, 2, 1)
        b, i = torch.tensor([0]), torch.tensor([0])
        f.trade(b, i, torch.tensor([[4.0, 0.0]]), torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([False]))
        self.assertAlmostEqual(float(f.pool[0, 0, 0]), 4.0)
        # passes 2 of 4 acquired units (holding 4) for something useful -> 2 units completed pass-through
        f.trade(b, i, torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 0.0]]), torch.tensor([[4.0, 0.0]]), torch.tensor([True]))
        self.assertAlmostEqual(float(f.passed[0, 0]), 2.0, places=5)
        # consumption removes pool proportionally
        f.remove(b, i, torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 1.0]]))
        self.assertAlmostEqual(float(f.pool[0, 0, 0]), 1.0, places=5)


class TestInvariants(unittest.TestCase):
    def test_random_play_keeps_invariants(self):
        set_seed(3)
        world = EconomyBatch(get_preset("B"), 8)
        roll = collect_rollout(world, RandomPolicy())
        x, flows = world.x, world.record.flows
        self.assertTrue((x >= 0).all())
        load = (x * world.weight.unsqueeze(1)).sum(-1)
        self.assertTrue((load <= world.capacity + 1e-3).all())
        self.assertTrue((flows.pool >= -1e-5).all() and (flows.pool <= x + 1e-4).all())
        self.assertTrue((x[~world.alive] == 0).all())
        alive = roll.alive.float()
        self.assertTrue((alive[1:] <= alive[:-1]).all(), "nobody comes back to life")
        self.assertTrue((roll.rewards[~roll.alive[:-1]] == 0).all(), "dead agents get no reward")
        self.assertEqual(roll.X.shape[0], world.R + 1)

    def test_no_trade_policy_never_trades(self):
        set_seed(4)
        world = EconomyBatch(get_preset("quick_B"), 4)
        collect_rollout(world, NoTradePolicy(RandomPolicy()))
        self.assertEqual(world.record.trades()["b"].numel(), 0)


class TestGAE(unittest.TestCase):
    def test_death_cuts_and_truncation_bootstraps(self):
        rewards = torch.ones(3, 1, 2)
        values = torch.zeros(3, 1, 2)
        dones = torch.zeros(3, 1, 2, dtype=torch.bool)
        dones[1, 0, 0] = True                       # agent 0 dies in round 1
        bootstrap = torch.full((1, 2), 10.0)
        adv, ret = gae(rewards, values, dones, bootstrap, gamma=0.5, lam=1.0)
        self.assertAlmostEqual(float(ret[0, 0, 0]), 1.0 + 0.5 * 1.0, places=5)          # nothing after death
        self.assertAlmostEqual(float(ret[2, 0, 1]), 1.0 + 0.5 * 10.0, places=5)         # bootstrapped


class TestTrainingSmokeB(unittest.TestCase):
    def test_train_and_evaluate(self):
        cfg = get_preset("quick_B")
        with tempfile.TemporaryDirectory() as run_dir:
            ckpt = train(cfg, run_dir, verbose=False)
            loaded_cfg, policy, _, _, _ = load_checkpoint(ckpt)
            report = evaluate_all(loaded_cfg, policy, batches=1)
            self.assertIn("no_trade_counterfactual", report)
            self.assertEqual(report["no_trade_counterfactual"]["trades"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestHarvestingStageC(unittest.TestCase):
    def test_access_coverage_and_feasibility(self):
        set_seed(6)
        world = EconomyBatch(get_preset("C"), 32)
        world.reset()
        raw = world.goods.tier[0] == 0
        access = world.harvest > 0
        self.assertFalse(access[..., ~raw].any(), "only raw goods can be harvested")
        self.assertTrue((access.sum(-1) == world.cfg.production.access_goods_per_agent).all())
        self.assertTrue(access.any(1)[:, raw].all(), "every raw good is accessible to someone")
        self.assertTrue((world.income == 0).all(), "no exogenous income in stage C")
        req = raw_requirements(world.recipe, world.goods.tier)
        need = (world.gamma.unsqueeze(-1) * req.unsqueeze(1)).sum((1, 2))
        even = (world.harvest / access.sum(-1, keepdim=True).clamp(min=1)).sum(1)
        self.assertTrue((world.cfg.production.subsistence_surplus * need[:, raw] <= even[:, raw] + 1e-4).all())

    def test_harvest_labour_adds_yield(self):
        set_seed(7)
        world = EconomyBatch(get_preset("C"), 2)
        world.reset()
        k = int(torch.nonzero(world.harvest[0, 0] > 0)[0])
        world.x[0, 0] = 0.0
        world.labour(torch.tensor([0]), torch.tensor([0]), torch.tensor([1 + k]), torch.tensor([0.5]))
        self.assertAlmostEqual(float(world.x[0, 0, k]), float(world.harvest[0, 0, k]), places=5)
        self.assertEqual(float(world.record.flows.labour_counts[0, 0, 1 + k]), 1.0)

    def test_no_specialization_control(self):
        set_seed(8)
        world = EconomyBatch(get_preset("C_no_specialization"), 4)
        world.reset()
        raw = world.goods.tier[0] == 0
        h = world.harvest[..., raw]
        self.assertTrue((h > 0).all())
        torch.testing.assert_close(h, h[:, :1].expand_as(h))      # identical yields across agents

    def test_train_smoke(self):
        with tempfile.TemporaryDirectory() as run_dir:
            train(get_preset("quick_C"), run_dir, verbose=False)
