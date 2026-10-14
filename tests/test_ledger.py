"""Stage D: public ledger records, observability, switch semantics, overrides, before/after metrics."""

import tempfile
import unittest

import torch

from agents.baselines import RandomPolicy
from analysis.money_metrics import compute_metrics
from configs.base import apply_overrides
from configs.presets import get_preset
from economy.observations import LEDGER_FEATURES
from economy.world import EconomyBatch
from learning.rollout import collect_rollout
from learning.seeding import set_seed
from learning.trainer import train

LEDGER_TOKEN_SLICE = slice(-LEDGER_FEATURES, None)


class TestLedgerRecords(unittest.TestCase):
    def test_records_balance(self):
        set_seed(9)
        world = EconomyBatch(get_preset("D"), 8)
        roll = collect_rollout(world, RandomPolicy())
        self.assertGreater(world.record.trades()["b"].numel(), 0)
        # conservation: current records plus the retired records of agents replaced by newcomers sum to zero
        balance = world.contribution.sum(1) + world.retired_contribution
        torch.testing.assert_close(balance, torch.zeros(8, world.G), atol=1e-3, rtol=0)
        torch.testing.assert_close(world.bilateral, -world.bilateral.transpose(1, 2), atol=1e-5, rtol=0)

    def test_visibility_follows_ledger_state(self):
        set_seed(10)
        for preset, expect_visible in (("C", False), ("D", True)):
            world = EconomyBatch(get_preset(preset), 8)
            roll = collect_rollout(world, RandomPolicy())
            ledger_tokens = roll.trade["tokens"][..., LEDGER_TOKEN_SLICE]
            flag = roll.trade["context"][:, -1]
            self.assertEqual(bool((ledger_tokens != 0).any()), expect_visible)
            self.assertTrue(bool((flag == (1.0 if expect_visible else 0.0)).all()))
            self.assertTrue((roll.labour["tokens"][..., LEDGER_TOKEN_SLICE] == 0).all(), "ledger only in trade mode")

    def test_switch_flips_mid_episode(self):
        set_seed(11)
        cfg = apply_overrides(get_preset("D"), ["ledger.switch_round=12"])
        self.assertEqual(cfg.ledger.switch_round, 12)
        world = EconomyBatch(cfg, 8)
        roll = collect_rollout(world, RandomPolicy())
        flag = roll.trade["context"][:, -1]
        r = roll.trade["r"]
        self.assertTrue((flag[r < 12] == 1).all() and (flag[r >= 12] == 0).all())
        m = compute_metrics(world.record, cfg.metrics)
        for key in ("pass_through_rate_before", "pass_through_rate_after", "trade_rate_before", "trade_rate_after"):
            self.assertIn(key, m)


class TestOverrides(unittest.TestCase):
    def test_types(self):
        cfg = apply_overrides(get_preset("B"), ["production.income_period_range=[1,1]", "market.full_information=true",
                                               "goods.raw_consumable_prob=1", "network.kind=complete"])
        self.assertEqual(cfg.production.income_period_range, (1, 1))
        self.assertIs(cfg.market.full_information, True)
        self.assertIsInstance(cfg.goods.raw_consumable_prob, float)
        self.assertEqual(cfg.network.kind, "complete")


class TestTrainingSmokeD(unittest.TestCase):
    def test_train(self):
        with tempfile.TemporaryDirectory() as run_dir:
            train(get_preset("quick_D"), run_dir, verbose=False)


if __name__ == "__main__":
    unittest.main()
