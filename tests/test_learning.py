"""Learning pipeline: paired null reproduces economies, training/evaluation smoke test, checkpoint round trip."""

import os
import tempfile
import unittest

import torch

from agents.baselines import RandomPolicy
from configs.base import ExperimentConfig
from configs.presets import get_preset
from economy.world import EconomyBatch
from learning.evaluate import evaluate_policy
from learning.rollout import collect_rollout
from learning.seeding import set_seed
from learning.trainer import train, load_checkpoint


class TestPairedNull(unittest.TestCase):
    def test_same_rng_state_gives_same_economy(self):
        set_seed(5)
        world = EconomyBatch(get_preset("quick"), 4)
        state = torch.get_rng_state()
        collect_rollout(world, RandomPolicy())
        first = (world.alpha.clone(), world.record.x0.clone(), world.adj.clone(), world.goods.spoilage.clone())
        torch.set_rng_state(state)
        collect_rollout(world, RandomPolicy())
        second = (world.alpha, world.record.x0, world.adj, world.goods.spoilage)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))


class TestTrainingSmoke(unittest.TestCase):
    def test_train_evaluate_roundtrip(self):
        cfg = get_preset("quick")
        with tempfile.TemporaryDirectory() as run_dir:
            ckpt = train(cfg, run_dir, verbose=False)
            self.assertTrue(os.path.exists(os.path.join(run_dir, "metrics.csv")))
            loaded_cfg, policy, _, _, it = load_checkpoint(ckpt)
            self.assertEqual(it, cfg.ppo.iterations)
            self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())
            report = evaluate_policy(loaded_cfg, policy, batches=1)
            self.assertIn(report["stage"], ("autarky", "barter", "indirect_exchange", "money"))

    def test_config_roundtrip(self):
        cfg = get_preset("A_complete")
        self.assertEqual(ExperimentConfig.from_dict(cfg.to_dict()), cfg)


class TestResume(unittest.TestCase):
    def test_resume_is_bit_identical(self):
        import csv
        cfg = get_preset("quick")
        cfg.ppo.iterations = 4
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            train(cfg, straight, verbose=False)
            first = get_preset("quick")
            first.ppo.iterations = 2
            train(first, split, verbose=False)
            train(cfg, split, verbose=False, resume=True)
            _, p1, c1, _, it1 = load_checkpoint(os.path.join(straight, "checkpoint.pt"))
            _, p2, c2, _, it2 = load_checkpoint(os.path.join(split, "checkpoint.pt"))
            self.assertEqual(it1, it2)
            for a, b in zip(list(p1.state_dict().values()) + list(c1.state_dict().values()),
                            list(p2.state_dict().values()) + list(c2.state_dict().values())):
                self.assertTrue(torch.equal(a, b))

            def rows(d):
                with open(os.path.join(d, "metrics.csv"), newline="") as f:
                    return [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(f)]
            self.assertEqual(rows(straight), rows(split))

    def test_resume_old_checkpoint_without_rng(self):
        cfg = get_preset("quick")
        with tempfile.TemporaryDirectory() as run_dir:
            first = get_preset("quick")
            first.ppo.iterations = 1
            ckpt = train(first, run_dir, verbose=False)
            data = torch.load(ckpt, weights_only=False)
            del data["rng"], data["elapsed"]                     # format of checkpoints written before resume support
            torch.save(data, ckpt)
            train(cfg, run_dir, verbose=False, resume=True)
            _, _, _, _, it = load_checkpoint(ckpt)
            self.assertEqual(it, cfg.ppo.iterations)


if __name__ == "__main__":
    unittest.main()
