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


class TestMemory(unittest.TestCase):
    def _cfg(self, name="quick_B"):
        cfg = get_preset(name)
        cfg.model.memory = True
        cfg.model.memory_size = 8
        return cfg

    def test_experience_bookkeeping(self):
        cfg = get_preset("B")
        set_seed(3)
        world = EconomyBatch(cfg, 4)
        ro = collect_rollout(world, RandomPolicy())
        e = torch.expm1(ro.experience)                                   # (S, B, N, G, 4) units of the last round
        self.assertTrue((e >= -1e-6).all())
        self.assertTrue(torch.allclose(e[..., 0].sum(2), e[..., 1].sum(2), atol=1e-3))   # received = given per world
        self.assertGreater(float(e[..., 2].sum()), 0.0)                    # partners asked for goods
        self.assertGreater(float(e[..., 3].sum()), 0.0)                    # partners offered goods
        self.assertEqual(float(e[0].abs().sum()), 0.0)                     # nothing remembered before the first round

    def test_train_resume_and_evaluate_with_memory(self):
        import csv
        from learning.evaluate import evaluate_all
        cfg = self._cfg()
        cfg.ppo.iterations = 4
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as split:
            train(cfg, straight, verbose=False)
            first = self._cfg()
            first.ppo.iterations = 3
            train(first, split, verbose=False)
            train(cfg, split, verbose=False, resume=True)
            _, p1, _, _, _ = load_checkpoint(os.path.join(straight, "checkpoint.pt"))
            loaded_cfg, p2, _, _, _ = load_checkpoint(os.path.join(split, "checkpoint.pt"))
            self.assertIsNotNone(p2.memory)
            for a, b in zip(p1.state_dict().values(), p2.state_dict().values()):
                self.assertTrue(torch.equal(a, b))
            report = evaluate_all(loaded_cfg, p2, batches=1)
            self.assertIn("no_trade_counterfactual", report)

    def test_memory_is_trained(self):
        from learning.ppo import ppo_update
        from learning.trainer import build_models
        cfg = self._cfg()
        set_seed(0)
        policy, critic, optimizer = build_models(cfg)
        before = {k: v.clone() for k, v in policy.memory.state_dict().items()}
        world = EconomyBatch(cfg, cfg.ppo.worlds)
        ppo_update(collect_rollout(world, policy), world, policy, critic, optimizer, cfg.ppo)
        changed = [not torch.equal(before[k], v) for k, v in policy.memory.state_dict().items()]
        self.assertTrue(any(changed))                                     # gradients flow through the GRU

    def test_warm_start_into_memory_model_is_function_preserving(self):
        from learning.trainer import build_models, load_compatible
        from economy.observations import PUBLIC_FEATURES, PRIVATE_FEATURES, DECISION_TOKEN_FEATURES, DECISION_CONTEXT_FEATURES
        from economy.trade import response_mask
        base = get_preset("quick_B")
        set_seed(1)
        p_old, c_old, _ = build_models(base)
        p_new, c_new, _ = build_models(self._cfg())
        load_compatible(p_new, p_old.state_dict())
        load_compatible(c_new, c_old.state_dict())
        S, N, G, D, M = 2, 4, 9, 5, 8
        pub, priv = torch.randn(S, N, G, PUBLIC_FEATURES), torch.randn(S, N, G, PRIVATE_FEATURES)
        adj = torch.ones(S, N, N, dtype=torch.bool)
        z0, h0 = p_old.embed(pub, priv, adj)
        z1, h1 = p_new.embed(pub, torch.cat([priv, torch.randn(S, N, G, M)], -1), adj)   # any memory content
        self.assertTrue(torch.allclose(z0, z1, atol=1e-6) and torch.allclose(h0, h1, atol=1e-6))
        tokens, context = torch.randn(D, G, DECISION_TOKEN_FEATURES), torch.randn(D, DECISION_CONTEXT_FEATURES)
        dmask = torch.ones(D, G, 3, dtype=torch.bool)
        rmask = response_mask(torch.ones(D, dtype=torch.long))
        a = p_old.decide(z0[0, :1].expand(D, -1), h0[0, :1].expand(D, -1), tokens, context, dmask, rmask, greedy=True)
        b = p_new.decide(z0[0, :1].expand(D, -1), h0[0, :1].expand(D, -1), torch.cat([tokens, torch.randn(D, G, M)], -1),
                         context, dmask, rmask, greedy=True)
        self.assertTrue(torch.equal(a.response, b.response) and torch.equal(a.direction, b.direction))
        self.assertTrue(torch.allclose(a.log_prob, b.log_prob, atol=1e-5))
        v0 = c_old(priv, adj)
        v1 = c_new(torch.cat([priv, torch.randn(S, N, G, 4)], -1), adj)
        self.assertTrue(torch.allclose(v0, v1, atol=1e-6))
