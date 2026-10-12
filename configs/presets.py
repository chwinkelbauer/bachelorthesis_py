"""
Named experiment presets. Each preset is one experimental condition; see docs/FRAMEWORK.md, section "Presets".
"""

from typing import Callable, Dict

from configs.base import (
    ExperimentConfig, NetworkConfig, MarketConfig, PreferenceConfig, ProductionConfig, PPOConfig, LedgerConfig,
)


def stage_a() -> ExperimentConfig:
    """Stage A: pure exchange economy on a small-world network, private preferences."""
    return ExperimentConfig(name="A", stage="A")


def stage_a_complete() -> ExperimentConfig:
    """Stage A on a complete network (minimal search friction)."""
    return ExperimentConfig(name="A_complete", stage="A", network=NetworkConfig(kind="complete"))


def stage_a_full_info() -> ExperimentConfig:
    """Stage A with observable partner preferences."""
    return ExperimentConfig(name="A_full_info", stage="A", market=MarketConfig(full_information=True))


def stage_b() -> ExperimentConfig:
    """
    Stage B: production economy with lumpy raw-goods income (every P_i in {1..4} rounds), processing along the recipe
    DAG, learned consumption, Stone-Geary subsistence on one basic good and death after 5 deprived rounds.
    Rollouts of 24 rounds are truncated and bootstrapped (discount 0.97).
    """
    return ExperimentConfig(
        name="B", stage="B",
        preferences=PreferenceConfig(subsistence=True, subsistence_range=(0.2, 0.5)),
        production=ProductionConfig(income=True, processing=True, income_period_range=(1, 4)),
        market=MarketConfig(economy="production", rounds=24),
        ppo=PPOConfig(worlds=32, gamma=0.97),
    )


def stage_b_constant_income() -> ExperimentConfig:
    """Stage B control: income arrives every round (no timing mismatch, F5b off)."""
    cfg = stage_b()
    cfg.name = "B_constant_income"
    cfg.production.income_period_range = (1, 1)
    return cfg


def stage_c() -> ExperimentConfig:
    """
    Stage C: production as a choice. No exogenous income; each round an agent's labour is idle, harvesting one raw
    good it has access to (2 of 4 raw goods, skill ~ LogNormal(0, 0.5)), or processing. Otherwise as stage B.
    """
    cfg = stage_b()
    cfg.name, cfg.stage = "C", "C"
    cfg.production.income = False
    cfg.production.harvesting = True
    return cfg


def stage_c_no_specialization() -> ExperimentConfig:
    """Stage C control (F2 knock-out): every agent can harvest every raw good with equal skill."""
    cfg = stage_c()
    cfg.name = "C_no_specialization"
    cfg.production.access_goods_per_agent = cfg.goods.goods_per_tier[0]
    cfg.production.skill_sigma = 0.0
    return cfg


def stage_d() -> ExperimentConfig:
    """
    Stage D: stage C with a public trade ledger (F6). Trained with the ledger always on; the disturbance experiment
    evaluates C and D checkpoints with `--set ledger.switch_round=12` (ledger switched off / on mid-episode).
    """
    cfg = stage_c()
    cfg.name, cfg.stage = "D", "D"
    cfg.ledger = LedgerConfig(enabled=True)
    return cfg


def _shrink(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg.market.n_agents = 6
    cfg.market.rounds = 3
    cfg.ppo.worlds = 4
    cfg.ppo.iterations = 2
    cfg.ppo.minibatches = 2
    cfg.ppo.epochs = 1
    cfg.ppo.benchmark_every = 1
    cfg.model.hidden = 16
    cfg.model.attn_heads = 2
    cfg.model.gnn_layers = 1
    cfg.model.transformer_layers = 1
    return cfg


def quick_b() -> ExperimentConfig:
    """Tiny stage-B configuration for smoke tests."""
    cfg = _shrink(stage_b())
    cfg.name = "quick_B"
    return cfg


def quick_c() -> ExperimentConfig:
    """Tiny stage-C configuration for smoke tests."""
    cfg = _shrink(stage_c())
    cfg.name = "quick_C"
    return cfg


def quick_d() -> ExperimentConfig:
    """Tiny stage-D configuration for smoke tests."""
    cfg = _shrink(stage_d())
    cfg.name = "quick_D"
    return cfg


def quick() -> ExperimentConfig:
    """Tiny configuration for smoke tests."""
    cfg = ExperimentConfig(name="quick", stage="A")
    cfg.market.n_agents = 6
    cfg.market.rounds = 3
    cfg.ppo.worlds = 4
    cfg.ppo.iterations = 2
    cfg.ppo.minibatches = 2
    cfg.ppo.epochs = 1
    cfg.ppo.benchmark_every = 1
    cfg.model.hidden = 16
    cfg.model.attn_heads = 2
    cfg.model.gnn_layers = 1
    cfg.model.transformer_layers = 1
    return cfg


PRESETS: Dict[str, Callable[[], ExperimentConfig]] = {
    "A": stage_a,
    "A_complete": stage_a_complete,
    "A_full_info": stage_a_full_info,
    "B": stage_b,
    "B_constant_income": stage_b_constant_income,
    "C": stage_c,
    "C_no_specialization": stage_c_no_specialization,
    "D": stage_d,
    "quick": quick,
    "quick_B": quick_b,
    "quick_C": quick_c,
    "quick_D": quick_d,
}


def get_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name]()
