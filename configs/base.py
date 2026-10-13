"""
Configuration dataclasses for the money-emergence economy.

Every numeric default here is documented in docs/FRAMEWORK.md (section "Parameters").
If you change a default, update that document in the same change.
"""

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Tuple, Any, Dict


@dataclass
class GoodsConfig:
    """Random production graph (DAG) of goods. Tier 0 = raw resources."""
    goods_per_tier: Tuple[int, ...] = (4, 3, 2)
    weight_range: Tuple[float, float] = (0.2, 2.0)        # weight per unit, sampled log-uniformly
    spoilage_range: Tuple[float, float] = (0.0, 0.10)     # fraction lost per round, sampled uniformly
    raw_consumable_prob: float = 0.5                      # probability that a raw good has use value
    recipe_inputs: Tuple[int, int] = (1, 3)               # number of inputs per recipe (inclusive range)
    recipe_coef_range: Tuple[float, float] = (0.5, 2.0)   # input units per output unit

    @property
    def n_goods(self) -> int:
        return int(sum(self.goods_per_tier))


@dataclass
class PreferenceConfig:
    """Sparse Dirichlet log utility with optional Stone-Geary subsistence."""
    goods_valued: int = 3                                 # m: size of each agent's preference support
    dirichlet_beta: float = 1.0                           # concentration of the weights on the support
    subsistence: bool = False                             # Stone-Geary gamma > 0 (used from stage B on)
    subsistence_goods: int = 1                            # how many valued goods carry a subsistence level
    subsistence_range: Tuple[float, float] = (0.5, 1.5)   # gamma per round (consumption flow) from stage B on
    subsistence_max_tier: int = 1                         # basic needs: subsistence goods are drawn from tiers <= this
    goods_valued_range: Tuple[int, int] = (0, 0)          # (lo, hi): each agent values n ~ U{lo..hi} goods; (0, 0) = goods_valued
    tier_weights: Tuple[float, ...] = ()                  # mean utility weight multiplier per tier (last entry for higher tiers); () = off
    tier_weight_sigma: Tuple[float, ...] = ()             # log-normal spread of that multiplier per tier, per agent and good; () = fixed
    min_processed_valued: int = 0                         # with subsistence: at least this many valued goods are processed (tier >= 1)


@dataclass
class EndowmentConfig:
    """Stage A: one-off initial endowments (pure exchange economy)."""
    goods_per_agent: int = 2
    amount_range: Tuple[float, float] = (2.0, 10.0)
    consumable_only: bool = True                          # endow only goods that have use value for someone


@dataclass
class ProductionConfig:
    """Stages B-D: raw-goods income (B), harvesting (C) and processing along the recipe DAG."""
    income: bool = False                                  # stage B: exogenous raw-goods income
    income_goods_per_agent: int = 1                       # raw goods each agent earns (every raw good has an earner)
    income_flow_range: Tuple[float, float] = (1.0, 2.0)   # mean units per round, sampled per agent and good
    income_period_range: Tuple[int, int] = (1, 4)         # income arrives as a lump every P_i rounds; (1, 1) = constant flow
    subsistence_surplus: float = 2.0                      # per raw good: income flow >= this x aggregate subsistence need
    subsistence_buffer_rounds: float = 0.0                # starting reserve of the subsistence good, in rounds of gamma (0 = none)
    processing: bool = False                              # agents may turn inputs into higher-tier goods (Leontief)
    harvesting: bool = False                              # stage C: labour can harvest raw goods (only raw goods)
    access_goods_per_agent: int = 2                       # raw goods an agent can harvest (every raw good is accessible)
    skill_sigma: float = 0.5                              # skill_ir ~ LogNormal(0, sigma)
    max_tier: int = -1                                    # technology level: goods above this tier cannot be processed (-1 = all)


@dataclass
class LedgerConfig:
    """Stage D: public record of past trades (F6, "money is memory")."""
    enabled: bool = False                                 # ledger observable from round 0
    switch_round: int = -1                                # from this round on the state is flipped (-1 = never)


@dataclass
class NetworkConfig:
    """Trade network on which agents are matched."""
    kind: str = "small_world"                             # small_world | complete | ring
    degree: int = 4                                       # neighbours per agent (small_world, ring)
    rewire_prob: float = 0.1                              # Watts-Strogatz rewiring probability


@dataclass
class MarketConfig:
    economy: str = "exchange"                             # exchange (stage A) | production (stages B-D)
    n_agents: int = 12
    rounds: int = 8                                       # A: matching rounds per episode; B+: rollout length (truncation)
    capacity: float = 40.0                                # max carried weight per agent
    full_information: bool = False                        # partners' preferences observable
    spoilage: bool = True
    death_after: int = 5                                  # B+: consecutive rounds below subsistence until death
    death_penalty: float = 0.0                            # B+: one-off reward -p in the round an agent dies (0 = off)


@dataclass
class CurriculumConfig:
    """
    Learning sessions within one run. Each session is {"until": iteration, "set": [dotted overrides]} and applies to the
    iterations after the previous session up to and including `until`. Later iterations use the configuration itself
    (the full economy), which is also the only one ever evaluated. () = no sessions.
    """
    sessions: Tuple[Dict[str, Any], ...] = ()


@dataclass
class ModelConfig:
    hidden: int = 64
    gnn_layers: int = 2
    attn_heads: int = 4
    transformer_layers: int = 2


@dataclass
class PPOConfig:
    worlds: int = 64                                      # parallel economies per rollout
    iterations: int = 300
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    benchmark_every: int = 10                             # iterations between Walras/planner benchmarks
    checkpoint_every: int = 50


@dataclass
class MetricsConfig:
    no_use_alpha: float = 1e-6                            # alpha below this = good has no use value for the agent
    autarky_trade_rate: float = 0.02                      # trades per meeting below this = autarky
    indirect_share: float = 0.05                          # minimal value share of no-use acceptances
    excess_resale: float = 0.05                           # completed-resale rate above the random null
    excess_hhi: float = 0.15                              # concentration (HHI) of no-use acceptances above the null


@dataclass
class ExperimentConfig:
    name: str = "A"
    stage: str = "A"
    seed: int = 0
    goods: GoodsConfig = field(default_factory=GoodsConfig)
    preferences: PreferenceConfig = field(default_factory=PreferenceConfig)
    endowments: EndowmentConfig = field(default_factory=EndowmentConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, data)


def apply_overrides(cfg: ExperimentConfig, assignments) -> ExperimentConfig:
    """Applies dotted overrides such as "ledger.switch_round=12" or "production.income_period_range=[1,1]" (JSON values)."""
    import json
    for item in assignments or []:
        key, raw = item.split("=", 1)
        *path, leaf = key.split(".")
        target = cfg
        for part in path:
            target = getattr(target, part)
        current = getattr(target, leaf)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(current, tuple):
            value = tuple(value)
        elif isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(target, leaf, value)
    return cfg


def _from_dict(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default
        if is_dataclass(default):
            value = _from_dict(type(default), value)
        elif isinstance(default, tuple):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)
