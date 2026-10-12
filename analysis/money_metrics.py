"""
Money metrics computed from the transaction ledger. All definitions are symmetric over goods: no good is
assumed to be money in advance. Definitions are stated in docs/FRAMEWORK.md, section "Measurement".

Stage A (no production): an agent has *no use* for good k iff alpha_ik = 0. Receiving such a good can only be
motivated by passing it on later, so a no-use acceptance is an act of indirect exchange by definition.
Quantities are valued at Walrasian prices p_k so that goods measured in different units are comparable.
"""

from typing import Dict

import numpy as np
import torch

from configs.base import MetricsConfig
from economy.ledger import EpisodeRecord
from economy.preferences import utility
from analysis.benchmarks import welfare_benchmarks


def _receipts(trades: dict):
    """Both sides of every trade as receipt rows: (b, r, agent, inflow (G), outflow (G), du)."""
    return dict(
        b=torch.cat([trades["b"], trades["b"]]),
        r=torch.cat([trades["r"], trades["r"]]),
        agent=torch.cat([trades["a"], trades["p"]]),
        inflow=torch.cat([trades["recv_a"], trades["give_a"]]),
        outflow=torch.cat([trades["give_a"], trades["recv_a"]]),
        du=torch.cat([trades["du_a"], trades["du_p"]]),
    )


def resale_volume(rec: EpisodeRecord, rows: dict, no_use: torch.Tensor, costly_only: bool = False) -> torch.Tensor:
    """
    Units of each good that were received without use and later passed on by the same agent *in exchange for a
    good the agent values* (a completed indirect exchange), (B, G).
    Per (agent, good) a pool of no-use receipts is kept; an outflow is counted as resale up to the pool size.
    costly_only: only receipts from trades in which the agent gave up a good it values enter the pool.
    """
    B, N, G = rec.alpha.shape
    pool = np.zeros((B, N, G))
    resold = np.zeros((B, G))
    order = np.argsort(rows["r"].numpy(), kind="stable")
    b, a = rows["b"].numpy(), rows["agent"].numpy()
    inflow, outflow, nu = rows["inflow"].numpy(), rows["outflow"].numpy(), no_use.numpy()
    # within a round each agent trades at most once, so per row: outflow uses the pool from earlier rounds
    for t in order:
        bt, at = b[t], a[t]
        gets_valued = float((inflow[t] * (1.0 - nu[bt, at])).sum() > 0)
        passed = np.minimum(outflow[t], pool[bt, at]) * gets_valued
        resold[bt] += passed
        pool[bt, at] -= passed
        costly = float((outflow[t] * (1.0 - nu[bt, at])).sum() > 0)
        pool[bt, at] += inflow[t] * nu[bt, at] * (costly if costly_only else 1.0)
    return torch.from_numpy(resold).float()


def compute_money_metrics(rec: EpisodeRecord, cfg: MetricsConfig) -> Dict[str, float]:
    trades = rec.trades()
    bench = welfare_benchmarks(rec.alpha, rec.gamma, rec.x0, rec.spoilage, rec.rounds)
    prices = bench["prices"]                                          # (B, G), mean 1 over traded goods
    B, N, G = rec.alpha.shape
    no_use = (rec.alpha < cfg.no_use_alpha).float()                    # (B, N, G)

    rows = _receipts(trades)
    n_trades = int(trades["b"].numel())
    out: Dict[str, float] = {"meetings": float(rec.meetings), "trades": float(n_trades)}
    out["trade_rate"] = n_trades / max(rec.meetings, 1)

    # --- indirect exchange -------------------------------------------------------------------------
    recv_value = torch.zeros(B, G)
    nu_value = torch.zeros(B, G)
    if n_trades:
        val = rows["inflow"] * prices[rows["b"]]
        recv_value.index_put_((rows["b"],), val, accumulate=True)
        nu_value.index_put_((rows["b"],), val * no_use[rows["b"], rows["agent"]], accumulate=True)
    resold_value = resale_volume(rec, rows, no_use) * prices
    total_recv, total_nu = float(recv_value.sum()), float(nu_value.sum())
    out["no_use_share"] = total_nu / total_recv if total_recv > 0 else 0.0
    out["resale_rate"] = float(resold_value.sum()) / total_nu if total_nu > 0 else 0.0

    # descriptive (sufficient, not necessary): no-use goods acquired by giving up a valued good
    costly_nu = 0.0
    if n_trades:
        gave_valued = ((rows["outflow"] * (1.0 - no_use[rows["b"], rows["agent"]])).sum(-1) > 0).float()
        costly_nu = float((rows["inflow"] * prices[rows["b"]] * no_use[rows["b"], rows["agent"]]).sum(-1).mul(gave_valued).sum())
    costly_resold = float((resale_volume(rec, rows, no_use, costly_only=True) * prices).sum())
    out["costly_no_use_share"] = costly_nu / total_recv if total_recv > 0 else 0.0
    out["costly_resale_rate"] = costly_resold / costly_nu if costly_nu > 0 else 0.0

    # --- concentration of indirect exchange on one good (per world) -------------------------------
    world_nu = nu_value.sum(-1)
    active = world_nu > 1e-9
    shares = nu_value / world_nu.clamp(min=1e-12).unsqueeze(-1)
    hhi = (shares ** 2).sum(-1)
    out["money_hhi"] = float(hhi[active].mean()) if active.any() else 0.0
    out["worlds_with_indirect_exchange"] = float(active.float().mean())

    # --- properties of the dominant medium of exchange (rank 0 = lowest in its world) -------------
    if active.any():
        top = shares.argmax(-1)[active]
        spoil_rank = rec.spoilage.argsort(-1).argsort(-1).float() / max(G - 1, 1)
        weight_rank = rec.weight.argsort(-1).argsort(-1).float() / max(G - 1, 1)
        out["money_good_spoilage_rank"] = float(spoil_rank[active].gather(1, top[:, None]).mean())
        out["money_good_weight_rank"] = float(weight_rank[active].gather(1, top[:, None]).mean())
    else:
        out["money_good_spoilage_rank"] = float("nan")
        out["money_good_weight_rank"] = float("nan")

    # --- motives ------------------------------------------------------------------------------------
    if n_trades:
        out["double_coincidence_share"] = float(((trades["du_a"] > 0) & (trades["du_p"] > 0)).float().mean())
        out["speculative_acceptance_share"] = float((rows["du"] < 0).float().mean())
    else:
        out["double_coincidence_share"] = 0.0
        out["speculative_acceptance_share"] = 0.0

    # --- welfare ------------------------------------------------------------------------------------
    u_final = utility(rec.x_final, rec.alpha, rec.gamma)
    u_no_trade = utility(bench["endowment_end"], rec.alpha, rec.gamma)
    W_final = u_final.sum(-1)
    gain_walras = (bench["W_walras"] - bench["W_no_trade"]).sum()
    gain_planner = (bench["W_planner"] - bench["W_no_trade"]).sum()
    gain = (W_final - bench["W_no_trade"]).sum()
    out["welfare_gain_per_agent"] = float(gain) / (B * N)
    out["efficiency_vs_walras"] = float(gain / gain_walras) if gain_walras > 0 else 0.0
    out["efficiency_vs_planner"] = float(gain / gain_planner) if gain_planner > 0 else 0.0
    out["ex_post_ir_violation"] = float((u_final < u_no_trade - 1e-6).float().mean())
    out["walras_residual"] = float(bench["walras_residual"].max())

    return out


STAGE_NAMES = ("autarky", "barter", "indirect_exchange", "money")


def classify_stage(m: Dict[str, float], null: Dict[str, float], cfg: MetricsConfig) -> str:
    """
    Stage of an economy, judged against a random-policy null run on the identical economies:
        autarky            almost no trade
        barter             no-use acceptance negligible, or completed indirect exchanges not above the null
        indirect_exchange  agents complete indirect exchanges (resale) more often than random agents do
        money              ... and concentrate them on one good markedly more than random agents do
    """
    if m["trade_rate"] < cfg.autarky_trade_rate:
        return "autarky"
    if m["no_use_share"] < cfg.indirect_share or m["resale_rate"] < null["resale_rate"] + cfg.excess_resale:
        return "barter"
    if m["money_hhi"] >= null["money_hhi"] + cfg.excess_hhi:
        return "money"
    return "indirect_exchange"


# ================================================================== production economies (stages B-D)

def compute_production_metrics(rec: EpisodeRecord, cfg: MetricsConfig) -> Dict[str, float]:
    """
    Fate-based money metrics (see economy/ledger.py, FlowTracker). Volumes of good k in world b are normalised by the
    good's total supply S_bk in that world, so that goods measured in different units are comparable without prices
    (input-only goods would get a Walras price of 0).
        pass_through_rate  sum_bk passed_bk / S_bk  /  sum_bk acquired_bk / S_bk
        money_hhi          per world: HHI of the shares of passed_bk / S_bk over goods (worlds with pass-through)
    """
    flows = rec.flows
    trades = rec.trades()
    B, N, G = rec.alpha.shape
    n_trades = int(trades["b"].numel())
    out: Dict[str, float] = {"meetings": float(rec.meetings), "trades": float(n_trades)}
    out["trade_rate"] = n_trades / max(rec.meetings, 1)

    supply = flows.supply.clamp(min=1e-9)
    acquired = flows.acquired / supply
    passed = flows.passed / supply
    out["pass_through_rate"] = float(passed.sum() / acquired.sum()) if float(acquired.sum()) > 0 else 0.0
    out["traded_share_of_supply"] = float(flows.acquired.sum() / flows.supply.sum().clamp(min=1e-9))

    world_passed = passed.sum(-1)
    active = world_passed > 1e-9
    shares = passed / world_passed.clamp(min=1e-12).unsqueeze(-1)
    hhi = (shares ** 2).sum(-1)
    out["money_hhi"] = float(hhi[active].mean()) if active.any() else 0.0
    out["worlds_with_pass_through"] = float(active.float().mean())
    if active.any():
        top = shares.argmax(-1)[active]
        for name, prop in (("spoilage", rec.spoilage), ("weight", rec.weight), ("tier", rec.goods_features[..., 2])):
            rank = prop.argsort(-1).argsort(-1).float() / max(G - 1, 1)
            out[f"money_good_{name}_rank"] = float(rank[active].gather(1, top[:, None]).mean())
    else:
        for name in ("spoilage", "weight", "tier"):
            out[f"money_good_{name}_rank"] = float("nan")

    if n_trades:
        out["double_coincidence_share"] = float(((trades["du_a"] > 0) & (trades["du_p"] > 0)).float().mean())
    else:
        out["double_coincidence_share"] = 0.0

    # division of labour: HHI of each agent's non-idle labour choices (1 = always the same activity)
    work = flows.labour_counts[..., 1:]
    total_work = work.sum(-1)
    worked = total_work > 0
    labour_hhi = ((work / total_work.clamp(min=1e-12).unsqueeze(-1)) ** 2).sum(-1)
    out["labour_specialization"] = float(labour_hhi[worked].mean()) if worked.any() else float("nan")
    all_labour = flows.labour_counts.sum((0, 1))
    n_labour = float(all_labour.sum().clamp(min=1.0))
    out["idle_share"] = float(all_labour[0]) / n_labour
    out["harvest_share"] = float(all_labour[1:G + 1].sum()) / n_labour
    out["process_share"] = float(all_labour[G + 1:].sum()) / n_labour

    sw = rec.ledger_switch_round
    if 0 <= sw < flows.passed_by_round.shape[0]:
        # disturbance analysis: the same measures before and after the ledger switch
        rounds_meetings = torch.tensor(rec.meetings_by_round, dtype=torch.float)
        r_trades = trades["r"]
        for label, sl in (("before", slice(0, sw)), ("after", slice(sw, None))):
            acq = (flows.acquired_by_round[sl].sum(0) / supply).sum()
            pas = (flows.passed_by_round[sl].sum(0) / supply).sum()
            out[f"pass_through_rate_{label}"] = float(pas / acq) if float(acq) > 0 else 0.0
            in_window = (r_trades >= sl.start) & (r_trades < (sl.stop if sl.stop is not None else 10 ** 9))
            out[f"trade_rate_{label}"] = float(in_window.sum()) / max(float(rounds_meetings[sl].sum()), 1.0)

    out["mean_flow_utility"] = float(flows.utility.sum() / flows.alive_rounds.sum().clamp(min=1.0))
    out["survival_rate"] = float(1.0 - flows.deaths.mean())
    out["consumed_share_of_supply"] = float(flows.consumed.sum() / flows.supply.sum().clamp(min=1e-9))
    return out


def classify_production_stage(m: Dict[str, float], null: Dict[str, float], cfg: MetricsConfig) -> str:
    """
    Same logic as the exchange economy, on the fate-based measures:
        autarky            trade_rate below threshold
        barter             completed pass-through not above the random-trade null
        indirect_exchange  pass-through above the null, not concentrated
        money              ... and concentrated on one good markedly more than under the null
    """
    if m["trade_rate"] < cfg.autarky_trade_rate:
        return "autarky"
    if m["pass_through_rate"] < null["pass_through_rate"] + cfg.excess_resale:
        return "barter"
    if m["money_hhi"] >= null["money_hhi"] + cfg.excess_hhi:
        return "money"
    return "indirect_exchange"


def compute_metrics(rec: EpisodeRecord, cfg: MetricsConfig) -> Dict[str, float]:
    return compute_production_metrics(rec, cfg) if rec.flows is not None else compute_money_metrics(rec, cfg)


def classify(m: Dict[str, float], null: Dict[str, float], cfg: MetricsConfig) -> str:
    return classify_production_stage(m, null, cfg) if "pass_through_rate" in m else classify_stage(m, null, cfg)
