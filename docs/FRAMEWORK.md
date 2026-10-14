# Money from Scratch: Framework Specification (living document)

> **This document describes the current state of the project.** Every change to the code, the model, the
> parameters or the experimental design must update the affected sections and add a line to the
> [Changelog](#13-changelog) in the same change. If code and this document disagree, that is a bug.
>
> Last updated: 2026-10-14 · Implemented stages: **A, B, C, D** · Experiments: stage A (3 seeds + 1 partial); stage B in three designs (B, B2, B3 + perishability treatment, 2 seeds): barter, no money; C and D not yet run

---

## 1. Research question and contribution

**Question.** Under which minimal conditions do learning agents with heterogeneous, agent-specific preferences
*invent* money (a commonly accepted medium of exchange), and which condition causes it?

**Position.**
- **Built-in money in existing agentic economies:** AI Economist (Zheng et al. 2022) has a fixed coin, and LLM economies (e.g. EconAgent, Li et al. 2024) have given money and prices. In both, money is assumed.
- **Kiyotaki–Wright simulations with learning agents:** Marimon, McGrattan & Sargent (1990) and the lab experiments of Duffy & Ochs (1999) use fixed types and fixed specialisation, with no production.
- **New Monetarist economics:** money is *essential* only under frictions: a missing double coincidence of wants, limited commitment, and missing record keeping (Kocherlakota 1998, "money is memory"; Lagos & Wright 2005; Wallace's dictum).

**Contribution.** This is a multi-agent reinforcement learning economy in which **no good is designated as money and
no reward mentions money or trade**:
- Production, consumption and trade are learned.
- Money is identified *functionally*, symmetrically over all goods.
- Its emergence is tested against null models and theoretical benchmarks.
- Its causes are isolated by switching frictions on and off.

**Why reinforcement learning agents, not LLM agents.** LLMs know about money from their training data, so their
"invention" of money would be recall. RL agents start tabula rasa.

---

## 2. Implementation status

| Component | Status | Where |
|---|---|---|
| Random goods space (tiers, properties, recipe DAG) | implemented | `economy/goods.py` |
| Sparse Dirichlet log preferences + Stone–Geary subsistence (basic goods); variable support size, random tier weights, guaranteed processed good | implemented | `economy/preferences.py` |
| Stage A endowments | implemented | `economy/endowments.py` |
| Stage B income (lumpy / constant), feasibility scaling, Leontief processing, usefulness | implemented | `economy/production.py` |
| Consumption, deprivation, death | implemented | `economy/world.py` |
| Trade network + random maximal matching (living agents) | implemented | `economy/network.py` |
| Offer encoding, capacity scaling | implemented | `economy/trade.py` |
| Alternating-offer protocol (K = 2 offers); labour / trade / consumption phases | implemented | `learning/rollout.py` |
| Batched world (exchange and production economy), ledger, flow tracker | implemented | `economy/world.py`, `economy/ledger.py` |
| Unified observations (identical layout in all stages) | implemented | `economy/observations.py` |
| Policy (GATv2 + goods transformer, trade / labour / consumption heads) and centralised critic | implemented | `agents/policy.py`, `agents/layers.py` |
| MAPPO with composite actions, deaths, truncation bootstrap | implemented | `learning/ppo.py` |
| Baselines: random null, ZI-C, random-trade null, no-trade counterfactual | implemented | `agents/baselines.py` |
| Walras and planner benchmarks (exchange economy) | implemented | `analysis/benchmarks.py` |
| Money metrics: no-use based (A), fate-based (B+); null-referenced stage classification | implemented | `analysis/money_metrics.py` |
| Change-point detection, IQM / bootstrap CI | implemented | `analysis/changepoints.py`, `analysis/stats.py` |
| Training / evaluation CLI | implemented | `scripts/train.py`, `scripts/evaluate.py` |
| Run report: curves vs. paired null, stage timeline, change points, summary (also for interrupted runs) | implemented | `scripts/report.py` |
| Cross-run summary: table of all runs + IQM/bootstrap CI across seeds per condition | implemented | `scripts/summarize.py` |
| Agent memory: per-good GRU over rounds, fed with own trade experience (`model.memory`, off by default) | implemented | `agents/policy.py`, `economy/world.py`, `learning/rollout.py`, `learning/ppo.py` |
| Stage C: harvesting raw goods (access, skills), division-of-labour metrics | implemented | `economy/production.py`, `economy/world.py` |
| Stage D: public ledger (contributions, bilateral balances), on/off switch, before/after metrics | implemented | `economy/world.py`, `economy/observations.py` |
| Config overrides from the command line (`--set key=value`) | implemented | `configs/base.py`, `scripts/` |
| Resumable training (`--resume`; atomic checkpoints with optimizer, iteration, elapsed time, RNG states) | implemented | `learning/trainer.py`, `scripts/train.py` |
| Curriculum: warm start between stages (`--init`) and learning sessions within a run (technology level + death-threshold ramp) | implemented | `learning/trainer.py`, `configs/presets.py`, `scripts/train.py` |
| Experiment queue runner (unattended nights: parallel jobs, barriers, logs, stop file) | implemented | `scripts/run_queue.py`, `configs/night_queue.txt` |
| Multi-seed sweep and report | partial: queue runner + `scripts/summarize.py` | — |
| Benchmark for production economies (dynamic planner / steady state) | **planned** | — |

---

## 3. Economy model

Notation: worlds b = 1..B (independent, simulated in parallel), agents i = 1..N, goods k = 1..G, rounds
t = 0..R−1. All randomness is redrawn for every episode, so no convention can attach to a fixed good.

### 3.1 Goods space
- **Tiers:** goods are arranged in tiers τ_k ∈ {0, 1, …, L}. Tier 0 consists of *raw resources*. Default: 4 + 3 + 2 = 9 goods.
- **Properties** (no labels), drawn per world:
  - weight w_k ~ LogUniform[0.2, 2.0]
  - spoilage δ_k ~ U[0, 0.10] per round
  - spoilage treatment `B_perishable` (factor F4): δ_k ~ U[0, 0.25]
  - consumable flag: always true for τ_k ≥ 1; true with probability 0.5 for raw goods (presets B–D: probability 1, so every raw good can be a subsistence good; with 0.5, 6% of worlds had no consumable raw good)
- **Recipe DAG:** a good k with τ_k ≥ 1 has 1–3 inputs from lower tiers, at least one of them from tier τ_k − 1. Its input coefficients are a_kj ~ U[0.5, 2] units of j per unit of k.
- **Property feature vector** (per good token): (ln w_k, δ_k, τ_k / L, 1[τ_k = 0], consumable_k).

### 3.2 Preferences (agent-specific, randomised)

$$U_i(x) = \sum_k \alpha_{ik}\,\ln\!\big(1 + \max(x_k - \gamma_{ik}, 0)\big)$$

In stage A the argument x is *holdings*. From stage B on it is the *consumption flow* c_t of the round.

- **Sparse support:** each agent values m_i consumable goods S_i: exactly m = 3 by default (stage A), or a per-agent draw m_i ~ U{lo..hi} (`goods_valued_range`; presets B–D: 2–5). On S_i, d_i ~ Dirichlet(β = 1); elsewhere α_ik = 0 *exactly*.
- **Why sparse:** α_ik = 0 makes "agent i has no use for good k" exact. The support size controls how likely a double coincidence of wants is (the Jevons factor).
- **Stone–Geary subsistence** (stage B on): `subsistence_goods` = 1 good of S_i is drawn from the **basic goods** (consumable, tier ≤ `subsistence_max_tier`) and gets γ_ik ~ U[0.2, 0.5] per round. Presets B–D: `subsistence_max_tier` = 0, so **survival needs only raw goods**. In stage A, γ = 0.
- **Basic needs vs. higher wants (presets B–D):** utility weights are scaled by random, tier-dependent multipliers:
  $$\alpha_{ik} = \frac{d_{ik}\,w_{ik}}{\sum_j d_{ij}\,w_{ij}},\qquad w_{ik} = \bar w_{\tau_k}\,\exp\!\big(s_{\tau_k} z_{ik} - s_{\tau_k}^2/2\big),\quad z_{ik}\sim N(0,1)$$
  with means w̄ = (0.2, 1, 2) and log-spreads s = (0.25, 0.75, 1.0) for tiers 0, 1, 2 (`tier_weights`, `tier_weight_sigma`). Raw goods keep agents alive but give little utility; processed goods are **high-utility, high-variance** goods.
- **Optional:** `min_processed_valued` = k forces every agent to value at least k processed goods. It is implemented and tested but **off in all presets** (user decision 2026-10-13). With it off, 12% of agents value no processed good; renormalisation then gives their raw goods the full weight.
- **Calibration check** (500 worlds × 12 agents, preset B):
  - The preferences remain fully random: 346 distinct support patterns, no two agents with identical weights, and every good index is valued about equally often within its tier.
  - Weights: subsistence good mean 0.19 (p10–p90 0.01–0.52); tier 1 mean 0.39 (0.04–0.85); tier 2 mean 0.46 (0.05–0.91).
  - Non-subsistence goods carry on average 82% of an agent's utility weight (p10 48%). Only 3% of agents put less than 20% on them.
  - The subsistence good outweighs every processed good for 23% of agents, including the 12% who value none (heterogeneity, not a rule).
  - Utility per embodied raw unit (one unit above need) has median 0.06 for raw goods, 0.09 for tier 1 and 0.065 for tier 2, so processing pays.
  - **Bare survival gives no utility:** consuming exactly γ yields 0. One extra unit of the subsistence good yields 0.13 on average; one extra unit of every valued processed good yields 0.45.
- With all three options off (the defaults), sampling is bit-identical to earlier versions (checked against the committed code).
- **Why log/Cobb–Douglas:** the exchange economy has a unique, computable Walrasian equilibrium (gross substitutes, §5.2). CES with a random σ_i is a planned robustness check.
- **Shared policy:** α_i and γ_i are part of agent i's own observation (a preference-conditioned shared policy).

### 3.3 Stage A: pure exchange economy
- **Endowments:** each agent receives `goods_per_agent` = 2 distinct consumable goods, with amounts ~ U[2, 10], scaled down if the weight exceeds 80% of capacity.
- **No** income, production or consumption during the episode. The episode has R = 8 rounds, and its horizon is known (round/(R−1) is observed).
- **Round:** matching → negotiation → execution → spoilage.
- **Reward:** r_{i,t} = U_i(x_{i,t+1}) − U_i(x_{i,t}). This is potential-based and telescopes to the final utility of holdings.

### 3.4 Stage B: production economy with income
**Income (lumpy):**
- Each agent earns `income_goods_per_agent` = 1 raw good at a mean flow f_ik ~ U[1, 2] per round.
- Every raw good has at least one earner.
- The income is paid as a **lump** f_ik·P_i every P_i rounds, with P_i ~ U{1..4} and a random phase.
- Agents start with one lump and a **reserve of their subsistence good** worth `subsistence_buffer_rounds` = 10 rounds of γ (capacity-limited). Without it, 89% of agents hold nothing they can eat and must complete the right trade within 5 rounds. Untrained agents never manage that, so only self-sufficient agents survived: a learning trap, not an economic result (§11).
- **Control:** constant flow (P_i = 1), preset `B_constant_income`.
- This creates the *timing mismatch* needed for a store-of-value motive (factor F5b).

**Feasibility:**
- raw_requirements gives the raw units embodied in one unit of each good (recursive Leontief).
- Per world and raw good, income is scaled up until flow ≥ κ · Σ_i γ_i · (raw requirement of i's subsistence good), with **κ = 2** (`subsistence_surplus`).
- Without this, about 57% of random worlds could not feed everybody even in principle (measured over 200 worlds), and deaths would reflect built-in scarcity rather than behaviour.

**Processing (Leontief):**
- One unit of good k uses a_kj units of every input j.
- The maximal output is q_max,k = min_{j: a_kj>0} x_j / a_kj.
- The agent chooses a good and a fraction a ∈ (0,1): it produces q = a·q_max,k, capacity-limited, and the inputs are used up.

**Consumption:**
- Per good, the agent chooses keep or consume, and a fraction a ∈ (0,1) of its holding (c_k = a·x_k).
- Consuming goods with α_ik = 0 is masked (it is a dominated action).
- **Flow utility:** u_i(c) = Σ_k α_ik ln(1 + max(c_k − γ_ik, 0)) (default), or with **continuous subsistence** (`preferences.continuous_subsistence`, presets B–D since 2026-10-14) u_i(c) = Σ_k α_ik ln(1 + c_k − γ_ik), which requires γ < 1.
  - Why: with the kink, eating below the need is as good as eating the need (both 0), so skipping meals is free until death; agents learned to fast up to 4 rounds and eat in lumps (feast-and-fast, §11). For the B preset preferences, feast-and-fast beats steady eating for 100% of agents with the kink and for 0% with the continuous form (mean −0.17 per 5 rounds).
  - A fully hungry round now costs α ln(1 − γ): mean −0.085 (range 0 to −0.66), against typical flow utilities of 0.15–0.18. Goods without a need (γ = 0) are unaffected, so stage A is unchanged. By concavity, steady consumption beats lumpy consumption, and eating mistakes are penalised in the same round (dense feedback) instead of only by a later death.
  - Flow utilities of runs with continuous subsistence include hunger costs and are not comparable to earlier runs; compare each with its own no-trade counterfactual.

**Deprivation and death:**
- **Hunger debt** (`market.hunger_debt`, presets B–D since 2026-10-14): the agent carries a debt d_i in *rounds of missing food*, d_i ← max(0, d_i + 1 − c_ik / γ_ik) (mean over its subsistence goods; one in all presets). Eating half the need adds 0.5, eating nothing adds 1, eating more than the need repays debt. The agent dies when d_i ≥ `death_after`. The private observation shows d_i / `death_after`.
  - Why (2026-10-14): the old rule counted a round as *deprived* if any subsistence good was consumed below γ (eating 99% of the need = eating nothing; 10–18% of deprived rounds in the B3 policies were such partial meals) and reset the counter after one adequate meal (chronic undernutrition never killed). Hunger debt is continuous and cumulative, like an energy deficit.
- **Old rule** (default, `hunger_debt` off): a round is *deprived* if any subsistence good is consumed below γ_ik; the counter resets after a non-deprived round.
- When the debt (or the consecutive count) reaches `death_after` = 5, the agent dies: it leaves the matching, its holdings perish, and it receives no further reward. During the learning sessions (§6.1) the threshold is 20, then 10.
- **Death penalty (optional, `market.death_penalty` = p, default 0):** a one-off reward −p in the round an agent dies. Without it, dying costs only the lost future flow utility (≈ 33 × the mean flow utility at γ_RL = 0.97), which untrained agents barely experience. It is a one-off cost, not a per-round bonus for being alive, so bare survival stays worth 0. It refers to no good and no trade, and the money metrics do not use rewards. Risk: a large p could induce precautionary hoarding of the subsistence good (autarky, or a raw good that looks like money out of fear of death), so p stays small and identical across conditions. **Status:** A/B test (B2, §12): p = 0.5 led the early learning but then drifted into hoarding (survival 0.38 vs. 0.46 without). **Presets B–D now use p = 0.5** (user decision 2026-10-13, "always have a penalty for now"), together with a stronger entropy bonus (0.03, §4.3) against the policy freezing that accompanied the drift.

**Technology level** (`production.max_tier`, default −1 = all tiers): goods above this tier cannot be produced (their recipe rows are zero, so processing is masked). It is used only by the learning sessions (§6.1).

**Newcomers** (`market.respawn_after` = k, presets B–D: 2; −1 = off; since 2026-10-14):
- A slot that has been empty for k rounds gets a newcomer at the start of the next round (begin of round, before income).
- The newcomer keeps the slot's **position**: network node, income good, income size and schedule (stage B), harvest access and skills (stage C), like someone taking over a farm. So every raw good keeps an earner and the feasibility scaling stays valid; the newcomer's needs γ are new, so aggregate need drifts slightly (the surplus κ = 2 absorbs this).
- It draws **new preferences** (support, weights, need) exactly like the round-0 agents, and **starts like them**: one income lump (and harvest lump) plus the subsistence reserve, debt 0.
- It has **no history**: empty experience and memory, and its public ledger record starts at 0 (the predecessor's record is archived, so the ledger still balances).
- Why: dead agents left the market for good, so with survival ≈ 0.5 half of the market was gone by round 24. Trade chains broke and trade fell partly for that reason, which confounded the money metrics. With newcomers the market stays ≈ 95% full (barter heuristic: 0.95–0.96 vs. 0.72–0.78), and newcomers must learn the existing conventions, as with entry in Kiyotaki–Wright economies with turnover.
- Learning: each life is its own trajectory. A death ends it (terminal for GAE, death penalty), and the newcomer's rounds start fresh, so no value leaks between the two lives. Cost: the full economy runs ≈ 2× slower per iteration (more living agents, more decisions).

**Round order:**
0. Newcomers enter slots that have been empty for k rounds.
1. Income arrives (capacity-limited).
2. Labour: process or idle.
3. Matching + negotiation + execution.
4. Consumption → reward r_{i,t} = u_i(c_{i,t}).
5. Spoilage, deprivation, death.

**Horizon (effectively infinite):**
- Rollouts are truncated after R = 24 rounds, and the value of the next state is bootstrapped.
- Agents do **not** observe the round index.
- This prevents money from unravelling by backward induction near a known end.
- Discount γ = 0.97.

### 3.5 Stage C: production as a choice
- **No exogenous income.** Each round, the labour action is **idle**, **harvest** one raw good, or **process** (as in B). Only raw goods can be harvested; every higher good requires inputs.
- **Access (part of the resource graph):** each agent can harvest `access_goods_per_agent` = 2 of the 4 raw goods, and every raw good is accessible to at least one agent.
- **Skill:** skill_ir ~ LogNormal(0, σ = 0.5).
- **Yield:** one round of harvesting r yields y_ir = base_r · skill_ir (capacity-limited).
- **Feasibility:** base_r ≥ 1 is set so that, if everybody split their labour evenly over their accessible goods, the harvest would cover κ = 2 times the aggregate subsistence requirement of r. This is the same guarantee as in B, expressed in labour.
- **Start:** agents begin with one harvest of each accessible good.
- **Everything else as in B:** subsistence, death, horizon, consumption, processing.
- **Control `C_no_specialization`** (F2 knock-out): full access and σ = 0 (identical yields for everyone).

### 3.6 Stage D: public ledger (F6, "money is memory")
- **Physical record:** the world always keeps, per agent and good, the cumulative **net contribution** (given − received) and, per pair, the **bilateral balance** (net units i received from j). Contributions sum to 0 per good (counting the archived records of agents replaced by newcomers, `retired_contribution`), and balances are antisymmetric.
- **Ledger on:** in trade mode, the decider additionally observes, per good, sign(v)·ln(1+|v|) of its partner's net contribution and of their bilateral balance (the two reserved token slots), plus the ledger flag in the context. Gift/credit relationships thereby become observable and enforceable by reputation.
- **Ledger off:** anonymity; the slots are 0.
- **Switch:** the state in round t is `enabled` XOR (switch_round ≥ 0 and t ≥ switch_round).
- **Disturbance design:**
  - Train C (ledger off) and D (ledger on).
  - Evaluate each with `--set ledger.switch_round=12` (the ledger switches on / off mid-episode).
  - Compare pass-through and trade before vs. after the switch (§5.3): does money use collapse, appear, or persist (hysteresis)?

### 3.7 Trade network and matching (search friction)
- **Network kinds:** Watts–Strogatz small world (degree 4, rewiring probability 0.1, connected), complete, or ring. A new network is drawn every episode.
- **Matching each round:** a uniformly random maximal matching among living agents. Every edge gets a random priority, and locally dominant edges are matched repeatedly.

### 3.8 Negotiation protocol
**Roles:** in each matched pair, a proposer i and a responder j are assigned at random.

**Turns:**

| Turn | Who | Options |
|---|---|---|
| 0 | i | Makes an opening offer o_i |
| 1 | j | **accept** o_i, **reject**, or **counter** with o_j |
| 2 (only after a counter) | i | **accept** o_j or **reject** |

**Offer encoding** (offer by o to partner p): per good k, a direction d_k ∈ {none, give, want} and an amount a_k ∈ (0,1):

$$\text{give}_k = a_k\,x_{o,k}\;[d_k=\text{give}], \qquad \text{want}_k = a_k\,x_{p,k}\;[d_k=\text{want}]$$

**Constraints:**
- Giving and wanting the same good is impossible.
- *give* needs own stock > 10⁻⁶; *want* needs the partner's stock > 10⁻⁶.
- Gifts and pure demands are allowed.
- **Capacity (C = 40):** the trade is scaled by the largest s ∈ [0,1] that keeps both sides within capacity. This preserves exchange ratios.

**Timing:** offers are computed on the holdings at the time of the trade phase. Pairs are disjoint.

### 3.9 Information structure
- **Private:** own holdings, α_i, γ_i, income (amount, rounds to the next lump), deprivation counter, processing options, harvest yields (C).
- **Ledger (D, when on):** the partner's public net contribution and the bilateral balance.
- **Public** (observable by neighbours): holdings. Preferences are public only if `full_information = True`.
- **Critic:** the centralised critic sees every agent's private information during training (CTDE). Execution is decentralised.

---

## 4. Agents and learning

### 4.1 Observations (identical layout in all stages; unused features are 0)
| View | Per-good features | Count |
|---|---|---|
| public token | properties(5), ln(1+x), α·FI, MU·FI | 8 |
| private token | properties(5), ln(1+x), α, MU, γ, ln(1+income lump), rounds-to-next-income/P, deprivation/D, ln(1+q_max), input value, ln(1+harvest yield) | 15 |
| decision token | properties(5); self: ln(1+x), α, MU, γ, ln(1+income), ln(1+q_max), input value, ln(1+harvest); partner (trade only): ln(1+x), α·FI, MU·FI; standing offer: ln(1+recv), ln(1+give), ΔU_k; ledger (trade, D, when on): signed-log partner net contribution, signed-log bilateral balance | 21 |
| decision context | round/(R−1) (A only), mode one-hot (trade/labour/consume), turn one-hot (trade), is_proposer, has_offer, ΔU_total if accepted, own load/C, partner load/C, FI, rounds-to-next-income/P, deprivation/D, ledger on | 16 |

Notation:
- MU = α_k / (1 + max(x_k − γ_k, 0)); with continuous subsistence MU = α_k / (1 + x_k − γ_k).
- Input value v_ik = max over goods using k directly of α_i,out.
- FI = 1 under full information, else 0.
- recv/give are the standing offer in **absolute units** from the decider's perspective.

The layout is identical across stages, so weights can be carried A → B → C (curriculum).

### 4.2 Architecture (`agents/policy.py`)
- **Agent embedding (once per round, from the state at the start of the round = market context):**
  - Public neighbourhood view: h = GATv2^L(DeepSets_pub(public tokens)) over the living part of the trade network, with L = 2, 4 heads and pre-LayerNorm residual blocks.
  - Private embedding: z_i = MLP([DeepSets_priv(private tokens), h_i]).
  - Only *public* information is propagated, so private preferences don't leak.
- **Decision trunk:**
  - Context c = MLP([context, z_self, h_other]), where h_other is the partner's h in trade mode and the agent's own h otherwise.
  - Sequence = [c; W·tokens + c], fed through a 2-layer Transformer encoder (pre-LN, no positional encoding). The tokens describe the *current* holdings of the phase.
- **Heads:**

  | Mode | Heads |
  |---|---|
  | trade | response {accept, counter, reject} (masked by turn); per good direction {none, give, want} (masked) + Beta amount |
  | labour | categorical over {idle} ∪ {harvest k} ∪ {process k} (harvest only for accessible raw goods, stage C; process only if q_max > 0) + Beta fraction of q_max for the chosen good |
  | consume | per good {keep, consume} (consume only if held and α > 0) + Beta fraction of holdings |

  All Beta parameters are softplus(·) + 1.
- **Critic:** V_i = MLP([h_i, mean over living j of h_j]), with h = GATv2^L(DeepSets(private tokens)). It has separate parameters from the policy.
- **Hidden size:** 64.

**Agent memory (optional, `model.memory`, size `model.memory_size` = 16; approved 2026-10-14).** Why: goods and preferences are redrawn every episode, so *which goods others readily accept* (Menger's saleability) can only be learned within an episode, and that requires memory. Without it, an agent cannot know that its last partners all asked for the same good.
- **Experience features** (private, per good, from the last round; `world.experience`, log units): units received, units given, units a partner *asked for* (want side of offers made to the agent), units a partner *offered* (give side). Reset at the start of an episode.
- **Per-good GRU:** μ_ik,t = GRUCell(μ_ik,t−1, [experience_ik,t, properties_k, ln(1 + x_ik,t)]), μ_ik,0 = 0, shared over goods (permutation-equivariant). μ is appended to the agent's own goods tokens (agent embedding) and to its decision tokens (per-good heads), so "good k is in demand" reaches the decision for good k directly.
- **Training:** the PPO update recomputes μ over all rounds of the minibatch worlds from the stored experience and snapshots, so gradients flow through time (truncated at the 24-round rollout, as in recurrent MAPPO).
- **Critic:** sees the raw experience features in addition to all private tokens.
- **Compatibility:** with memory off the model and all runs are unchanged (bit-identical smoke runs before/after). Warm starts into a memory model zero-pad the new input weights (`learning/trainer.load_compatible`), so the model initially computes exactly what the source computed (tested). Cost: ≈ 1.1× time per iteration.
- **Role in the design:** private memory of one's own trades vs. the public ledger of stage D separates "learning saleability" from "money is memory" (F6).

### 4.3 MAPPO (`learning/ppo.py`)
- **RL step = one agent-round.** The composite action is labour + up to two trade decisions + consumption, with log π = the sum of the parts.
- **Masking:** agents dead at the start of a round are masked out. Agent-rounds without a decision still train the critic.
- **Advantages:** GAE(γ, λ = 0.95) with per-agent termination at death.
  - Exchange economy: terminal after R rounds, γ = 1.
  - Production economy: truncated after R rounds with bootstrap V(s_R), γ = 0.97.
- **Normalisation:** advantages are normalised over acting agent-rounds.
- **Loss:** clipped surrogate (ε = 0.2) + 0.5·MSE(V, return) − c_H·mean entropy (over all decisions), c_H = 0.01 (stage A) and **0.03 in presets B–D**. In the B2 runs with 0.01, policy entropy collapsed from 0.6 to 0.1–0.2 and trade fell with it (§11).
- **Optimisation:** Adam with lr 3e-4, gradient-norm clipping at 0.5, 4 epochs, 4 minibatches over whole worlds.
- **Storage:** each decision is stored with its exact non-differentiable inputs (tokens, context, masks). The differentiable embeddings are recomputed from the round snapshots.
- **No behavioural cloning, no expert, no shaping.**

### 4.4 Baselines and counterfactuals (`agents/baselines.py`)
- **RandomPolicy:** uniformly random allowed choices in all modes. This is the null of the exchange economy.
- **ZI-C** (Gode & Sunder 1993): random offers, accepts only if ΔU_total ≥ 0 (myopic individual rationality).
  - It *does* accept goods it has no use for whenever that costs nothing (e.g. swapping one useless good for another), and it passes them on by chance.
  - In the A pilot, that gave a resale rate of 0.21 vs. 0.12 for random play.
  - It is therefore a second, stronger null, not a "no indirect exchange" reference.
- **RandomTradeNull(policy) / ZICTradeNull(policy):** the policy's own labour and consumption, with random / ZI-C trading. These are the nulls of the production economy: they isolate trade behaviour.
- **NoTradePolicy(policy):** the policy's own labour and consumption, with trading switched off. Comparing against it gives the gains from trade.

---

## 5. Measurement

### 5.1 Ledger and flow tracker (`economy/ledger.py`)
**Trade ledger: one row per executed trade**, recording:
- world, round, proposer a, responder p
- units a receives/gives
- the immediate ΔU of both sides (utility of holdings, a proxy in B)
- the accepting turn

**Also stored:** α, γ, x₀, x_final, weights, spoilage and the number of meetings.

**FlowTracker** (production economies) follows every unit acquired through trade until it leaves the agent:
- **Pool:** each holding x_ik contains a pool P_ik ≤ x_ik of acquired units.
- **Proportional attribution:** every outflow of k (consumption, production input) removes the fraction outflow/x_ik of the pool. Spoilage multiplies it by (1−δ_k). Death empties it.
- **Trade outflows:** their pool share counts as **completed pass-through** only if the same trade brings the agent a *useful* good. Useful means α_ik > 0, or the good is transitively an input to a useful good.
- **Totals per (world, good):** supply (initial + income + production), acquired, passed, consumed, produced.
- **Totals per agent:** summed flow utility, alive rounds, deaths.

### 5.2 Benchmarks (exchange economy, `analysis/benchmarks.py`)
**Common basis:** spoilage is good-specific, so aggregate end-of-episode supply is X_k(1−δ_k)^R whatever the trades were. Benchmarks use the *spoiled endowments* e_i = x_{i,0}(1−δ)^R.

- **Walrasian equilibrium:**
  - Demand is exact water-filling: sort goods by α_k / p_k, then λ = Σ_A α / (m + Σ_A p) on the active set and x_k = max(0, α_k/(λ p_k) − 1).
  - Prices come from tâtonnement p ← p·exp(0.3·clip(z/X, −1, 1)) for 300 steps, normalised to a mean of 1 over traded goods. The residual is about 1e-7.
- **Utilitarian planner:** per good, water-filling x_ik = max(0, α_ik/μ_k − 1).
- **Welfare** W = Σ_i U_i under: no trade, Walras, planner, and the realised allocation.
- **Production economies:** no closed-form benchmark yet (planned). Gains from trade are measured against the no-trade counterfactual instead.

### 5.3 Money metrics (`analysis/money_metrics.py`), symmetric over all goods
**Exchange economy (stage A).** Values are measured at Walras prices p_k. ν_ik = 1[α_ik < 10⁻⁶] means "no use".

| Metric | Definition |
|---|---|
| trade_rate | executed trades / meetings |
| no_use_share | Σ_receipts Σ_k p_k in_k ν_ik / Σ_receipts Σ_k p_k in_k |
| resale_rate (completed indirect exchange) | Receipts are processed in round order with a no-use pool per (agent, good). An outflow counts as resale up to the pool, and only if the same trade brings a valued good. resale_rate = Σ p·resold / Σ p·no-use inflow |
| money_hhi | per world, HHI over goods of the no-use acceptance value shares, averaged over worlds with any |
| money_good_*_rank | within-world spoilage / weight rank (0 = lowest) of the top no-use good |
| double_coincidence_share, speculative_acceptance_share | trades where both sides gain immediately / receipts with ΔU < 0 |
| efficiency_vs_walras / _planner | (W_final − W_no-trade) / (W_bench − W_no-trade) |
| ex_post_ir_violation | share of agents ending below their no-trade utility |
| costly_no_use_share, costly_resale_rate | *descriptive only:* the same measures restricted to no-use goods acquired by giving up a valued good. This is a sufficient sign of purposeful indirect exchange, but not a necessary one: in Kiyotaki–Wright, agents give up their own non-consumed production good, which costs nothing. It is therefore not used for classification |

**Production economy (stage B on), fate-based.** Volumes are normalised per (world, good) by the supply S_bk, not by prices: input-only goods would get a Walras price of 0 and drop out of exactly the analysis that matters.

| Metric | Definition |
|---|---|
| pass_through_rate | Σ_bk passed_bk/S_bk / Σ_bk acquired_bk/S_bk |
| money_hhi | per world, HHI of the shares of passed_bk/S_bk over goods (worlds with pass-through) |
| money_good_{spoilage,weight,tier}_rank | within-world rank of the good with the largest pass-through share |
| traded_share_of_supply, consumed_share_of_supply | market depth, consumption |
| mean_flow_utility, survival_rate | welfare per living agent-round; share of the agents present at round 0 still alive at the end (unchanged meaning without newcomers) |
| births, death_rate, population_share | newcomers per world; deaths per living agent-round; share of agent slots occupied over the rollout (market thickness) |
| labour_specialization | mean over working agents of the HHI of their non-idle labour choices (1 = always the same activity). Read relative to `C_no_specialization` and the null: random labour already gives about 0.48 with 2 accessible goods vs. 0.30 with full access |
| idle_share, harvest_share, process_share | composition of labour |
| pass_through_rate_before/after, trade_rate_before/after | (D, if switch_round ≥ 0) the same measures in rounds before / from the ledger switch |

### 5.4 Stage classification (null-referenced)
**Why the null is needed:** an untrained/random policy already produces large no-use shares (about 0.49 in A). Volume alone is therefore not evidence of money.

**Method:** every benchmark batch is re-run with **two nulls** on the **identical economies** (same RNG state at reset), and is judged against the **stronger null** (element-wise maximum of their metrics):
- exchange: RandomPolicy and ZICPolicy
- production: RandomTradeNull(policy) and ZICTradeNull(policy)

Evaluations report both nulls (`null_random_*`, `null_zic_*`) and the maximum (`null_*`).

| Stage | Exchange (A) | Production (B+) |
|---|---|---|
| autarky | trade_rate < 0.02 | trade_rate < 0.02 |
| barter | no_use_share < 0.05 **or** resale_rate < stronger null + 0.05 | pass_through_rate < stronger null + 0.05 |
| indirect_exchange | otherwise, unless money | otherwise, unless money |
| money | money_hhi ≥ stronger null + 0.15 | money_hhi ≥ stronger null + 0.15 |

The thresholds are **provisional** (`MetricsConfig`) and will be fixed before the confirmatory experiments.

### 5.5 Timing of emergence and statistics
- **Change points:** binary segmentation for mean shifts in the benchmark-iteration curves, with a BIC-type penalty 2σ²ln n and σ from the MAD of first differences. It reports the training iteration at which each curve changes regime.
- **Across seeds:** interquartile mean with percentile-bootstrap 95% CIs (Agarwal et al. 2021).

---

## 6. Experimental design

### 6.1 Stages A–D (each is an economy type and a curriculum step)
| Stage | Economy | Status | Money motive tested |
|---|---|---|---|
| A | Pure exchange: one-off endowments, trade only | **implemented** | Transaction role (Feldman 1973; Ostroy & Starr 1974) |
| B | Lumpy raw-goods income, processing via recipes, learned consumption, subsistence and death, infinite horizon | **implemented** | Store of value across time (timing mismatch, spoilage) |
| C | Production choice: agents harvest raw goods according to access and skills (only raw goods are harvested; higher goods require inputs) | **implemented** | Co-emergence of division of labour and money |
| D | Public ledger: trained on (D) vs. off (C), evaluated with a mid-episode switch (F6) | **implemented** | "Money is memory" (Kocherlakota) vs. credit (Graeber) |

**Curriculum:** A → B → C with carried-over weights (`--init`), **and** every stage is also trained from scratch. Comparing the two separates inherited money conventions (path dependence) from emergent ones.

**Learning sessions within a run** (`curriculum.sessions`; presets B–D, decided 2026-10-13):

| Session | Iterations | Technology (`production.max_tier`) | Death after |
|---|---|---|---|
| S1 | 1–75 | raw goods only (no processing) | 20 deprived rounds |
| S2 | 76–150 | tiers ≤ 1 | 10 |
| S3 | 151–end | all tiers (the full economy) | 5 |

- Each session is the base configuration plus dotted overrides. The world is rebuilt at a session boundary, which consumes no random numbers, so resumes stay bit-identical (tested). metrics.csv logs a `session` column.
- **Only the full economy is ever evaluated.** The last session must end before the final iteration (validated).
- **Rationale:** survival in B requires intermediation along network chains (§11). Learners died before they could discover it, so the gradient pushed toward autarky. The sessions first let agents learn trade and consumption in a simpler economy with a gentle death rule. Death is ramped down rather than switched off, because below subsistence utility is 0: without death there would be no incentive to secure the subsistence good at all.
- **Interpretation:** this is a training curriculum, comparable to humans learning progressively while technology expands. It is not simulated history: within an episode, every good of the session is available from round 0. Technology unlocking *within* the economy would be a separate, economic condition.
- **Risks and controls:**
  - Path dependence: raw goods are traded before processed goods exist, so a raw good may become the medium of exchange because it came first. Every condition uses the identical schedule; the tier of the money good is reported (`money_good_tier_rank`); a run without sessions (`--set curriculum.sessions=[]`) is the control.
  - The sessions never mention or reward any good or trade; they only change the technology level and the death threshold.
  - Presets C and D inherit the sessions from B (C: harvesting from session 1). A C run warm-started from B therefore repeats the schedule.

### 6.2 Causal identification
1. **Frictionless baseline:** complete network, durable goods, full information (and, in D, a public ledger). Money must *not* emerge (validity check).
2. **Knock-in:** each friction alone, added to the baseline (tests sufficiency).
3. **Knock-out:** each friction alone, removed from the full world (tests necessity).
4. **Dose–response:** sweeps of the decisive factors.
5. **Mechanism:** money must be used disproportionately in meetings *without* a double coincidence.
6. **Swap test:** move the properties of the money good to another good; that good should become money.

### 6.3 Factors and hypotheses
| Factor | Parameter | Hypothesis |
|---|---|---|
| F1 Preference sparsity (double coincidence) | `preferences.goods_valued` / `goods_valued_range`, `dirichlet_beta`, `goods.goods_per_tier`; treatment presets `B_rare`, `B_rare_perishable` | Sparser preferences → more money (Jevons) |
| F2 Resource access / skills (stage C) | `production.access_goods_per_agent`, `skill_sigma`; control `C_no_specialization` | No division of labour → no money |
| F3 Production depth | `goods.goods_per_tier` | Deeper production graphs → earlier, stronger money (new hypothesis) |
| F4 Holding costs | `goods.spoilage_range`, `weight_range`, `market.spoilage`; treatment preset `B_perishable` (δ ~ U[0, 0.25]) | The good with the lowest holding cost becomes money (Menger's saleability); perishable food creates a store-of-value demand for durable goods (Kiyotaki–Wright storage costs) |
| F5 Search friction | `network.kind`, `degree` | Inverted U |
| F5b Timing mismatch (stage B) | `production.income_period_range` (lumpy vs. (1,1)) | Lumpy income → store-of-value demand → more money |
| F6 Memory / anonymity (stage D) | `ledger.enabled`, `ledger.switch_round` | A public ledger substitutes for money (pass-through falls when the ledger is on and rises after it is switched off) |

---

## 7. Parameters (defaults, `configs/base.py`; preset overrides in §8)
| Group | Parameter | Default |
|---|---|---|
| goods | goods_per_tier / weight_range / spoilage_range | (4,3,2) / (0.2, 2.0) / (0, 0.10) |
| goods | raw_consumable_prob / recipe_inputs / recipe_coef_range | 0.5 / (1,3) / (0.5, 2.0) |
| preferences | goods_valued m / dirichlet_beta | 3 / 1.0 |
| preferences | subsistence / subsistence_goods / subsistence_range / subsistence_max_tier | off / 1 / (0.5, 1.5) / 1 |
| preferences | continuous_subsistence | off (presets B–D: on) |
| preferences | goods_valued_range / tier_weights / tier_weight_sigma / min_processed_valued | (0, 0) = fixed m / () = off / () = fixed / 0 |
| endowments (A) | goods_per_agent / amount_range / consumable_only | 2 / (2, 10) / true |
| production (B) | income / income_goods_per_agent / income_flow_range / income_period_range | off / 1 / (1, 2) / (1, 4) |
| production (B) | subsistence_surplus κ / processing / subsistence_buffer_rounds | 2.0 / off / 0 (presets B–D: 10) |
| production (C) | harvesting / access_goods_per_agent / skill_sigma | off / 2 / 0.5 |
| production | max_tier (technology level) | −1 (all tiers) |
| curriculum | sessions | () = none |
| ledger (D) | enabled / switch_round | off / −1 (never) |
| network | kind / degree / rewire_prob | small_world / 4 / 0.1 |
| market | hunger_debt / respawn_after | off / −1 (presets B–D: on / 2) |
| market | economy / n_agents / rounds / capacity / full_information / spoilage / death_after / death_penalty | exchange / 12 / 8 / 40 / false / true / 5 / 0 |
| model | hidden / gnn_layers / attn_heads / transformer_layers | 64 / 2 / 4 / 2 |
| model | memory / memory_size | off / 16 |
| ppo | worlds / iterations / lr / epochs / minibatches / clip | 64 / 300 / 3e-4 / 4 / 4 / 0.2 |
| ppo | gamma / gae_lambda / value_coef / entropy_coef / max_grad_norm | 1.0 / 0.95 / 0.5 / 0.01 (presets B–D: 0.03) / 0.5 |
| ppo | benchmark_every / checkpoint_every | 10 / 50 |
| metrics | no_use_alpha / autarky_trade_rate / indirect_share / excess_resale / excess_hhi | 1e-6 / 0.02 / 0.05 / 0.05 / 0.15 |

## 8. Presets (`configs/presets.py`)
| Name | Condition |
|---|---|
| `A` | Stage A default (small world, private preferences) |
| `A_complete` | Stage A, complete network (minimal search friction) |
| `A_full_info` | Stage A, observable partner preferences |
| `B` | Stage B: economy = production, income on (P ∈ {1..4}), processing on, subsistence on one **raw** good with γ ~ U[0.2, 0.5], all raw goods consumable, 2–5 valued goods, tier weights (0.2, 1, 2) with log-spreads (0.25, 0.75, 1.0), starting reserve 10 rounds, learning sessions S1–S3 (§6.1), death penalty 0.5, hunger-debt death rule, newcomers after 2 empty rounds, continuous subsistence utility, entropy coefficient 0.03, rounds = 24, worlds = 32, γ_RL = 0.97 |
| `B_constant_income` | `B` with constant income (P = 1); control for F5b |
| `B_perishable` | `B` with spoilage δ ~ U[0, 0.25] instead of U[0, 0.10]; treatment for F4 |
| `B_rare` | `B` with 15 goods (6 / 5 / 4 per tier) and 2–3 valued goods per agent; treatment for F1. Neighbour pairs whose incomes are mutually useful (incl. as inputs): 0.28 instead of 0.66 (direct consumption only: 0.04 instead of 0.23); 24% of agents value no processed good (`min_processed_valued` stays off). Runs ~1.15× slower |
| `B_rare_perishable` | `B_rare` + spoilage U[0, 0.25]: F1 × F4 (Kiyotaki–Wright-like) |
| `C` | `B` without income, with harvesting (access 2 of 4 raw goods, skill σ = 0.5) |
| `C_no_specialization` | `C` with full access and σ = 0; control for F2 |
| `D` | `C` with the public ledger on |
| `quick`, `quick_B`, `quick_C`, `quick_D` | Tiny configurations for smoke tests (`quick_B/C/D`: one iteration per learning session, then one on the full economy) |

## 9. Code map
```
configs/    base.py (dataclasses), presets.py (named conditions), night_queue.txt (job list for scripts/run_queue.py)
economy/    goods.py, preferences.py, endowments.py, production.py, network.py, trade.py, observations.py,
            ledger.py (trade ledger + FlowTracker), world.py (exchange + production economy)
agents/     layers.py (GATv2, DeepSets), policy.py (policy + critic), baselines.py (nulls, ZI-C, counterfactuals)
learning/   rollout.py (phases + protocol), ppo.py (MAPPO), trainer.py, evaluate.py (held-out eval + paired null), seeding.py
analysis/   benchmarks.py, money_metrics.py, changepoints.py, stats.py
scripts/    train.py, evaluate.py, report.py, summarize.py, run_queue.py (CLI)
tests/      test_economy.py, test_analysis.py, test_learning.py, test_production.py (B, C), test_ledger.py (D)
docs/       FRAMEWORK.md (this file), HOW_IT_WORKS.md (detailed plain-language explanation, kept in sync)
results/    run outputs (git-ignored)
```

## 10. How to run
```bash
python -m unittest discover -s tests -t .           # all invariants and smoke tests
python -m scripts.train --preset A --seed 0          # -> results/A_s0/{config.json, metrics.csv, checkpoint.pt, evaluation.json}
python -m scripts.train --preset B --seed 0          # stage B from scratch
python -m scripts.train --preset B --seed 0 --init results/A_s0/checkpoint.pt --run-dir results/B_from_A_s0   # warm start
python -m scripts.evaluate --run-dir results/A_s0    # held-out evaluation vs. nulls/baselines + change points
python -m scripts.train --preset C --seed 0 --set preferences.goods_valued=2           # any parameter via --set
python -m scripts.evaluate --run-dir results/D_s0 --set ledger.switch_round=12 --out evaluation_switch_off.json
python -m scripts.train --resume --run-dir results/B_s0   # continue an interrupted run (its own config.json is used)
python -m scripts.train --preset B --seed 0 --set curriculum.sessions=[]   # control: no learning sessions
python -m scripts.report --run-dir results/A_s0           # -> report.png + JSON summary (stages, change points, evaluation)
python -m scripts.summarize --out results/summary.md      # all runs in one table + aggregates across seeds
python -m scripts.run_queue configs/night_queue.txt --parallel 2   # unattended queue (start detached; results/queue.log)
```
**Report figure:**
- Small multiples over training iterations (benchmark iterations only), one y-axis each: the learned policy (blue, solid, markers) vs. the paired null (orange, dashed), and change points (dotted).
- An ordinal stage strip below (autarky → money, light → dark blue).
- Learning-session starts are drawn as solid grey lines: jumps there come from the curriculum, not from emergence.
- The palette was validated for colour-vision deficiency (ΔE 24.7 learned vs. null) and the ramp is monotone.
**Resuming:**
- A run continues from its last checkpoint (written every `checkpoint_every` = 50 iterations and at the end).
- Weights, optimizer state, the iteration counter and elapsed time are restored. metrics.csv rows after the checkpoint are dropped and the log is continued.
- Checkpoints contain the torch/NumPy/Python RNG states, so a resumed run is **bit-identical** to an uninterrupted one (tested).
- Checkpoints written before resume support (no RNG state) are resumed with a deterministic reseed (seed + iteration). That is statistically equivalent, but not bit-identical.
- Checkpoints are written atomically (temporary file + rename), so shutting down during a save cannot corrupt the previous checkpoint.
**Runtime** (CPU, 12 threads):
- Preset A: about 12 s per iteration (≈ 1 h for 300 iterations).
- Preset B: about 35 s per iteration, measured while an A run shared the CPU (≈ 3 h for 300 iterations).

Runs are deterministic given the seed (verified: a relaunch reproduces the logged metrics exactly).

**Unattended nights:** **Dark Awake** (a system tool outside the repo): `C:\Users\cwink\Tools\dark-awake.ps1`, started with the hotkey **Ctrl+Alt+D** (Start-menu shortcut "Dark Awake"). It blacks out the screen (brightness 0 + black overlay) while the display stays logically on, so Modern Standby cannot suspend the runs. It runs until any key or click, then restores the brightness. Only one instance runs at a time.

**Long runs** should be started detached from the editor/terminal session, so that a session restart does not kill them
(Windows PowerShell):
```powershell
Start-Process python -ArgumentList "-u","-m","scripts.train","--preset","A","--seed","0" `
  -RedirectStandardOutput results\A_s0.log -RedirectStandardError results\A_s0.err -WindowStyle Hidden
```

## 11. Assumptions, limitations, open issues
- **Stage A has no time dimension beyond R = 8 rounds.** Only money's transaction role can appear there (by design, §6.1).
- **Agent embeddings are computed once per round from the round-start state.** Phase-specific changes (after labour or trade) enter only through the decision tokens, which use current holdings. This is an approximation: agents observe their neighbourhood at the start of the round.
- **Agent memory is optional** (`model.memory`, §4.2) and off in all presets so far. Without it, the standing offer is the only negotiation history, and with the ledger off agents have neither public nor private memory of past trades. The memory treatment (night 4) separates private memory from the public ledger (F6).
- **Stage-B pilot (seed 0, no reserve), diagnosed at iteration 50.** Overall survival was 11%:
  - Agents that earn their subsistence good, or all of its raw inputs, survived (100% / 94%).
  - The 89% that need inputs from others all died, at round ≈ 4.
  - The policy drifted toward autarky (71% idle labour, trade rate 0.42 → 0.11).

  Fix: the starting reserve (§3.4), plus warm-starting B from a trained A model. The pilot was stopped and is kept as a record of this failure mode (`results/pilot_B_s0_no_reserve`). Re-evaluated in reserve economies, its policy still eats the reserve within about 6 rounds: consumption smoothing has to be learned with the reserve in training.
- **Stage-B survival requires intermediation (diagnosed 2026-10-13, with the reserve).** In B runs at iteration 50, the dependent agents (89%) survive 0–3.5%.
  - A hand-coded sensible barter policy reaches only 22% overall survival.
  - It does not improve with a 4× supply surplus, more trade sessions per round, later death, or a money convention. It does improve with raw-only subsistence goods (55%) and a complete network (63%).
  - In aggregate, dependents receive more than they need, but only 56% ever receive their subsistence good: many have no neighbour who earns it, and goods must travel along chains via intermediaries.
  - Survival is therefore gated by the very coordination problem money solves, and learners die before discovering it. **Decided 2026-10-13:** learning sessions with a death-threshold ramp and technology unlocking, plus raw-only subsistence with low-utility basic goods (§3.2, §6.1). Not yet tested in training.
  - The warm-started run, continued to 300 iterations, shows that survival does improve, but only slowly (0.10 → 0.26), so longer training alone is impractical.
- **Why trade falls and no money forms in B2 (diagnosed 2026-10-13 on the final B2 policies, 4 × 32 held-out economies):**
  - *Exploration collapse:* policy entropy fell from 0.6 to 0.1–0.2 (entropy coefficient 0.01), and the trade rate fell with it. → entropy coefficient 0.03 (presets B–D).
  - *Eating mistakes:* in 31–42% of deprived rounds the agent held enough of its subsistence good but ate too little; when it held enough, it ate a median 2.2× its need (28% of its stock), emptying the reserve within a few rounds. Consumption is a fraction of the stock, so hitting "need / stock" exactly is hard to learn. Deaths thin the market, and the trade rate falls when session 3 tightens the death rule. Proposed fix (automatic subsistence consumption; learned consumption above need) **not adopted** for now (user decision); deaths therefore still include eating mistakes.
  - *No acceptance of unusable goods:* opening offers are accepted 17% of the time; offers that give the partner nothing usable only 3%. Agents accept only goods they can use, which rules out a medium of exchange. 45% of trades are double coincidences, so the need for a medium is weak, and all goods are fairly durable (δ ≤ 0.10), so storing one's own food is cheap and no good stands out as a store of value. → perishability treatment `B_perishable` (F4). A hand-coded barter policy survives 0.43 with perishable goods vs. 0.53 without, so survival is harder by construction; conditions are compared on the money metrics.
  - Not adopted for now (proposed): rarer double coincidence (12 goods, 2–4 valued; F1), lumpier income (P 1–8; F5b), 24 agents, a softer death ramp (20 → 10 → 7 → 5).
- **Feast-and-fast: agents game the death rule (diagnosed 2026-10-14 on B3).** Death requires `death_after` = 5 *consecutive* deprived rounds, and utility counts only consumption *above* γ (max(c − γ, 0)), so eating exactly γ gives 0, while skipping meals costs nothing until the 5th round. Eating kγ once every k ≤ 5 rounds yields α ln(1 + (k − 1)γ) > 0, i.e. lumpy eating strictly beats steady eating.
  - Evidence (held-out, 4 × 32 economies): B3 survivors eat ≥ γ in only 61% (seed 0) / 50% (seed 1) of their rounds, and 3.1–3.8× γ when they eat; the longest hungry streak of 33% / 51% of survivors is exactly 4 rounds (one short of death). In 73% / 88% of deprived rounds the agent held enough food. B2 (no penalty, entropy 0.01) showed the pattern weakly (15% at streak 4); with perishable goods it is weak too (11–12%), because stored food spoils.
  - Consequence: B3's higher survival is partly rule gaming, not better provisioning, and the need to trade every round is weakened. A model fix is proposed (CLAUDE.md §8).
- **No store-of-value selection (2026-10-14):** about 20% of held units are goods without any use for the holder, but their mean spoilage equals that of all held goods (B3: 0.042 vs. 0.045; B3_perishable: 0.10 vs. 0.11). Offers that give the partner nothing usable are accepted 2–4% (B3) and 5–6% (B3_perishable) of the time. Perishability alone therefore did not make agents hold durable goods as a medium or store of value.
- **Untrained stage-B agents mostly die** (random policy: about 7% survive 24 rounds), because subsistence requires consuming ≥ γ every round. After the feasibility scaling (§3.4), these deaths are behavioural, not scarcity.
- **Consuming goods with α = 0 is masked** (dominated); such goods can still be held, processed or traded.
- **The immediate ΔU in the stage-B ledger** is a holdings-utility proxy. The double-coincidence share in B is therefore indicative only.
- **Stage thresholds** (§5.4) are provisional. The classification is per benchmark batch, and the nulls are stochastic, so seeds and batches must be aggregated (§5.5).
- **Pass-through and resale use pooled attribution** (units are fungible): proportional in B, pool-first in A.
- **Walras/planner benchmarks assume γ = 0** and an exchange economy. There is no benchmark for production economies yet.
- **Labour specialisation needs a reference:** restricted access mechanically raises the index (§5.3), so it is interpreted against `C_no_specialization` and the null.
- **The first stage-A pilot run (seed 0) used the pre-unification feature layout** (8/9/15/10 features). Its checkpoint does not load into the current model, so stage A must be re-trained for curriculum warm starts.

## 12. Results log

### Current status (2026-10-13, 10:30)
All simulations are stopped. Cross-run table: `python -m scripts.summarize` (written to `results/summary.md`).

| Condition | Runs | Outcome |
|---|---|---|
| **A** (pure exchange) | seeds 0, 1, 2 (300 iterations); seed 3 stopped at the iteration-100 checkpoint (resumable) | **Efficient barter + purposeful indirect exchange, no stable money.** Efficiency vs. Walras 0.54 [0.53, 0.56] vs. ZI-C 0.23. Resale 0.39 [0.37, 0.42] vs. stronger null 0.21. Costly resale 0.39. HHI 0.33 [0.30, 0.38]: an excess over the null of +0.06/+0.14/+0.07, where +0.15 is needed. Money conventions appear transiently late in training and dissolve again. |
| **B** warm-started from A | seed 0, 300 iterations | **Barter with clear gains from trade, survival too low.** Flow utility 0.212 vs. 0.161 without trade (+32%) and 0.172 with random trade. Survival 0.25 vs. 0.15 / 0.17, rising slowly (~+0.05 per 100 iterations). Pass-through 0.16 below the stronger null 0.20. The HHI of the little pass-through there is lies above the null (+0.09 to +0.16). |
| **B** from scratch | seed 0, stopped at iteration 69 | Autarky collapse (trade rate → 0). |
| **B2** (redesign: learning sessions + basic-needs preferences) warm-started from A | seed 0, 300 iterations, death penalty 0 and 0.5 | **Survival 0.46 (greedy 0.50)** vs. 0.25 in the old design; still rising, gate 0.5 not reached. Barter with gains from trade (+37% flow utility vs. no trade); pass-through 0.22 vs. null 0.19 (+0.03, threshold +0.05); no money (HHI below the null). The **death penalty 0.5 hurt**: survival 0.38, trade rate falling to 0.25, idle labour 74% (hoarding/autarky drift). |
| **B3** (B2 + death penalty 0.5 + entropy 0.03) and **B3_perishable** (+ spoilage U[0, 0.25], F4), warm from A | seeds 0 and 1, 300 iterations each (night 2) | **B3 passes the survival gate nominally** (0.51 / 0.59; greedy 0.61 / 0.74), but largely by **gaming the death rule** (feast-and-fast, §11). Trade 0.34 / 0.25, pass-through *below* the null, HHI at or below the null: barter. **Perishability did not create money:** survival 0.33 / 0.35, trade 0.34 / 0.40, pass-through and HHI below the null; held goods without use are not more durable than average. |
| **C, D** | — | Not run; they depend on a working stage B. |

**Next step:** night 3 (prepared 2026-10-14, starts on the user's command): B4 = preset B with continuous subsistence, hunger debt and newcomers (seed 0, warm from A_s0); tonight only the two corners of the 2×2 screening, `B4` and `B4_rare_perishable` (F1 × F4), ~6.5 h, because newcomers make iterations ≈ 2× slower; `B4_rare` and `B4_perishable` follow in night 4. Pre-flight (barter heuristic, debt + newcomers): survival of the initial agents 0.55 / 0.48 / 0.54 / 0.46 for B / perishable / rare / rare + perishable, market 95–96% full. Then the memory treatment, C and D.

### Night 1 summary**Pilot A, seed 0** (300 iterations, pre-unification features; `results/pilot_A_s0_old_features`, held-out evaluation over 4 batches × 64 economies):

| | learned greedy | learned stochastic | ZI-C | random |
|---|---|---|---|---|
| efficiency vs. Walras | 0.54 | 0.53 | 0.23 | 0.09 |
| double-coincidence share of trades | 0.69 | 0.61 | 0.15 | 0.10 |
| speculative acceptance | 0.06 | 0.08 | 0.25 | 0.37 |
| ex-post IR violation | 0.015 | 0.028 | 0.28 | 0.37 |
| resale rate (completed indirect exchange) | 0.40 | 0.37 | 0.21 | 0.12 |
| money_hhi | 0.36 | 0.36 | 0.24 | 0.25 |
| stage vs. stronger null (resale 0.21, HHI 0.245) | indirect exchange | indirect exchange | barter | barter |

- **What emerged:** efficient, mostly mutually beneficial barter plus purposeful indirect exchange. Change points in resale, no-use share and efficiency fall at **iteration ≈ 50**.
- **What did not emerge:** money. Concentration is +0.11 over the stronger null, below the +0.15 threshold. The most re-traded good has average spoilage and weight ranks (≈ 0.5), so there is no saleability selection.
- **Theory check:** consistent with stage A offering only a transaction motive. A single "money" batch at iteration 290 was not confirmed on held-out economies (noise).

**A, seed 0** (unified features, the curriculum base; `results/A_s0`, 300 iterations, held-out evaluation against the stronger null):

| | learned greedy | learned stochastic | ZI-C | random |
|---|---|---|---|---|
| efficiency vs. Walras | 0.55 | 0.56 | 0.23 | 0.09 |
| double-coincidence share | 0.72 | 0.60 | 0.15 | 0.10 |
| ex-post IR violation | 0.021 | 0.037 | 0.28 | 0.37 |
| resale rate (stronger null 0.21) | 0.43 | 0.42 | 0.21 | 0.12 |
| costly resale rate (descriptive) | 0.43 | 0.44 | 0.21 | 0.11 |
| money_hhi (stronger null 0.245) | 0.31 | 0.30 | 0.24 | 0.25 |
| stage (4/4 batches) | indirect exchange | indirect exchange | barter | barter |

- **Replication:** the pilot result is reproduced with the new feature layout: efficient barter plus purposeful indirect exchange, no money.
- **Costly indirect exchange:** agents also *give up valued goods* to acquire no-use goods that they later resell (costly resale 0.44 vs. 0.21). That is clear evidence of purposeful indirect exchange.
- **Change points:** resale and HHI at iterations 60 and 110; efficiency at 50, 100 and 150.
- **Observation (not a claim):** an early transient. HHI peaked at 0.49 around iteration 30, and two training batches (iterations 40–50) were classified as "money", while trade per meeting was very low (≈ 0.12). Concentration then decayed as trading widened. Note that the training log of this run used the random-only null (code loaded before the stronger-null change); only the held-out evaluation uses the stronger null.

**A, seed 1** (`results/A_s1`, 300 iterations; training log and evaluation both use the stronger null):

| | learned greedy | learned stochastic | ZI-C | random |
|---|---|---|---|---|
| efficiency vs. Walras | 0.54 | 0.53 | 0.23 | 0.09 |
| double-coincidence share | 0.71 | 0.62 | 0.15 | 0.10 |
| resale rate (stronger null 0.21) | 0.39 | 0.38 | 0.21 | 0.12 |
| costly resale rate | 0.38 | 0.36 | 0.21 | 0.11 |
| money_hhi (stronger null 0.245; money ≥ 0.395) | 0.36 | 0.38 | 0.24 | 0.25 |
| stage | indirect exchange 4/4 | indirect exchange 3/4, money 1/4 | barter | barter |

- **Replicates seed 0** (efficient barter + purposeful indirect exchange), but sits **closer to the money threshold** (HHI +0.12/+0.14 vs. +0.06 in seed 0).
- **Late convergence trend:** after iteration ≈ 150, HHI keeps rising (0.28 → 0.42; change points at 60 and 250) while the no-use acceptance share falls (0.30 → 0.25). Agents become more selective *and* concentrate their indirect exchange on one good. The 300-iteration budget may cut off an ongoing convergence toward money, so A_s1 is extended to 600 iterations as a check.

**A, seed 2** (`results/A_s2`, 300 iterations): efficiency 0.53, resale 0.37 (stronger null 0.21), costly resale 0.36, HHI 0.32 (null 0.245). Indirect exchange in 4/4 held-out batches. During training there were **money episodes** at iterations 160–180 and 250 (HHI ≈ 0.40, +0.15 over the null). Each coincided with a dip in the no-use acceptance share (≈ 0.21: more selective acceptance), and each dissolved again (HHI 0.30 at iteration 300).

**Stage A across seeds 0–2** (held-out, learned stochastic; IQM with 95% bootstrap CI over 3 seeds, indicative only):

| metric | IQM | CI | per seed | stronger null |
|---|---|---|---|---|
| efficiency vs. Walras | 0.54 | [0.53, 0.56] | 0.56 / 0.53 / 0.53 | ZI-C 0.23 |
| resale rate | 0.39 | [0.37, 0.42] | 0.42 / 0.38 / 0.37 | 0.21 |
| costly resale rate | 0.39 | [0.36, 0.44] | 0.44 / 0.36 / 0.36 | 0.21 |
| money_hhi | 0.33 | [0.30, 0.38] | 0.30 / 0.38 / 0.32 | 0.245 (money ≥ 0.395) |
| double-coincidence share | 0.61 | [0.60, 0.62] | 0.60 / 0.62 / 0.60 | 0.15 |

- **Robust across seeds:** efficient barter plus purposeful (also costly) indirect exchange.
- **Money is not robust:** the HHI excess over the null is +0.055, +0.135 and +0.07 (+0.15 required).
- **Convention instability:** money conventions appear *transiently* late in training (seed 1: iteration 290; seed 2: iterations 160–180 and 250) and are not stable. This is consistent with stage A offering only a weak transaction motive (12 agents, 8 rounds, no time or production) to stabilise a medium of exchange.

**B, seed 0, warm-started from A_s0** (`results/B_fromA_s0`, with reserve; **gate failed at iteration 150**: survival 0.23 < 0.5; held-out evaluation at iteration 150):

| | learned greedy | learned stochastic | no-trade counterfactual | random-trade null |
|---|---|---|---|---|
| flow utility per living agent-round | 0.239 | 0.235 | 0.191 | 0.197 |
| survival over 24 rounds | 0.20 | 0.19 | 0.14 | 0.14 |
| trade rate | 0.35 | 0.42 | 0 | 0.49 |
| completed pass-through (stronger null 0.175–0.184) | 0.12 | 0.15 | — | 0.12 |
| money_hhi (stronger null 0.40) | 0.49 | 0.43 | — | 0.40 |
| stage (4/4 batches) | barter | barter | autarky | barter |

- **Gains from trade:** learned trading raises welfare by about 23% over no trade and over random trading, and improves survival. Unlike the scratch run, the warm start (trading skills from A) prevents the autarky collapse. That is the curriculum effect.
- **Below the null:** pass-through stays below the stronger null, so the stage is barter.
- **Concentration:** the pass-through that does occur is more concentrated than under the null (+0.03 to +0.09 held-out, up to +0.14 in training).
- **Still rising at the gate:** survival 0.10 → 0.23 (change point at iteration 100), and labour specialisation 0.66 → 0.82.
- **Consequence:** the C/D chain is blocked by the gate. The run is resumed to 300 iterations to test whether longer training is enough (option (c) in CLAUDE.md §8).

**Final, at iteration 300** (resumed bit-identically from 151; held-out evaluation):

| | learned greedy | learned stochastic | no-trade counterfactual | random-trade null |
|---|---|---|---|---|
| flow utility | 0.212 | 0.212 | 0.161 | 0.172 |
| survival | 0.28 | 0.25 | 0.15 | 0.17 |
| trade rate | 0.28 | 0.35 | 0 | 0.51 |
| completed pass-through (stronger null 0.20–0.22) | 0.15 | 0.16 | — | 0.17 |
| money_hhi (stronger null 0.32–0.33) | 0.48 | 0.42 | — | 0.33 |
| labour specialisation | 0.84 | 0.74 | 0.89 | 0.76 |
| stage (4/4 batches) | barter | barter | autarky | barter |

- **Gains from trade:** +32% flow utility over no trade, +23% over random trade.
- **Survival keeps rising slowly** (0.10 → 0.26 over 300 iterations; change point at 130; about +0.05 per 100 iterations). Reaching the 0.5 gate would take roughly 800 iterations if the trend held, i.e. longer training is possible in principle but impractical.
- **Pass-through** rises toward the null (0.11 → 0.16–0.18 vs. 0.20) but stays below it. Its concentration lies above the null (greedy +0.16).

**B, seed 0, from scratch** (`results/B_s0`, with reserve; stopped at iteration 69 after failing the gate): autarky from iteration ≈ 50 (trade rate 0.41 → 0.00), survival 10–15%. This replicates the pilot-B collapse despite the reserve. See §11 for the intermediation diagnosis.

**A, seed 3** (`results/A_s3`, stopped at the iteration-100 checkpoint to end the session; resumable with `--resume --run-dir results/A_s3`): at iteration 100, efficiency 0.48, resale 0.29 vs. 0.19, HHI 0.38 vs. 0.23, indirect exchange. It matches seeds 0–2 at the same stage of training. Not evaluated (partial).

**Night 2 (2026-10-13/14): B3 and B3_perishable, seeds 0 and 1** (queue runner, 21:01–03:32, no failures; 3.1–3.3 h per run with two in parallel). B3 = preset B (learning sessions, raw subsistence, random tier weights) + death penalty 0.5 + entropy coefficient 0.03; B3_perishable = B3 with spoilage U[0, 0.25] (F4). Warm starts from A_s0 / A_s1. Held-out evaluation, learned stochastic policy (greedy in brackets):

| run | survival | flow utility (no trade) | trade rate | pass-through (null) | HHI (null) | idle labour | stage |
|---|---|---|---|---|---|---|---|
| `B3_fromA_s0` | 0.507 (0.608) | 0.175 (0.128) | 0.336 | 0.181 (0.200) | 0.292 (0.306) | 0.55 | barter 4/4 |
| `B3_fromA_s1` | 0.592 (0.741) | 0.150 (0.110) | 0.249 | 0.142 (0.213) | 0.352 (0.273) | 0.70 | barter 4/4 |
| `B3_perishable_fromA_s0` | 0.333 (0.358) | 0.172 (0.142) | 0.338 | 0.123 (0.131) | 0.375 (0.408) | 0.53 | barter 4/4 |
| `B3_perishable_fromA_s1` | 0.353 (0.374) | 0.162 (0.131) | 0.396 | 0.112 (0.146) | 0.374 (0.392) | 0.58 | barter 4/4 |
| `B2_fromA_s0` (reference) | 0.462 (0.501) | 0.189 (0.138) | 0.417 | 0.221 (0.190) | 0.253 (0.364) | 0.52 | barter 3/4 |

- **The entropy fix worked:** policy entropy stayed at 1.0–1.3 (B2: fell to 0.15), and the penalty no longer led to an autarky drift (B2_penalty: survival 0.38, trade 0.25).
- **Survival passes the gate, but for the wrong reason:** B3 survivors exploit the consecutive-rounds death rule (feast-and-fast, §11). Survival rises through session 3 (seed 1: 0.30 → 0.66) while trade falls (0.44 → 0.29).
- **Trade still declines in session 3** (B3: 0.52–0.56 in session 1 → 0.35–0.39), though less than in B2.
- **No money in any run:** pass-through excess over the null is −0.02 to −0.07 at evaluation, and the concentration of pass-through goods is at or below the null. Gains from trade remain (+37% / +36% flow utility vs. no trade in B3; +21% / +24% in B3_perishable).
- **Perishability (F4):** more trade during sessions 2–3 (up to 0.58, brief indirect-exchange episodes around iterations 150–220 in seed 0), then a decline; survival flat at ≈ 0.33 (a barter heuristic also loses 10 points with perishable goods, §11). No selection of durable goods as store of value.

**B2, seed 0, redesigned stage B** (learning sessions S1 raw only/death 20 → S2 tier ≤ 1/death 10 → S3 full economy; subsistence from raw goods; random tier weights; warm start from `results/A_s0`; 300 iterations, 3.8 h with both runs in parallel (62 / 49 / 35 s per iteration in sessions 1 / 2 / 3); held-out evaluation, learned stochastic policy):

| | survival | flow utility | trade rate | pass-through (null) | HHI (null) | idle labour | stage |
|---|---|---|---|---|---|---|---|
| `B2_fromA_s0` (no penalty) | **0.462** (greedy 0.501) | 0.189 | 0.417 | 0.221 (0.190) | 0.253 (0.364) | 0.52 | barter |
| `B2_penalty_fromA_s0` (p = 0.5) | 0.380 (greedy 0.395) | 0.178 | 0.250 | 0.123 (0.175) | 0.453 (0.386) | 0.74 | barter |
| no-trade counterfactual (no penalty run) | 0.255 | 0.138 | 0 | — | — | 0.86 | autarky |
| random-trade null (no penalty run) | 0.266 | 0.147 | 0.494 | 0.160 | 0.364 | 0.60 | barter |
| old design `B_fromA_s0` (for comparison) | 0.253 | 0.212* | 0.352 | 0.163 (0.201) | 0.420 (0.326) | 0.62 | barter |

\* Flow utilities are not comparable across the two designs (different preference weights); compare each with its own no-trade counterfactual: +37% (new) vs. +32% (old).

- **The redesign works partially.** Survival in the full economy is 1.8× the old design and 1.8× the no-trade counterfactual, i.e. trade keeps agents alive. Session-3 survival rose steadily (means per 50 iterations 0.33 → 0.37 → 0.42); the gate (0.5) is reached only by the greedy policy.
- **Session transitions are visible:** survival drops at each session start (0.85 → 0.52 at iteration 76; 0.61 → 0.32 at 151), as the death rule tightens and new goods appear. The curriculum did what it should: in sessions 1–2, trade rose (up to 0.50) and brief indirect-exchange episodes appeared (iterations 60, 100).
- **Death penalty 0.5 hurt in the end.** The penalty run was ahead until about iteration 230 (indirect exchange at the end of session 2, trade rate 0.54; survival 0.42–0.44 in session 3), then drifted into hoarding: trade 0.43 → 0.25, idle labour 74%, pass-through below the null, survival 0.38. This matches the predicted risk (precautionary hoarding of the subsistence good). One seed only, so the comparison is suggestive, but the penalty is not adopted.
- **No money:** the HHI of pass-through goods lies below the null in the no-penalty run; pass-through exceeds the null by +0.03 (threshold +0.05).

**Pilot B, seed 0** (no reserve; `results/pilot_B_s0_no_reserve`, stopped at iteration 86): collapsed into autarky (trade rate 0.42 → 0.002, survival 9% → 14%). Diagnosis and fix in §11.

## 13. Changelog
- **2026-10-12:** Complete rewrite. The grid-world survival simulation (agent.py, environment.py, dataset.py, training.py, network.py, sim_check.py, main.py, config.py, evaluation.py, metrics.py and the expert data/checkpoints) was removed; it remains in git history. Implemented:
  - stage A of the network economy
  - sparse Dirichlet log preferences
  - the alternating-offer protocol with per-good direction/amount offers
  - the GATv2 + goods-transformer policy with a centralised critic and MAPPO
  - the ledger and the Walras/planner benchmarks
  - null-referenced money metrics, change points and the tests
- **2026-10-12:** Validity fix: stage classification is now relative to a random-policy null on identical economies, after observing that an untrained policy already shows a no-use share of about 0.49. Resale is counted only when the agent receives a valued good in return.
- **2026-10-12:** Documented detached long runs (§10). The first stage-A run was killed by a session restart at iteration 25; it was relaunched, and the relaunch reproduced the logged metrics exactly.
- **2026-10-12:** **Stage B implemented** (approved design):
  - lumpy raw-goods income (constant-income control)
  - Leontief processing
  - learned consumption with Stone–Geary subsistence on basic goods and death after 5 deprived rounds
  - an effectively infinite horizon (truncation + bootstrap, no round index, γ = 0.97)
  - three-phase rounds (labour → trade → consumption) with composite actions in MAPPO
  - a FlowTracker with the fate-based pass-through money metric (supply-normalised) and production-stage classification
  - the RandomTradeNull and NoTradePolicy counterfactuals
  - presets `B`, `B_constant_income`, `quick_B`
  - 11 new tests (26 in total)

  **Observation layout unified across stages** (public 8, private 15, decision token 21, context 16), with ledger slots reserved for D. Stage-A runs made before this change are not checkpoint-compatible. Stage-A log column renamed: `mean_episode_utility_gain` → `mean_episode_reward`.
- **2026-10-12:** **Subsistence feasibility scaling** (κ = 2, `production.subsistence_surplus`). Income is scaled per world and raw good so that it covers twice the aggregate subsistence need, after measuring that 57% of random worlds were infeasible.
- **2026-10-12:** **Stage C implemented:**
  - harvesting of raw goods only, with access to 2 of 4 raw goods and LogNormal skills
  - labour-based feasibility scaling (κ = 2)
  - an initial harvest lump
  - labour-choice counts with the metrics `labour_specialization`, `idle_share`, `harvest_share`, `process_share`
  - presets `C`, `C_no_specialization`, `quick_C`
  - 4 tests
- **2026-10-12:** **Stage D implemented:**
  - a physical ledger of net contributions and bilateral balances, observable in trade mode when on (the reserved slots are now used, so feature sizes are unchanged)
  - `ledger.enabled` / `switch_round`, with before/after-switch pass-through and trade rates (per-round flow tracking)
  - generic `--set key=value` overrides for training and evaluation, and `--out` for evaluation files
  - presets `D`, `quick_D`
  - 5 tests (35 in total)
- **2026-10-12:** **Resumable training:**
  - `--resume` in `scripts/train.py`, `train(..., resume=True)`
  - checkpoints now also store elapsed time and RNG states, and are written atomically
  - resume truncates metrics.csv to the checkpoint iteration
  - tests: bit-identical resume and resume of an old-format checkpoint (37 tests in total)

  Motivation: the stage-B pilot (300 iterations, about 3 h) cannot finish before a planned shutdown.
- **2026-10-12:** Added `scripts/report.py` (run report figure + JSON summary with the stage timeline, first iteration per stage, change points and evaluation). It also works on interrupted runs.
- **2026-10-12:** Added the **subsistence reserve** (`production.subsistence_buffer_rounds`; default 0 keeps old configurations reproducible; presets B–D: 10) after diagnosing the stage-B learning trap (§11). Decided:
  - stop the B pilot
  - re-train A with the unified feature layout, as the base for the A→B→C curriculum

  2 tests (39 in total).
- **2026-10-12:** **Stronger null:** stages are now classified against the element-wise maximum of two paired nulls (random and ZI-C; production: RandomTradeNull and ZICTradeNull). The change was prompted by ZI-C itself being classified as indirect exchange in the A pilot.
  - Added descriptive `costly_no_use_share` / `costly_resale_rate`. They are not used for classification, because a "costly only" criterion would exclude Kiyotaki–Wright indirect exchange (giving up a non-consumed production good). This is a deliberately narrower version of the approved fix; the user was informed.
  - Resume now rewrites metrics.csv with the current columns.
  - Added the results log (§12). 41 tests.
- **2026-10-12:** A_s0 (unified features) finished and was re-evaluated against the stronger null (results log §12). The report legend now reads "paired null (as logged during training)", because older logs contain the random-only null.
- **2026-10-12:** `CLAUDE.md` rewritten as the operating manual. It covers working rules, environment pitfalls, standard commands, the current state, the unattended/overnight protocol with gates, the experiment queue, pending decisions and key literature. The model specification stays in this document.
- **2026-10-13:** Added `scripts/keep_awake.ps1` after the PC suspended for about 1.5 h during the first unattended night (both B runs paused between iterations ≈ 15 and 16; results unaffected).
- **2026-10-13:** First night:
  - B_s0 (scratch) stopped at iteration 69 after failing the gate (autarky).
  - Stage-B survival diagnosis recorded (§11): survival requires intermediation; heuristic feasibility tests were run.
  - The proposal is in CLAUDE.md §8. A_s1 started (queue item 9).
- **2026-10-13:** First night, continued:
  - B_fromA_s0 failed the gate at iteration 150 (survival 0.23); it was evaluated, and its results are logged in §12. It will be resumed to 300 to test the trend.
  - The keep-awake helper now also requests `ES_DISPLAY_REQUIRED`: the laptop uses Modern Standby, which starts when the display turns off (it suspended the runs repeatedly between 23:02 and 02:37).
  - A_s2 started.
- **2026-10-13:** A_s1 finished and evaluated (results log §12): indirect exchange, HHI near the money threshold, with a rising late trend. B_fromA_s0 resumed from iteration 151 (exact RNG state).
- **2026-10-13:** Night 1 summary added (§12). The PC was in sleep-button/lid standby from 03:26 to 08:59. A_s2 and the resumed B_fromA_s0 continue in the morning.
- **2026-10-13:** A_s2 finished and evaluated. Added the first cross-seed aggregate for stage A (seeds 0–2) and the transient money episodes to the results log (§12). A_s3 started.
- **2026-10-13:** `keep_awake.ps1 -Dark` (dark-screen mode). The user wants the screen dark at night; switching the display off would trigger Modern Standby, so the display stays logically on at brightness 0 behind a black overlay.
- **2026-10-13:** The keep-awake helper was moved out of the project at the user's request: `scripts/keep_awake.ps1` was removed and replaced by the system tool **Dark Awake** (`C:\Users\cwink\Tools\dark-awake.ps1`, hotkey Ctrl+Alt+D).
- **2026-10-13:** End of session:
  - All runs stopped: A_s3 at its iteration-100 checkpoint; B_fromA_s0 finished at 300 and was evaluated.
  - Added `scripts/summarize.py`, a cross-run table with IQM/CI across seeds (1 test, 42 in total).
  - Added the current-status block and the final B_fromA_s0 results to §12.
  - Correction: survival in B_fromA_s0 is not flat, but rises slowly (~+0.05 per 100 iterations).
- **2026-10-13:** **Stage-B redesign (approved):**
  - Learning sessions within a run (`curriculum.sessions`, `learning/trainer.py`): S1 raw goods only with death after 20, S2 tiers ≤ 1 with death after 10, S3 the full economy (presets B–D). Technology level `production.max_tier`. Session column in metrics.csv; session starts in the report figure.
  - Basic needs vs. higher wants: subsistence only from raw goods (all raw goods consumable), variable support size 2–5, random log-normal tier multipliers (means 0.2/1/2, log-spreads 0.25/0.75/1.0). Optional `min_processed_valued` (at least k processed valued goods), off in all presets. The user asked that utilities stay truly random and that non-subsistence goods stay valued; the calibration check is in §3.2.
  - All new options default to off; old configurations sample bit-identically.
  - 7 tests (49 in total). Also closed file handles in `scripts/summarize.py` and its test.
- **2026-10-13:** `min_processed_valued` switched off in the presets (kept in the code; user decision); the calibration in §3.2 was re-measured. Added the optional death penalty `market.death_penalty` (default 0), to be A/B-tested with p = 0.5 (§3.4). 1 test (50 in total).
- **2026-10-13:** B2 (redesigned stage B) trained warm from A_s0, with and without death penalty 0.5, and evaluated (results log §12): survival 0.46 vs. 0.25 in the old design; the penalty led to late hoarding (survival 0.38). Death penalty not adopted.
- **2026-10-13:** Diagnosis of the B2 trade drop (§11): exploration collapse, eating mistakes, no acceptance of unusable goods. Decided: death penalty 0.5 and entropy coefficient 0.03 in presets B–D; new treatment preset `B_perishable` (spoilage U[0, 0.25], F4). Added `scripts/run_queue.py` (unattended job queue; 2 tests, 52 in total) and `configs/night_queue.txt` for night 2 (B3 vs. B3_perishable, seeds 0–1). End-to-end dry run of the queue at tiny size passed.
- **2026-10-14:** Night 2 finished (B3, B3_perishable, seeds 0–1; results log §12): barter, no money; survival gate passed nominally through feast-and-fast gaming of the death rule (diagnosed, §11); perishability did not create store-of-value holdings. Housekeeping: removed `SUB_SLEEP` / `SUB_VIDEO` (accidental files from a PowerShell alias clash, `ac` = Add-Content, committed by mistake in c0b3718) and four stray files from the first run of the queue-runner test (backslashes in Windows paths); `scripts/run_queue.py` now rejects backslashes in queue files (1 test, 53 in total).
- **2026-10-14:** **Continuous subsistence utility** (`preferences.continuous_subsistence`, on in presets B–D; approved): u = α ln(1 + c − γ) below the need, against feast-and-fast gaming of the death rule (§3.4). New treatment presets `B_rare` (F1: 15 goods, 2–3 valued) and `B_rare_perishable` (F1 × F4). `configs/night_queue.txt` = night 3 (2×2 screening, seed 0); dry run at tiny size passed. 3 tests (56 in total).
- **2026-10-14:** **Agent memory implemented** (approved design: per-good GRU over rounds + private trade experience: received, given, asked for, offered; `model.memory`, off by default; §4.2). Warm starts into memory models are function-preserving (`load_compatible`); runs without memory are bit-identical to before (smoke-run fingerprints). ≈ 1.1× time per iteration. 4 tests (60 in total).
- **2026-10-14:** **Death mechanics revised (approved):** cumulative hunger debt instead of consecutive deprived rounds (`market.hunger_debt`), and newcomers refill slots empty for 2 rounds (`market.respawn_after`; same position and income, new preferences, no history; archived ledger records keep the ledger balanced); both on in presets B–D, off by default (runs with both off are bit-identical to before). `survival_rate` now counts the agents present at round 0; new metrics `births`, `death_rate`, `population_share`. Night 3 reduced to the two corner runs (iterations ≈ 2× slower). 5 tests (65 in total).
- **2026-10-14:** Added `docs/HOW_IT_WORKS.md`, a detailed plain-language explanation of every mechanic, the network and the training procedure (user request); it is updated with every change of mechanics or optimisation settings (CLAUDE.md §2.3).
