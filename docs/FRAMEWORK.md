# Money from Scratch: Framework Specification (living document)

> **This document describes the current state of the project.** Every change to the code, the model, the
> parameters or the experimental design must update the affected sections and add a line to the
> [Changelog](#13-changelog) in the same change. If code and this document disagree, that is a bug.
>
> Last updated: 2026-10-12 · Implemented stages: **A, B, C, D** · Experiments: pilots running (A, B)

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
| Sparse Dirichlet log preferences + Stone–Geary subsistence (basic goods) | implemented | `economy/preferences.py` |
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
| Agent memory (GRU across rounds) | **planned** | — |
| Stage C: harvesting raw goods (access, skills), division-of-labour metrics | implemented | `economy/production.py`, `economy/world.py` |
| Stage D: public ledger (contributions, bilateral balances), on/off switch, before/after metrics | implemented | `economy/world.py`, `economy/observations.py` |
| Config overrides from the command line (`--set key=value`) | implemented | `configs/base.py`, `scripts/` |
| Resumable training (`--resume`; atomic checkpoints with optimizer, iteration, elapsed time, RNG states) | implemented | `learning/trainer.py`, `scripts/train.py` |
| Curriculum controller (warm start vs. scratch) | partial: `--init` warm start exists | `scripts/train.py` |
| Multi-seed sweep and report | **planned** | — |
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
  - consumable flag: always true for τ_k ≥ 1; true with probability 0.5 for raw goods
- **Recipe DAG:** a good k with τ_k ≥ 1 has 1–3 inputs from lower tiers, at least one of them from tier τ_k − 1. Its input coefficients are a_kj ~ U[0.5, 2] units of j per unit of k.
- **Property feature vector** (per good token): (ln w_k, δ_k, τ_k / L, 1[τ_k = 0], consumable_k).

### 3.2 Preferences (agent-specific, randomised)

$$U_i(x) = \sum_k \alpha_{ik}\,\ln\!\big(1 + \max(x_k - \gamma_{ik}, 0)\big)$$

In stage A the argument x is *holdings*. From stage B on it is the *consumption flow* c_t of the round.

- **Sparse support:** each agent values exactly m = 3 consumable goods S_i. On S_i, the weights are α_i ~ Dirichlet(β = 1); elsewhere α_ik = 0 *exactly*.
- **Why sparse:** α_ik = 0 makes "agent i has no use for good k" exact. The support size m controls how likely a double coincidence of wants is (the Jevons factor).
- **Stone–Geary subsistence** (stage B on): `subsistence_goods` = 1 good of S_i is drawn from the **basic goods** (consumable, tier ≤ 1) and gets γ_ik ~ U[0.2, 0.5] per round. In stage A, γ = 0.
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
- **Flow utility:** u_i(c) = Σ_k α_ik ln(1 + max(c_k − γ_ik, 0)).

**Deprivation and death:**
- A round counts as *deprived* if any subsistence good is consumed below γ_ik.
- After `death_after` = 5 consecutive deprived rounds, the agent dies: it leaves the matching, its holdings perish, and it receives no further reward.

**Round order:**
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
- **Physical record:** the world always keeps, per agent and good, the cumulative **net contribution** (given − received) and, per pair, the **bilateral balance** (net units i received from j). Contributions sum to 0 per good, and balances are antisymmetric.
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
- MU = α_k / (1 + max(x_k − γ_k, 0)).
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

### 4.3 MAPPO (`learning/ppo.py`)
- **RL step = one agent-round.** The composite action is labour + up to two trade decisions + consumption, with log π = the sum of the parts.
- **Masking:** agents dead at the start of a round are masked out. Agent-rounds without a decision still train the critic.
- **Advantages:** GAE(γ, λ = 0.95) with per-agent termination at death.
  - Exchange economy: terminal after R rounds, γ = 1.
  - Production economy: truncated after R rounds with bootstrap V(s_R), γ = 0.97.
- **Normalisation:** advantages are normalised over acting agent-rounds.
- **Loss:** clipped surrogate (ε = 0.2) + 0.5·MSE(V, return) − 0.01·mean entropy (over all decisions).
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
| mean_flow_utility, survival_rate | welfare per living agent-round, share of agents alive at the end |
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
| F1 Preference sparsity (double coincidence) | `preferences.goods_valued`, `dirichlet_beta` | Sparser preferences → more money (Jevons) |
| F2 Resource access / skills (stage C) | `production.access_goods_per_agent`, `skill_sigma`; control `C_no_specialization` | No division of labour → no money |
| F3 Production depth | `goods.goods_per_tier` | Deeper production graphs → earlier, stronger money (new hypothesis) |
| F4 Holding costs | `goods.spoilage_range`, `weight_range`, `market.spoilage` | The good with the lowest holding cost becomes money (Menger's saleability) |
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
| endowments (A) | goods_per_agent / amount_range / consumable_only | 2 / (2, 10) / true |
| production (B) | income / income_goods_per_agent / income_flow_range / income_period_range | off / 1 / (1, 2) / (1, 4) |
| production (B) | subsistence_surplus κ / processing / subsistence_buffer_rounds | 2.0 / off / 0 (presets B–D: 10) |
| production (C) | harvesting / access_goods_per_agent / skill_sigma | off / 2 / 0.5 |
| ledger (D) | enabled / switch_round | off / −1 (never) |
| network | kind / degree / rewire_prob | small_world / 4 / 0.1 |
| market | economy / n_agents / rounds / capacity / full_information / spoilage / death_after | exchange / 12 / 8 / 40 / false / true / 5 |
| model | hidden / gnn_layers / attn_heads / transformer_layers | 64 / 2 / 4 / 2 |
| ppo | worlds / iterations / lr / epochs / minibatches / clip | 64 / 300 / 3e-4 / 4 / 4 / 0.2 |
| ppo | gamma / gae_lambda / value_coef / entropy_coef / max_grad_norm | 1.0 / 0.95 / 0.5 / 0.01 / 0.5 |
| ppo | benchmark_every / checkpoint_every | 10 / 50 |
| metrics | no_use_alpha / autarky_trade_rate / indirect_share / excess_resale / excess_hhi | 1e-6 / 0.02 / 0.05 / 0.05 / 0.15 |

## 8. Presets (`configs/presets.py`)
| Name | Condition |
|---|---|
| `A` | Stage A default (small world, private preferences) |
| `A_complete` | Stage A, complete network (minimal search friction) |
| `A_full_info` | Stage A, observable partner preferences |
| `B` | Stage B: economy = production, income on (P ∈ {1..4}), processing on, subsistence on with γ ~ U[0.2, 0.5], starting reserve 10 rounds, rounds = 24, worlds = 32, γ_RL = 0.97 |
| `B_constant_income` | `B` with constant income (P = 1); control for F5b |
| `C` | `B` without income, with harvesting (access 2 of 4 raw goods, skill σ = 0.5) |
| `C_no_specialization` | `C` with full access and σ = 0; control for F2 |
| `D` | `C` with the public ledger on |
| `quick`, `quick_B`, `quick_C`, `quick_D` | Tiny configurations for smoke tests |

## 9. Code map
```
configs/    base.py (dataclasses), presets.py (named conditions)
economy/    goods.py, preferences.py, endowments.py, production.py, network.py, trade.py, observations.py,
            ledger.py (trade ledger + FlowTracker), world.py (exchange + production economy)
agents/     layers.py (GATv2, DeepSets), policy.py (policy + critic), baselines.py (nulls, ZI-C, counterfactuals)
learning/   rollout.py (phases + protocol), ppo.py (MAPPO), trainer.py, evaluate.py (held-out eval + paired null), seeding.py
analysis/   benchmarks.py, money_metrics.py, changepoints.py, stats.py
scripts/    train.py, evaluate.py, report.py (CLI)
tests/      test_economy.py, test_analysis.py, test_learning.py, test_production.py (B, C), test_ledger.py (D)
docs/       FRAMEWORK.md (this file)
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
python -m scripts.report --run-dir results/A_s0           # -> report.png + JSON summary (stages, change points, evaluation)
```
**Report figure:**
- Small multiples over training iterations (benchmark iterations only), one y-axis each: the learned policy (blue, solid, markers) vs. the paired null (orange, dashed), and change points (dotted).
- An ordinal stage strip below (autarky → money, light → dark blue).
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

**Long runs** should be started detached from the editor/terminal session, so that a session restart does not kill them
(Windows PowerShell):
```powershell
Start-Process python -ArgumentList "-u","-m","scripts.train","--preset","A","--seed","0" `
  -RedirectStandardOutput results\A_s0.log -RedirectStandardError results\A_s0.err -WindowStyle Hidden
```

## 11. Assumptions, limitations, open issues
- **Stage A has no time dimension beyond R = 8 rounds.** Only money's transaction role can appear there (by design, §6.1).
- **Agent embeddings are computed once per round from the round-start state.** Phase-specific changes (after labour or trade) enter only through the decision tokens, which use current holdings. This is an approximation: agents observe their neighbourhood at the start of the round.
- **No agent memory across rounds yet.** The standing offer is the only negotiation history. With the ledger off, agents therefore have neither public *nor private* memory of past trades, so F6 currently contrasts "public record" with "no memory at all". A GRU memory (private memory) is planned to separate the two.
- **Stage-B pilot (seed 0, no reserve), diagnosed at iteration 50.** Overall survival was 11%:
  - Agents that earn their subsistence good, or all of its raw inputs, survived (100% / 94%).
  - The 89% that need inputs from others all died, at round ≈ 4.
  - The policy drifted toward autarky (71% idle labour, trade rate 0.42 → 0.11).

  Fix: the starting reserve (§3.4), plus warm-starting B from a trained A model. The pilot was stopped and is kept as a record of this failure mode (`results/pilot_B_s0_no_reserve`). Re-evaluated in reserve economies, its policy still eats the reserve within about 6 rounds: consumption smoothing has to be learned with the reserve in training.
- **Untrained stage-B agents mostly die** (random policy: about 7% survive 24 rounds), because subsistence requires consuming ≥ γ every round. After the feasibility scaling (§3.4), these deaths are behavioural, not scarcity.
- **Consuming goods with α = 0 is masked** (dominated); such goods can still be held, processed or traded.
- **The immediate ΔU in the stage-B ledger** is a holdings-utility proxy. The double-coincidence share in B is therefore indicative only.
- **Stage thresholds** (§5.4) are provisional. The classification is per benchmark batch, and the nulls are stochastic, so seeds and batches must be aggregated (§5.5).
- **Pass-through and resale use pooled attribution** (units are fungible): proportional in B, pool-first in A.
- **Walras/planner benchmarks assume γ = 0** and an exchange economy. There is no benchmark for production economies yet.
- **Labour specialisation needs a reference:** restricted access mechanically raises the index (§5.3), so it is interpreted against `C_no_specialization` and the null.
- **The first stage-A pilot run (seed 0) used the pre-unification feature layout** (8/9/15/10 features). Its checkpoint does not load into the current model, so stage A must be re-trained for curriculum warm starts.

## 12. Results log
**Pilot A, seed 0** (300 iterations, pre-unification features; `results/pilot_A_s0_old_features`, held-out evaluation over 4 batches × 64 economies):

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
- **2026-10-12:** `CLAUDE.md` rewritten as the operating manual. It covers working rules, environment pitfalls, standard commands, the current state, the unattended/overnight protocol with gates, the experiment queue, pending decisions and key literature. The model specification stays in this document.
