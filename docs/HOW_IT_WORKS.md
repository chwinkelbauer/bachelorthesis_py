# How It Works: the complete explanation of "Money from Scratch"

> **What this file is.** A detailed, plain-language explanation of everything the project does: the economy, every
> mechanic, what the agents see, the neural network, exactly how training works, the stages and learning sessions,
> how money is measured, the experiments and their history. It is written to be *understood* from top to bottom.
>
> **How it relates to the other documents.** `docs/FRAMEWORK.md` is the formal specification and logbook (results,
> changelog). `CLAUDE.md` is the operating manual for working sessions. This file explains; when it and the code
> disagree, the code is right and this file is out of date, which counts as a bug.
>
> **Maintenance rule.** This file is updated in the same change as **every change of the mechanics or of the
> optimisation/training settings** (see CLAUDE.md §2). The change log at the end lists every update.
>
> Last updated: 2026-10-14 (hunger debt, newcomers, agent memory, rare-coincidence presets).

---

## Contents

1. The idea in one page
2. Vocabulary
3. The economy: how a world is generated
4. The agents: preferences, needs, incomes
5. One round, step by step (stage A and stages B–D)
6. Trade in detail: network, matching, negotiation, execution
7. Eating, hunger debt, death and newcomers
8. What an agent observes
9. The neural network
10. Training: exactly what happens in one iteration
11. Curriculum: warm starts and learning sessions
12. The stages and all presets with their exact settings
13. Evaluation: how a finished run is judged
14. Measuring money: every metric explained
15. Experimental design: which friction causes money?
16. History of the stage-B designs and what each run showed
17. Running the project: commands, nights, speed
18. Code map
19. Known limitations and open issues
20. Change log of this guide

---

## 1. The idea in one page

**Question.** Do learning agents in a simulated economy *invent* money on their own, and which condition causes it?

**Approach.**

- We generate many small random economies. Each has 12 agents connected in a social network, a set of goods described
  only by physical properties (weight, how fast they spoil, how processed they are), and random preferences for every
  agent.
- All agents are controlled by **one shared neural network** (a "policy"). It receives each agent's own situation as
  input and outputs that agent's decisions: what to offer in a trade, whether to accept, what to produce, what to eat.
- The network is trained by **reinforcement learning**: it plays thousands of economies, receives rewards for the
  utility its agents get from consumption (and a small penalty when one dies), and gradually changes its behaviour to
  get more reward.
- **Nobody tells the agents about money.** No good is labelled as money, no reward mentions trade or money. If money
  appears, it appears because agents discovered that accepting a certain kind of good, only to pass it on later, helps
  them get what they actually want.
- We then **measure** whether agents accept goods they cannot use, pass them on, and concentrate on one such good,
  always compared with "null models" (random traders) on identical economies.
- **Causes are tested experimentally**: frictions known from monetary theory (rare double coincidence of wants,
  storage costs, search costs, timing mismatch between income and needs, missing records) are switched on and off.

**Stages.** The economy is built up in four stages, each richer than the last:

| Stage | Economy | Why it matters for money |
|---|---|---|
| A | Pure exchange: agents get goods once, trade for 8 rounds | Money as a means to complete trades without a double coincidence |
| B | Production: lumpy raw-goods income, processing, eating, survival | Money as a store of value across time |
| C | Production as a choice: harvesting raw goods by access and skill | Division of labour and money emerging together |
| D | A public ledger of past trades, switched on/off | "Money is memory": does a public record replace money? |

**Status (2026-10-14).** Stage A trained (barter + indirect exchange, no stable money). Stage B redesigned four
times; agents survive and trade, still without money. Stages C and D are built and tested but not yet trained.

---

## 2. Vocabulary

| Term | Meaning here |
|---|---|
| **World / economy** | One independent economy: 12 agents, their goods, network, preferences. Many worlds are simulated in parallel. |
| **Round** | One period inside an economy (income, work, trade, eating). Stage A: 8 rounds; B–D: 24 rounds per rollout. |
| **Episode / rollout** | Playing all rounds of a batch of worlds once. Every rollout uses **freshly generated** worlds. |
| **Iteration** | One training step: one rollout of 32 worlds (stage A: 64) followed by a network update. A run has 300 iterations. |
| **Run** | One complete training of one condition with one random seed, e.g. `results/B3_fromA_s0`. |
| **Seed** | The random-number start value; with the same seed a run is reproduced exactly. |
| **Preset** | A named set of all settings (one experimental condition), e.g. `B`, `B_perishable`. |
| **Policy** | The shared neural network that makes all agents' decisions. |
| **Critic** | A second network that estimates how good a situation is for an agent (its "value"); used only in training. |
| **Warm start** | Starting a run from the trained network of an earlier stage instead of from random weights. |
| **Learning session** | A phase inside a run (S1, S2, S3) in which the economy is deliberately simpler. |
| **Null model** | A non-learning reference (random or "zero-intelligence" traders) run on the identical economies. |
| **Benchmark iteration** | Iterations 1, 10, 20, …, 300, on which money metrics and null models are computed and logged. |
| **Pass-through / resale** | Accepting a good one cannot use and later trading it on for a useful good: indirect exchange. |
| **HHI** | Herfindahl–Hirschman index: concentration of indirect exchange on goods (1 = all on one good). |

---

## 3. The economy: how a world is generated

Every time a rollout starts, **every world is regenerated from scratch** (`EconomyBatch.reset`). The agents therefore
never see the same economy twice; what they learn must be a general way of behaving, not a memorised economy.

### 3.1 Goods

- **Tiers.** Goods are arranged in tiers: tier 0 = raw resources, tier 1 = made from raw goods, tier 2 = made from
  lower goods. Default 4 + 3 + 2 = **9 goods**; the rare-coincidence treatment (`B_rare`) uses 6 + 5 + 4 = **15**.
- **Properties**, drawn independently in every world:
  - **weight** w ~ log-uniform on [0.2, 2.0]. Each agent can carry at most **40 weight units** in total.
  - **spoilage** δ ~ uniform on [0, 0.10]: each round a fraction δ of every stock of that good disappears. Perishable
    treatment: [0, 0.25].
  - **consumable**: whether the good can be eaten/used. Tier ≥ 1 goods always are; raw goods are consumable with
    probability 0.5 in stage A and always in B–D (so every raw good can serve as food).
- **No names.** A good is shown to agents only as five numbers: ln(weight), spoilage, tier / max tier, a "raw" flag,
  the consumable flag. "Good number 3" means nothing to the network, because goods are reshuffled every episode.

### 3.2 The production graph (recipes)

- Every good of tier τ ≥ 1 gets a random recipe: it needs **1 to 3 inputs** from lower tiers, **at least one from tier
  τ − 1**. Each input coefficient a ~ U[0.5, 2] says how many units of that input one unit of output needs.
- Processing is **Leontief** (fixed proportions). With holdings x, the maximal output of good k is
  $$q_{\max,k} = \min_{j:\,a_{kj}>0} \frac{x_j}{a_{kj}}$$
  The agent chooses a fraction a ∈ (0, 1) and produces q = a·q_max; the inputs q·a_kj are used up. If the new load
  would exceed the 40-unit capacity, the output is reduced until it fits.
- **Embodied raw units** (raw goods needed in total, following the recipes down): tier-1 goods need about 2.5 raw
  units on average (90% range 0.9–4.2), tier-2 goods about 5 (1.7–9).
- **Technology level** (`production.max_tier`): during the early learning sessions, recipes above a tier are switched
  off (those goods cannot be produced). In the full economy all recipes are available.

### 3.3 The network of agents

- 12 agents on a **small-world network** (connected Watts–Strogatz): each agent is linked to 4 neighbours, 10% of
  links are rewired at random. A new network is drawn every episode.
- Agents can only trade with network neighbours. Alternatives: a complete network (everyone meets everyone; the
  minimal-search-friction control) or a ring.

### 3.4 Feasibility: making sure everyone *could* survive

In stage B, 57% of random worlds could not feed everybody even in principle (measured on 200 worlds). Deaths would then
reflect built-in scarcity, not behaviour. Therefore, per world and per raw good r:

$$\text{income}_r \leftarrow \text{income}_r \cdot \max\Big(1,\; \kappa\,\frac{\sum_i \gamma_i\,\text{(raw units of r in i's food)}}{\text{income flow of } r}\Big), \qquad \kappa = 2$$

So aggregate income covers at least **twice the aggregate food need**. In stage C the same is done with harvest
yields: base yields are raised until an even split of labour would cover twice the need.

---

## 4. The agents: preferences, needs, incomes

### 4.1 Utility

Every agent i has its own randomly drawn log utility (Cobb–Douglas with Stone–Geary needs):

$$U_i(c) = \sum_k \alpha_{ik}\,\ln\big(1 + c_k - \gamma_{ik}\big)$$

- **Stage A:** c = the agent's holdings at the end; γ = 0, so U = Σ α ln(1 + x). The reward each round is the change
  in this utility of holdings, so the total reward over the episode equals the final utility gain.
- **Stages B–D:** c = what the agent **eats in that round**; the reward of the round is this "flow utility".
- **α (weights):** how much the agent likes each good. **γ (need):** units of its subsistence good it needs per round.

### 4.2 Sparse wants: who values what

- Each agent values only a few goods: exactly 3 in stage A; in B–D a number drawn per agent from **2 to 5** (`B_rare`:
  2 to 3). All other weights are **exactly 0**.
- Why: if α_ik = 0, agent i has *no use* for good k. Accepting such a good can only make sense in order to pass it on,
  so it is a clean signal of indirect exchange. The fewer goods each agent values, the rarer a "double coincidence of
  wants" (both partners wanting what the other has), which is Jevons' classic reason for money.
- The weights on the valued goods are drawn from a Dirichlet(1) distribution (a random split of 1 into parts).

### 4.3 Basic needs vs. higher wants (stages B–D)

- One of the valued goods is the agent's **subsistence good**: a **raw** good with a need γ ~ U[0.2, 0.5] units per
  round. Survival therefore needs only raw goods; processed goods are never necessary to survive.
- The weights are multiplied by random tier-dependent factors and renormalised:
  $$\alpha_{ik} = \frac{d_{ik} w_{ik}}{\sum_j d_{ij} w_{ij}}, \qquad w_{ik} = \bar w_{\tau_k}\exp\big(s_{\tau_k} z_{ik} - s_{\tau_k}^2/2\big), \quad z_{ik} \sim N(0,1)$$
  with mean factors w̄ = 0.2 / 1 / 2 and spreads s = 0.25 / 0.75 / 1.0 for tiers 0 / 1 / 2. Raw goods thus carry
  little utility ("basic needs"), processed goods a lot, with large differences between agents ("higher wants").
- Check on 500 worlds × 12 agents: 346 different combinations of valued goods, no two agents with identical weights;
  mean weight 0.19 (subsistence good), 0.39 (tier 1), 0.46 (tier 2); on average 82% of an agent's weight lies on
  goods other than its food; 12% of agents value no processed good at all. Utility per embodied raw unit is higher
  for processed goods (0.09 / 0.065 vs. 0.06), so processing pays.
- An optional switch (`min_processed_valued`) can force every agent to value at least one processed good; it exists
  but is **off** in all presets.

### 4.4 Why bare survival is worth nothing and hunger hurts

- Eating exactly the need: ln(1 + γ − γ) = 0. Eating more: positive. Eating less: **negative** (since 2026-10-14,
  `continuous_subsistence`). A fully hungry round costs α ln(1 − γ), about −0.085 on average.
- One extra unit of the food good is worth about +0.13, one extra unit of each valued processed good about +0.45.
- Because the logarithm is concave, eating steadily beats eating in lumps. Before this change, utility counted only
  consumption above the need, so skipping meals was free until death, and agents learned to fast up to 4 rounds and
  then feast (see §16).

### 4.5 Income (stage B)

- Each agent **earns one raw good**. Every raw good has at least one earner: raw goods are shuffled and handed to
  agents in random order first, then every agent without one gets a random raw good.
- Mean income 1–2 units per round, paid as a **lump every P rounds**, P drawn per agent from {1, 2, 3, 4}, with a
  random phase. So many agents receive food or inputs only every few rounds but need food every round: a timing
  mismatch that creates a reason to store value.
- Income is then scaled up by the feasibility rule (§3.4).
- Control preset `B_constant_income`: P = 1 for everybody.

### 4.6 Harvesting (stage C)

- No income. Each agent has **access to 2 of the 4 raw goods** (every raw good accessible to someone) and a skill per
  good, skill ~ log-normal(0, 0.5). Harvesting good r for one round yields base_r × skill (capacity-limited).
- Control `C_no_specialization`: everyone can harvest everything with equal skill.

### 4.7 Starting holdings

- **Stage A:** every agent gets 2 random consumable goods, amounts U[2, 10], scaled down to at most 80% of capacity.
- **Stages B–D:** one income lump (stage C: one harvest of each accessible good), plus a **reserve of the subsistence
  good worth 10 rounds of need**. Without that reserve, 89% of agents had nothing to eat and died before they could
  learn to trade.

---

## 5. One round, step by step

### 5.1 Stage A (pure exchange), 8 rounds

1. **Matching:** random pairs of neighbours (§6.2).
2. **Negotiation and execution** (§6.3–6.4).
3. **Spoilage:** every stock shrinks by its good's spoilage rate.
4. **Reward** = change in the utility of holdings. Agents see the round number (round / 7), because stage A has a known
   end.

### 5.2 Stages B–D, 24 rounds per rollout

0. **Newcomers** enter slots that have been empty for 2 rounds (§7.4).
1. **Income arrives** for agents whose lump is due: (round − phase) mod P = 0 (capacity-limited).
2. **Labour:** each living agent chooses one action: idle, process one good (if it has the inputs and the technology is
   unlocked), or, in stage C, harvest one accessible raw good.
3. **Matching, negotiation, execution** (§6).
4. **Consumption:** each agent chooses, per valued good it holds, whether to eat and what fraction. The flow utility
   of what it eats is its **reward** for the round.
5. **Spoilage** of all stocks; **hunger debt** update; **death** if the debt reaches the threshold (§7).

**Horizon.** The economy has no end: after 24 rounds the rollout is cut, and the critic estimates the value of the
situation at that point ("bootstrapping"). Agents never see the round number, so there is no "last round" in which
money would lose its value (backward induction cannot unravel money). Future rewards are discounted by 0.97 per round.

---

## 6. Trade in detail

### 6.1 Who can trade

Only living network neighbours, at most **one meeting per agent per round**. There is no central market, no auctioneer
and no price list: every exchange is a bilateral bargain.

### 6.2 Matching

Every round, a **random maximal matching** is drawn: every possible pair gets a random priority, and pairs whose two
members both rank each other highest are matched, repeated until no further pair of free neighbours exists. In each
matched pair, one member is randomly the **proposer**, the other the **responder**.

### 6.3 Negotiation: alternating offers, at most two offers

| Turn | Who | Options |
|---|---|---|
| 0 | Proposer | must make an opening offer |
| 1 | Responder | accept the offer, reject it, or make a counter-offer |
| 2 (only after a counter) | Proposer | accept the counter-offer or reject it |

**What an offer is.** For every good the offering agent chooses a **direction** (none, give, want) and an **amount**
a ∈ (0.001, 0.999):

$$\text{give}_k = a_k\, x_{\text{own},k}, \qquad \text{want}_k = a_k\, x_{\text{partner},k}$$

So an offer is a whole bundle swap ("I give you 40% of my grain and 10% of my salt, you give me 30% of your iron"),
expressed in fractions of the actual stocks. Rules: you can only give what you hold and want what the partner holds;
the same good cannot be given and wanted at once; gifts (give only) and pure requests (want only) are allowed.

**Capacity.** If the swap would push either side above 40 weight units, the whole offer is scaled down by the largest
factor that fits, which keeps the exchange ratios.

### 6.4 Execution

An accepted offer moves the goods immediately. Every executed trade is written to the **trade ledger** (world, round,
both agents, units each way, each side's immediate utility change, the accepting turn). In stages B–D a **flow tracker**
also follows the traded units (§14.3), and each agent's **experience** of the round is recorded for its memory (§9.6):
units received, units given, units the partner asked for, units the partner offered.

### 6.5 What each side knows while negotiating

Own holdings, preferences, needs, income timing, hunger debt and production options; the partner's holdings (public);
the standing offer in units; the utility change the offer would bring. The partner's preferences are hidden, except in
the full-information variant of stage A. In stage D, while the ledger is on, also the partner's public trade record
and the bilateral balance between the two.

---

## 7. Eating, hunger debt, death and newcomers

### 7.1 Eating

Per good, keep or eat, plus the fraction of the stock to eat. Eating goods the agent does not value is blocked (it is
pointless). Eating reduces the stock; the flow utility of what was eaten is the reward.

### 7.2 Hunger debt (since 2026-10-14)

Each agent carries a debt d, measured in **rounds of missing food**:

$$d \leftarrow \max\big(0,\; d + 1 - c/\gamma\big)$$

Eating nothing adds 1, eating half the need adds 0.5, eating the need adds 0, eating twice the need pays one round back.
The agent observes its own debt (as d / threshold). Previously a round counted as "starving" if the agent ate even
slightly less than its need (99% counted like 0%), and one adequate meal reset the counter to zero.

### 7.3 Death

When d reaches the threshold (5 in the full economy; 20 and 10 in learning sessions 1 and 2), the agent dies:

- it leaves the market (no more matching), its goods disappear, its rewards stop;
- it receives a one-off reward of **−0.5** (the death penalty). The penalty refers to no good and no trade, and the
  money metrics never use rewards, so it cannot favour money directly.

For learning, a death ends that agent's trajectory: no future value is added after it.

### 7.4 Newcomers (since 2026-10-14)

A slot that has been empty for **2 rounds** gets a newcomer at the start of the next round. The newcomer:

- **keeps the slot's position**: network place, income good, income size and schedule (stage C: harvest access and
  skills), like a new person taking over a farm. Every raw good therefore keeps an earner.
- **draws new preferences and needs**, exactly like the first agents.
- **starts like them**: one income lump plus the 10-round food reserve, hunger debt 0.
- **has no history**: empty memory and experience; a clean public ledger record (the predecessor's record is archived,
  so the ledger still balances).

Why: without newcomers, dead agents left for good. With survival around 0.5, half the market was gone by round 24, so
trade chains broke and trade fell partly for that reason, which distorted the money metrics. With newcomers the market
stays about 95% full, and newcomers must learn the existing conventions, as with entry in monetary search models.
Cost: about twice as many decisions per round, so the full economy trains about 2× slower.

### 7.5 How survival is reported

- **survival_rate**: share of the agents present at round 0 still alive after 24 rounds (comparable to older runs).
- **births** (newcomers per world), **death_rate** (deaths per living agent-round), **population_share** (how full the
  market was on average).

---

## 8. What an agent observes

Goods are presented as **tokens**: one feature vector per good. The network processes the set of tokens with
operations that do not depend on the order of goods, so it works for any number of goods and treats all goods alike.

**Public token** (what neighbours can see about an agent), 8 features per good: the 5 good properties, ln(1 + holding),
and two preference features that are 0 unless full information is switched on.

**Private token** (what the agent knows about itself), 15 features per good: the 5 properties; ln(1 + holding); its
weight α; its marginal utility α/(1 + x − γ); its need γ; ln(1 + income lump); rounds to the next income lump / P;
hunger debt / threshold; ln(1 + maximal processing output); input value (the highest weight of any valued good that
uses this good directly as an input); ln(1 + harvest yield).

**Decision token** (one per good for each single decision), 21 features: the 5 properties; own holding, weight,
marginal utility, need, income, processing option, input value, harvest yield; in trade mode the partner's holding (and
partner preferences only under full information); the standing offer in units (receive / give); the utility change of
this good if the offer were executed; two ledger features (stage D, ledger on).

**Decision context** (16 numbers per decision): round fraction (stage A only); which mode (trade, labour, eat); which
turn; proposer or not; whether an offer is standing; total utility change if accepted; own and partner load relative to
capacity; full-information flag; rounds to next income; hunger debt; ledger flag.

**Memory input** (only with memory on), 10 numbers per good: the 4 experience values of the last round (received,
given, asked for, offered), the 5 properties, ln(1 + holding).

The layout is the same in all stages (unused features are 0), so a network trained in stage A can continue in B, C, D.

---

## 9. The neural network

All agents use **the same network**; it behaves differently for different agents because its inputs contain the
agent's own preferences, needs and situation. Hidden size 64 everywhere. Pure PyTorch, CPU. **189,201 parameters**
in the policy (192,593 with memory), **80,641** in the critic (80,897 with memory).

### 9.1 Step 1: the neighbourhood view (computed once per round)

1. Each agent's **public** goods tokens are encoded by a *DeepSets* encoder: every token goes through a small
   two-layer network (8 → 64 → 64, GELU), then the token outputs are pooled by mean and maximum and mapped to one
   64-number vector (Linear 128 → 64 + LayerNorm).
2. Two **graph-attention layers** (GATv2, Brody et al. 2022; 4 attention heads) let each agent combine the vectors of
   its living neighbours. Each layer: normalise, compute attention scores between connected agents, take the weighted
   sum of neighbour messages, add it to the agent's vector (residual), then a feed-forward block (64 → 128 → 64).
   Only *public* information flows through the network of agents, so private preferences never leak to others.

### 9.2 Step 2: the agent embedding

The agent's **private** goods tokens (plus its memory, if on) are encoded by a second DeepSets encoder (15 → 64), joined
with its neighbourhood vector and mapped (128 → 64 → 64 + LayerNorm) to one **agent embedding** z. This also happens
once per round, from the situation at the start of the round.

### 9.3 Step 3: the decision trunk (for every single decision)

- The 16-number context, the agent's embedding z and a second embedding (the partner's neighbourhood vector in trade,
  the agent's own otherwise) are combined into a **decision vector** c (144 → 64 → 64).
- Each good's decision token is projected to 64 numbers and c is added to it. The sequence [c, good 1, good 2, …] goes
  through a **2-layer Transformer** (4 heads, feed-forward width 128, pre-normalisation, no dropout, no position
  encoding, so goods have no order). Output: one summary vector (from c) and one vector per good.

### 9.4 Step 4: the decision heads

| Mode | What is decided | How |
|---|---|---|
| Trade | response: accept / counter / reject | from the summary vector (64 → 64 → 3); impossible options masked (turn 0 must counter = make the opening offer; turn 2 cannot counter) |
| | per good: none / give / want | from each good's vector (64 → 3); masked if not holdable/available |
| | per good: amount | Beta distribution from each good's vector (64 → 2, softplus + 1) |
| Labour | idle, harvest good k, process good k | one choice among 1 + 2·G options: "idle" from the summary vector, "harvest k"/"process k" from good k's vector; impossible options masked |
| | amount of processing | Beta distribution for the chosen good (fraction of the maximal output) |
| Eating | per good: keep / eat | from each good's vector (64 → 2); eating masked if not held or not valued |
| | per good: fraction eaten | Beta distribution |

During training, decisions are **sampled** from these distributions (exploration). In evaluation we report both the
sampled ("stochastic") policy and the **greedy** one (most likely choice, mean amount).

### 9.5 The critic

A separate network (same building blocks) that sees **every agent's private tokens** (centralised training,
decentralised execution): DeepSets encoder (15 → 64), 2 GATv2 layers over the living network, then for each agent
[own vector, mean of all living agents' vectors] → 64 → 1 = the estimated value of the agent's situation. It is used
only during training; agents act without it.

### 9.6 Memory (optional, built 2026-10-14)

- Why: which goods others readily accept (Menger's "saleability") can only be learned *within* an episode, because
  goods are redrawn every episode. That requires memory.
- What: each agent keeps a 16-number memory **per good**, updated once per round by a GRU cell (a standard recurrent
  unit) from that good's memory input (§8): what happened with this good in the agent's trades last round.
- How it is used: the memory of good k is appended to the agent's private token of good k (agent embedding) and to its
  decision token of good k (so "good k was in demand" reaches the decision about good k directly). The critic sees the
  raw experience values.
- Newcomers start with an empty memory. With memory off the model is exactly the memory-free model.

---

## 10. Training: exactly what happens in one iteration

Training is **MAPPO** (multi-agent proximal policy optimisation, Yu et al. 2022). One run = 300 iterations. Each
iteration has five parts.

### 10.1 Choose the configuration of this iteration

If the run has learning sessions, the iteration's session decides the settings (technology level, death threshold;
§11.2). When a new session starts, the worlds are rebuilt with its settings.

### 10.2 Rollout: play the economies

- Generate **32 fresh worlds** (stage A: 64) with the current settings, 12 agents each.
- Play **24 rounds** (stage A: 8) as in §5, with every agent's decisions sampled from the current policy.
- Store for every decision exactly what the network saw (tokens, context, masks), what was chosen, and its
  probability; plus each round's holdings snapshot, hunger debts, who was alive, who was born, the experience values,
  every reward and every death.
- After the last round, also store the next round's starting state for bootstrapping.

### 10.3 Measure (on benchmark iterations)

On iterations 1, 10, 20, …, 300, the money metrics of the rollout are computed (§14), the same economies are replayed
with the two null models (random and ZI-C trading; same random draws), and the iteration is classified (autarky /
barter / indirect exchange / money). Every iteration logs the mean episode reward and the trade rate.

### 10.4 Learning: the PPO update

**One learning step = one agent-round.** Everything an agent decided in one round (labour, up to two trade decisions,
eating) counts as one composite action; its log-probability is the sum of the parts. For a trade decision: response
probability, plus (if an offer was made) the probabilities of all directions and of the amounts of the goods involved.
Agents dead at the start of a round are skipped.

**Value targets and advantages (GAE).** The critic estimates V for every agent-round. With reward r_t:

$$\delta_t = r_t + \gamma\,(1 - \text{died}_t)\,V_{t+1} - V_t, \qquad A_t = \delta_t + \gamma\lambda\,(1 - \text{died}_t)\,A_{t+1}$$

with γ = 0.97 (stage A: 1.0) and λ = 0.95. After round 24 the critic's estimate of the next state replaces the
future (bootstrap); in stage A the episode simply ends. A death cuts the sum: no value after death flows back. The
return target is A + V. Advantages are normalised (mean 0, standard deviation 1) over all agent-rounds with decisions.

**Loss.** For every agent-round with decisions, ratio ρ = exp(new log-prob − old log-prob):

$$L = -\,\mathbb{E}\big[\min(\rho A,\ \text{clip}(\rho, 0.8, 1.2)\,A)\big] + 0.5\,\mathbb{E}\big[(V - \text{return})^2\big] - c_H\,\mathbb{E}[\text{entropy}]$$

- The clipped term (ε = 0.2) prevents too-large policy changes per update.
- The value loss trains the critic (only on agent-rounds alive at the round start).
- The entropy bonus c_H rewards keeping some randomness: **0.01 in stage A, 0.03 in B–D**. With 0.01 the stage-B
  policy became almost deterministic (entropy 0.6 → 0.1) and trade collapsed; 0.03 keeps it at about 1.0–1.3.

**Optimisation.** The 32 worlds are split into **4 minibatches of whole worlds** (8 each), and the data is used for
**4 epochs**, so 16 gradient steps per iteration. In each minibatch the network recomputes all embeddings of all rounds
of those worlds from the stored snapshots (and, with memory, runs the memory through all 24 rounds, so gradients flow
through time). Adam optimiser, learning rate 3·10⁻⁴ (constant), gradient-norm clipping at 0.5, policy and critic
updated together. Logged: policy loss, value loss, entropy, approximate KL divergence, clip fraction.

**What is deliberately absent.** No imitation of an expert, no hand-made example trades, no reward for trading or for
holding any good. The only rewards are consumption utility (stage A: holding utility) and the −0.5 at death.

### 10.5 Bookkeeping

- `metrics.csv` gets one row per iteration (benchmark iterations with all money metrics, null values and stage).
- Every 50 iterations and at the end, `checkpoint.pt` stores the networks, the optimiser state, the iteration, elapsed
  time and all random-number states; it is written atomically (a crash during saving cannot corrupt it).
- An interrupted run continues with `--resume` and gives **bit-identical** results to an uninterrupted run (tested).
- After the last iteration the run is evaluated automatically (§13), and a report figure is drawn.

### 10.6 All training settings

| Setting | Stage A | Stages B–D |
|---|---|---|
| Worlds per iteration | 64 | 32 |
| Agents per world | 12 | 12 |
| Rounds per rollout | 8 (episode ends) | 24 (cut + bootstrap) |
| Iterations per run | 300 | 300 |
| Discount γ | 1.0 | 0.97 |
| GAE λ | 0.95 | 0.95 |
| PPO clip ε | 0.2 | 0.2 |
| Value loss weight | 0.5 | 0.5 |
| Entropy coefficient | 0.01 | 0.03 |
| Learning rate (Adam) | 3·10⁻⁴ | 3·10⁻⁴ |
| Epochs × minibatches | 4 × 4 | 4 × 4 |
| Gradient clipping | 0.5 | 0.5 |
| Benchmark every | 10 iterations | 10 iterations |
| Checkpoint every | 50 iterations | 50 iterations |
| Hidden size / GAT layers / heads / Transformer layers | 64 / 2 / 4 / 2 | 64 / 2 / 4 / 2 |
| Memory | off | off (treatment: on, 16 per good) |

---

## 11. Curriculum: warm starts and learning sessions

### 11.1 Warm starts between stages

Stage B starts from the trained stage-A network (`--init results/A_s0/checkpoint.pt`), later C from B and D from C.
The idea: the agents already know how to bargain. If the new network has extra inputs (memory), the new input weights
start at zero, so it behaves exactly like the source at the start. Every stage is also trained **from scratch** as a
control; comparing both separates conventions inherited from earlier stages from conventions that emerge anew. Finding
so far: stage B from scratch collapses into autarky (agents stop trading and starve), the warm start does not.

### 11.2 Learning sessions within a run (stages B–D)

| Session | Iterations | Producible goods | Death when hunger debt reaches |
|---|---|---|---|
| S1 | 1–75 | raw goods only | 20 |
| S2 | 76–150 | up to tier 1 | 10 |
| S3 | 151–300 | all (the full economy) | 5 |

- Why: survival requires trading food along chains of intermediaries, and untrained agents died before they could
  discover that. The sessions let them first learn trade and eating in a simpler, more forgiving economy, as people
  learn progressively while technology expands.
- What they never do: mention, reward or favour any good. They change only technology and the death threshold.
- Only the full economy (S3 settings) is ever evaluated. All conditions get the identical schedule.
- Controls and risks: a run without sessions (`--set curriculum.sessions=[]`) is the control. A risk is path dependence
  (raw goods are traded first and might become money because they came first); the tier of the "money" good is
  therefore reported.
- In the report figure, session starts are marked with grey vertical lines; jumps there come from the curriculum.

---

## 12. The stages and all presets with their exact settings

### 12.1 Stage A presets

| Preset | Settings |
|---|---|
| `A` | 9 goods (4/3/2), raw goods consumable with p = 0.5, 3 valued goods per agent, 2 endowed goods U[2, 10], small-world network, private preferences, spoilage on, 8 rounds, 64 worlds |
| `A_complete` | `A` on a complete network |
| `A_full_info` | `A` with partner preferences observable |

### 12.2 Stage B presets (current design, "B4")

| Setting | Value in preset `B` |
|---|---|
| Goods | 9 (4/3/2), all raw goods consumable, spoilage U[0, 0.10] |
| Valued goods per agent | 2–5, one raw subsistence good with need U[0.2, 0.5] |
| Tier weights | means 0.2 / 1 / 2, spreads 0.25 / 0.75 / 1.0 |
| Utility below the need | continuous (hunger costs utility) |
| Income | 1 raw good per agent, 1–2 units/round, lumps every 1–4 rounds, feasibility κ = 2 |
| Processing | on (Leontief recipes) |
| Starting reserve | 10 rounds of need |
| Death | hunger debt ≥ 5 (sessions: 20, 10), penalty −0.5 |
| Newcomers | after 2 empty rounds |
| Rounds / worlds | 24 / 32, discount 0.97 |
| Learning sessions | S1–S3 as in §11.2 |
| Entropy coefficient | 0.03 |

| Treatment preset | Difference from `B` |
|---|---|
| `B_constant_income` | income every round (no timing mismatch; F5b control) |
| `B_perishable` | spoilage U[0, 0.25] (holding costs; F4) |
| `B_rare` | 15 goods (6/5/4), 2–3 valued goods per agent (rare double coincidence; F1). Neighbour pairs whose incomes are mutually useful: 28% instead of 66% |
| `B_rare_perishable` | both F1 and F4 |

### 12.3 Stages C and D

| Preset | Settings |
|---|---|
| `C` | `B` without income; labour can harvest 2 of 4 raw goods, skill log-normal(0, 0.5) |
| `C_no_specialization` | `C` with access to all raw goods and equal skill (F2 control) |
| `D` | `C` with the public ledger always on; evaluated with the ledger switched off mid-episode (and `C` with it switched on) |

### 12.4 Smoke-test presets

`quick`, `quick_B`, `quick_C`, `quick_D`: tiny versions (6 agents, 3 rounds, 4 worlds, small network) used by the
automatic tests; their learning sessions are compressed to one iteration each.

Any setting can be changed for one run with `--set key=value`, e.g. `--set model.memory=true`.

---

## 13. Evaluation: how a finished run is judged

- **Held-out economies:** 4 batches of 32 worlds generated from fixed seeds (1,000,000 + batch) that training never
  used. Every evaluated policy plays exactly these economies.
- **Policies evaluated:** the learned policy sampled ("stochastic") and greedy; in B–D also the **no-trade
  counterfactual** (the learned policy with trading switched off: gains from trade) and the **random-trade null** (the
  learned policy's production and eating, random trading). In stage A the ZI-C and random traders are evaluated as
  baselines instead.
- **Per batch:** money metrics (§14), both null models on the same economies, the stronger null, and the stage
  classification. The results are averaged over batches; the stage is the most frequent one.
- **Output:** `evaluation.json` in the run folder; `scripts/report.py` draws the training curves, change points and
  the stage timeline into `report.png`; `scripts/summarize.py` puts all runs into one table with cross-seed statistics.

---

## 14. Measuring money: every metric explained

### 14.1 Why null models

Random traders already accept many goods they cannot use (about 49% of receipts in stage A). Large volumes are
therefore no evidence of money. Every measurement is compared with two **null models** run on the identical economies:

- **Random:** uniformly random offers and answers.
- **ZI-C** ("zero intelligence, constrained", Gode & Sunder 1993): random offers, but accepts only if its own
  immediate utility does not fall; counters with probability 0.5 otherwise.

In stages B–D the nulls keep the learned policy's production and eating and replace only its trading. The comparison
always uses the **stronger null** (the larger value of the two for each metric).

### 14.2 Stage A metrics

All values are measured at the Walrasian equilibrium prices of the world (so goods in different units are comparable):

- **trade_rate:** executed trades per meeting.
- **no_use_share:** value share of received goods the receiver has no use for.
- **resale_rate** (completed indirect exchange): go through all trades in time order; each agent keeps a "pool" of
  no-use goods it received; when it later gives such a good away **in a trade that brings it a good it values**, that
  amount counts as resold. resale_rate = resold value / no-use value received.
- **money_hhi:** per world, the concentration (HHI) of no-use acceptances over goods, averaged over worlds.
- **money_good ranks:** spoilage and weight rank of the most-accepted no-use good (is the medium durable, light?).
- **efficiency_vs_walras / _planner:** realised welfare gain divided by the welfare gain of the Walrasian
  equilibrium / of a utilitarian planner. The equilibrium is computed exactly: demand by water-filling, prices by
  tâtonnement (300 steps), on the endowments after spoilage.
- **double_coincidence_share:** trades in which both sides gain immediately.
- **costly resale** (descriptive only): resale of no-use goods that were bought by giving up a valued good. Not used for
  classification, because in Kiyotaki–Wright agents give up goods that cost them nothing.

### 14.3 Stage B–D metrics (fate-based)

A **flow tracker** follows every unit an agent acquires through trade:

- Each agent's stock of a good contains a "pool" of acquired units. Every outflow (eating, production input) removes
  the same fraction from the pool; spoilage and death shrink it too.
- When the agent gives the good away in a trade, the pool's share of that outflow counts as **completed pass-through**
  if the same trade brings it something **useful** (a valued good, or an input of a valued good, followed down the
  recipes).
- Volumes are normalised by each good's total supply in its world.

Metrics: **pass_through_rate** (passed on / acquired, supply-normalised), **money_hhi** (concentration of pass-through
over goods), **money_good spoilage / weight / tier rank**, **traded_share_of_supply**, **consumed_share_of_supply**,
**mean_flow_utility** (per living agent-round), **survival_rate**, **births**, **death_rate**, **population_share**,
**labour_specialization** (how concentrated each agent's work is; compared with the no-specialisation control),
idle / harvest / process shares; in stage D before/after the ledger switch.

### 14.4 Stage classification (provisional thresholds)

| Stage | Rule (against the stronger null) |
|---|---|
| Autarky | fewer than 0.02 trades per meeting |
| Barter | resale / pass-through less than null + 0.05 (stage A also: no-use share < 0.05) |
| Indirect exchange | above that, not concentrated |
| Money | concentration (HHI) at least null + 0.15 |

### 14.5 Timing and statistics

- **Change points:** binary segmentation finds the iterations at which a training curve shifts its level (penalty
  2σ² ln n, σ estimated robustly). They tell *when* behaviour changes.
- **Across seeds:** interquartile mean (mean of the middle 50%) with percentile-bootstrap 95% confidence intervals
  (Agarwal et al. 2021).

---

## 15. Experimental design: which friction causes money?

1. **Frictionless baseline** (complete network, durable goods, full information; in D a public ledger): money must not
   appear. This checks that the metrics do not produce false positives.
2. **Knock-in:** add one friction to the baseline (is it sufficient?).
3. **Knock-out:** remove one friction from the full world (is it necessary?).
4. **Dose–response:** vary the decisive friction in steps.
5. **Mechanism:** money must be used disproportionately where there is no double coincidence.
6. **Swap test:** give the money good's properties to another good; that good should become money.

| Friction | Hypothesis | Preset |
|---|---|---|
| F1 Rare double coincidence (Jevons) | fewer direct swaps → more money | `B_rare` |
| F2 Specialisation | no division of labour → no money | `C_no_specialization` |
| F3 Production depth | deeper production graphs → earlier, stronger money | (to do) |
| F4 Holding costs (Menger, Kiyotaki–Wright) | the cheapest good to hold becomes money; perishable food needs a durable store of value | `B_perishable` |
| F5 Search friction | inverted U | `A_complete` |
| F5b Timing mismatch | lumpy income → store-of-value demand → money | `B_constant_income` |
| F6 Memory / anonymity | a public record substitutes for money | `D` + switch evaluations |

Confirmatory phase (not started): ≥ 5 seeds per condition, thresholds fixed beforehand.

---

## 16. History of the stage-B designs and what each run showed

| Design | Date | Main changes | Survival | Trade rate | Pass-through vs. null | What we learned |
|---|---|---|---|---|---|---|
| Pilot B | Oct 12 | no starting reserve | 0.09–0.14 | 0.42 → 0.002 | — | Agents starve before learning: added the 10-round reserve |
| B from A | Oct 12–13 | reserve, warm start | 0.25 | 0.35 | 0.16 vs. 0.20 | Survival needs intermediation along chains; too hard to discover |
| B from scratch | Oct 12–13 | no warm start | — | → 0 | — | Autarky collapse: warm start necessary |
| B2 | Oct 13 | learning sessions, raw-only needs, random tier weights | 0.46 | 0.42 | 0.22 vs. 0.19 | Best trade; survival still below 0.5 |
| B2 + penalty | Oct 13 | death penalty 0.5 | 0.38 | 0.25 | 0.12 vs. 0.18 | Policy froze (entropy 0.1), hoarding |
| B3 (2 seeds) | Oct 13–14 | penalty 0.5 + entropy 0.03 | 0.51 / 0.59 | 0.34 / 0.25 | 0.18 / 0.14 vs. 0.20 / 0.21 | Gate passed, but by fasting up to 4 rounds (feast-and-fast) |
| B3 perishable (2 seeds) | Oct 13–14 | + spoilage up to 0.25 | 0.33 / 0.35 | 0.34 / 0.40 | 0.12 / 0.11 vs. 0.13 / 0.15 | No store-of-value selection |
| B4 | from Oct 14 | continuous utility below need, hunger debt, newcomers | — | — | — | Night 3: B4 and B4_rare_perishable |

Key insights so far:

- **Survival is gated by the coordination problem money solves**: 73% of agents do not earn their own food, many have no
  neighbour who does, so food must travel through intermediaries.
- **Trade clearly pays**: welfare 21–37% above the no-trade counterfactual in every design.
- **Why no money yet**: agents accept goods they cannot use only 2–6% of the time; in the current economy 66% of
  neighbour pairs can swap their incomes directly (raw goods are useful inputs to almost everyone), so a medium of
  exchange has little to do; goods held without use are not more durable than average.
- **Stage A** (3 seeds): efficiency 0.54 of the Walrasian gain (ZI-C: 0.23), resale 0.39 vs. 0.21 for the stronger
  null, concentration only +0.06 to +0.14 above the null (money needs +0.15); brief money episodes late in training.

---

## 17. Running the project: commands, nights, speed

```bash
python -m unittest discover -s tests -t .                       # all tests (65)
python -m scripts.train --preset B --seed 0 --init results/A_s0/checkpoint.pt --run-dir results/B4_fromA_s0
python -m scripts.train --resume --run-dir results/B4_fromA_s0   # continue an interrupted run
python -m scripts.evaluate --run-dir results/B4_fromA_s0         # evaluate again (current code)
python -m scripts.report --run-dir results/B4_fromA_s0           # report.png + summary
python -m scripts.summarize --out results/summary.md             # all runs, cross-seed statistics
python -m scripts.run_queue configs/night_queue.txt --parallel 2 # unattended queue of runs
```

- **Unattended nights:** `configs/night_queue.txt` lists the jobs; `scripts/run_queue.py` runs two at a time, logs to
  `results/queue.log`, and a file `results/queue.stop` stops it gracefully. Before leaving, Ctrl+Alt+D starts the
  system tool Dark Awake (black screen, the laptop stays awake).
- **Speed (laptop CPU, 12 threads):** stage A ≈ 8 s per iteration (≈ 40 min per run). Stage B without newcomers: two
  runs in parallel ≈ 3.2–3.8 h per run. With newcomers the full economy is ≈ 2× slower: estimated ≈ 4.8 h (`B`) and
  ≈ 6.2 h (`B_rare_perishable`) with two runs in parallel.
- **Reproducibility:** fixed seeds, bit-identical resumes, held-out evaluation seeds, automatic tests.

---

## 18. Code map

| Folder / file | What it contains |
|---|---|
| `configs/base.py` | all settings as dataclasses (defaults) |
| `configs/presets.py` | the named conditions (A, B, C, D, treatments, smoke tests), learning sessions |
| `configs/night_queue.txt` | the job list of the next unattended night |
| `economy/goods.py` | goods, properties, recipes |
| `economy/preferences.py` | utility, preference sampling (also for newcomers) |
| `economy/endowments.py` | stage-A endowments |
| `economy/production.py` | income, feasibility scaling, processing, harvesting, usefulness |
| `economy/network.py` | social network, random matching |
| `economy/trade.py` | offer encoding, masks, capacity scaling |
| `economy/observations.py` | all observation features, memory inputs |
| `economy/ledger.py` | trade ledger, flow tracker |
| `economy/world.py` | the economy itself: reset, rounds, trade execution, eating, hunger debt, death, newcomers |
| `agents/layers.py` | DeepSets encoder, GATv2 layer |
| `agents/policy.py` | policy network (with memory) and critic |
| `agents/baselines.py` | random, ZI-C, null wrappers, no-trade counterfactual |
| `learning/rollout.py` | playing the rounds, the negotiation protocol, storing decisions |
| `learning/ppo.py` | GAE and the PPO update |
| `learning/trainer.py` | the training loop, sessions, checkpoints, resume, warm start |
| `learning/evaluate.py` | held-out evaluation with paired nulls |
| `analysis/` | benchmarks (Walras, planner), money metrics, change points, statistics |
| `scripts/` | command-line tools: train, evaluate, report, summarize, run_queue |
| `tests/` | 65 automatic tests |
| `docs/FRAMEWORK.md` | formal specification, results log, changelog |
| `docs/HOW_IT_WORKS.md` | this explanation |

---

## 19. Known limitations and open issues

- No closed-form benchmark (planner/equilibrium) for the production economy yet; gains from trade are measured against
  the no-trade counterfactual.
- The money thresholds (+0.05 pass-through, +0.15 concentration) are provisional.
- Pass-through uses proportional attribution: units of a good are interchangeable, so it measures volumes, not the
  path of individual units.
- Agents compute their view of the market once per round; changes within the round reach them through the decision
  inputs only.
- Learning sessions could create path dependence; identical schedules and a no-session control address this.
- Newcomers inherit their predecessor's income position; the feasibility scaling used the first agents' needs (the
  2× surplus absorbs the drift).
- Labour specialisation is counted per slot, so it mixes predecessor and newcomer.
- Compute: stage-B runs take several hours each, which limits seeds per night.

---

## 20. Change log of this guide

| Date | Change |
|---|---|
| 2026-10-14 | Guide created: complete description of the current mechanics (continuous subsistence utility, hunger debt, newcomers, agent memory, learning sessions, death penalty 0.5, entropy 0.03, rare-coincidence and perishability presets) and of the training procedure. |
