# Money from Scratch

A multi-agent reinforcement learning economy (Bachelor thesis, WU Wien) in which **money is not built in**.
Agents have randomised, agent-specific log utilities and trade bilaterally on a network by learned negotiation.
Money is identified functionally (an otherwise useless good that is accepted and passed on) and tested against
null models and theoretical benchmarks.

**The full specification (model, mathematics, metrics, design, status and changelog) is in
[docs/FRAMEWORK.md](docs/FRAMEWORK.md).** That document is kept in sync with the code.

## Status (2026-10-14)
- **Implemented:** stages A–D (pure exchange → income/production/consumption → harvesting as a labour choice → public ledger).
- **Stage A** (3 seeds): agents learn efficient barter and purposeful indirect exchange, but no stable money yet.
- **Stage B:** after the redesign (learning sessions, raw-only subsistence, random high-utility processed goods) survival reaches 0.5–0.6, but agents partly game the death rule (fasting just short of death), and they barter without money, also when goods are perishable. Next: fix the subsistence rule, then test rarer double coincidence and agent memory.
- **C and D:** not run yet.

Details are in [docs/FRAMEWORK.md](docs/FRAMEWORK.md) §12; a complete explanation of how everything works is in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md); the operating manual for working sessions is [CLAUDE.md](CLAUDE.md).

## Quickstart
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .           # invariants + smoke tests
python -m scripts.train --preset A --seed 0          # train stage A; writes results/A_s0/
python -m scripts.evaluate --run-dir results/A_s0    # held-out evaluation vs. random null and ZI-C
python -m scripts.report --run-dir results/A_s0      # figure + summary of one run
python -m scripts.summarize                          # table of all runs + statistics across seeds
python -m scripts.train --resume --run-dir results/A_s3   # continue an interrupted run
```

Stages: A (pure exchange), B (lumpy income + processing + consumption), C (harvesting as a choice), D (public ledger).
Presets: `A`, `A_complete`, `A_full_info`, `B`, `B_constant_income`, `C`, `C_no_specialization`, `D`, `quick*`
(see `configs/presets.py`; any parameter can be overridden with `--set key=value`).

## Layout
| Folder | Content |
|---|---|
| `configs/` | configuration dataclasses and named experimental conditions |
| `economy/` | goods graph, preferences, endowments, network/matching, offers, observations, ledger, batched world |
| `agents/` | GATv2 / goods-transformer policy, centralised critic, baselines |
| `learning/` | negotiation rollout, MAPPO, trainer, evaluation |
| `analysis/` | Walras/planner benchmarks, money metrics, change points, statistics |
| `scripts/` | command-line entry points |
| `tests/` | unit and smoke tests |
| `docs/` | `FRAMEWORK.md` (living specification, results log, changelog) and `HOW_IT_WORKS.md` (detailed plain-language explanation of every mechanic and of the training) |
| `CLAUDE.md` | operating manual: working rules, current state, experiment queue, pending decisions |
