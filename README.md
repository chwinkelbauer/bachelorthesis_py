# Money from Scratch

A multi-agent reinforcement learning economy (Bachelor thesis, WU Wien) in which **money is not built in**.
Agents have randomised, agent-specific log utilities and trade bilaterally on a network by learned negotiation.
Money is identified functionally (an otherwise useless good that is accepted and passed on) and tested against
null models and theoretical benchmarks.

**The full specification (model, mathematics, metrics, design, status and changelog) is in
[docs/FRAMEWORK.md](docs/FRAMEWORK.md).** That document is kept in sync with the code.

## Quickstart
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .           # invariants + smoke tests
python -m scripts.train --preset A --seed 0          # train stage A; writes results/A_s0/
python -m scripts.evaluate --run-dir results/A_s0    # held-out evaluation vs. random null and ZI-C
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
| `docs/` | living framework specification |
