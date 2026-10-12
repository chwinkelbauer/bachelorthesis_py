# Multi-Agent Grid World Economic Simulation & MAPPO (Python Port)

This project is a high-performance Python port of the Multi-Agent Grid World Economic Simulation originally developed in R for a Bachelor Thesis at WU Wien. 

It models economic agents navigating an $n \times k$ resource grid containing localized hubs of **Food**, **Wood**, and **Gold**, satisfying physiological needs (**Nutrition** and **Housing**), gathering resources subject to weight capacity and individual efficiencies, consuming inventory conservatively, engaging in barter trade, and learning multi-agent policies using PyTorch.

---

## 🏗️ Project Architecture & Structure

```
py/
├── config.py         # Simulation parameters, grid setup, agent initial states, pre-training settings
├── environment.py    # Resource map generation (exponential decay hubs), vision extractions, utility math
├── agent.py          # Agent state data structures, movement, gathering, dropping, consumption, death mechanics
├── network.py        # PyTorch Multi-Head Actor-Critic architecture with modular Encoder (MLP & GNN ready)
├── dataset.py        # Expert rule-based decision generator & PyTorch Dataset loader for supervised pre-training
├── training.py       # Supervised Pre-Training, Curriculum Learning, and MAPPO Reinforcement Learning loops
├── metrics.py        # Survival statistics (deaths, age & cause of death, survival rate) and global seeding
├── evaluation.py     # Learning-free evaluation of brain / expert / random policies on fresh maps
├── sim_check.py      # Monte Carlo balance checker & mathematical survival feasibility analyzer
├── main.py           # CLI entry point to run tests, dataset collection, pre-training, curriculum, and RL
├── requirements.txt  # Python dependency specification
└── README.md         # Documentation & guide
```

---

## 💡 Key Improvements in Python

1. **PyTorch Automatic Differentiation**: Replaces manual pure-R weight matrices, bias updates, and analytical backpropagation with PyTorch `nn.Module` and standard PyTorch optimizers (`Adam`/`AdamW`).
2. **Vectorized Spatial Operations**: NumPy matrix operations replace double R loops for vision extraction, spatial map decay computations, and observation building.
3. **Modular Encoder Abstraction for GNNs**: `MultiHeadGlobalBrain` features an extensible `encoder_type` parameter (`mlp` or `gnn`). The `GNNEncoder` class is pre-configured to plug in Graph Neural Networks (e.g. PyTorch Geometric `GCNConv` / `GATConv`) to model spatial agent topology and communication graphs.

## ⚖️ Economic Anti-Hoarding & Spoilage Mechanics

1. **Differentiated Inventory Spoilage**:
   - **Food ($f$)**: High perishability ($\approx 2.5\%$ per step) with super-linear accelerated decay on stockpiles ($>3.0$ units).
   - **Wood ($w$)**: Moderate weatherability ($\approx 0.5\%$ per step) with super-linear holding decay.
   - **Gold ($g$)**: **0.0% decay** (immutable and permanent store of value).
2. **Realistic Weight Limits & Overburden Penalty**:
   - Total inventory capacity limit: $\sigma_{\max} = 50.0$ weight units.
   - Unit weights: $\text{Food} = 0.5$, $\text{Wood} = 1.0$, $\text{Gold} = 0.2$ (light-weight, cheap to carry as a medium of exchange).
   - **Overburden Cost ($> 70\%$ capacity)**:
     - **Metabolic Exhaustion**: Heavy carrying load increases passive $N$ and $H$ drain by up to $+30\%$.
     - **Gathering Penalty**: Overfilled inventory lowers gathering efficacy by up to $-35\%$.
     - **Disutility**: Continuous negative reward penalty disincentivizes carrying useless dead weight.

## 👁️ Observation & Action Masking

1. **Agent Perception**: The vision window has four channels: Food, Wood, Gold and **Other Agents** (count of other living agents per tile), so agents can perceive potential trading partners and trade networks around them.
2. **Movement Action Masking**: Moves that would leave the grid or enter a tile occupied by another living agent are masked (logits set to $-10^9$) during sampling, PPO evaluation and supervised pre-training, so each tile holds at most one agent. Controlled by `mask_occupied_tiles` in `EnvConfig` (default `True`).
3. **Continuous Consumption Amounts**: The consumed inventory fraction is sampled from a Beta distribution $\text{Beta}(\alpha, \beta)$ over $(0, 1)$ with $\alpha, \beta \ge 1$ predicted by the network, and trained with the PPO ratio objective (pre-training: negative log-likelihood of the expert fraction). Not consuming at all is chosen via the consume type **None**, in which case the amount is ignored in all losses. The Beta entropy uses its own coefficient `mappo.consume_entropy_coeff`.

## 🎓 Curriculum Episodes

Curriculum episodes are capped at `pre_training.episode_max` steps (default 500) and agents and map are also reset at every stage change, so every drain level, including the final 100% phase, is trained from the true starting state instead of from reserves accumulated at low drain. Truncated trajectories are bootstrapped with the critic before the reset. The progress log reports, for finished episodes only, how many episodes died out vs. reached the cap, how many agents survived to the end, and the number, mean age and cause (hunger / cold) of deaths. Episodes cut short by a stage change are not counted. `--train` reports the same statistics per epoch and for the whole run.

---

## 🚀 Quickstart & Usage

### 1. Verification & Sanity Check
Run a quick 10-step simulation sanity check to verify neural network gradient flow:
```bash
python main.py --test
```

### 2. Monte Carlo Simulation Balance Check
Run 200 random map Monte Carlo analysis of resource yields, passive lifespans, and survival feasibility:
```bash
python main.py --check
```

### 3. Using the Pre-Trained Neural Network

You can use the pre-trained neural network in **two ways**:

#### Option A: Chained Pipeline Command (In-Memory Transfer)
Run pre-training, curriculum learning, and RL training in a single chained command. The pre-trained weights will be passed directly in memory to the RL training stage:

```bash
# Run Pre-Training + Curriculum + RL Training in one go
python main.py --collect --episodes 25
python main.py --pretrain --epochs 10 --curriculum --train --total-steps 20000
```

#### Option B: Save & Load Weight Checkpoints (`.pt` files)
Save pre-trained weights to a file and load them into subsequent training sessions or evaluations:

```bash
# 0. Expert dataset generation
python main.py --collect

# 1. Pre-train neural net and save weights to 'pretrained_brain.pt'
python main.py --pretrain --epochs 10 --save-brain pretrained_brain.pt

# 2. Run curriculum training on pre-trained weights and save result
python main.py --load-brain pretrained_brain.pt --curriculum --save-brain curriculum_brain.pt

# 2b. (Optional) Curriculum with N extra steps at 100% drain appended after the ramp
python main.py --load-brain pretrained_brain.pt --curriculum --full-drain-steps 5000 --save-brain curriculum_brain.pt

# 2c. (Optional) Continue an existing curriculum brain at 100% drain only (ramp skipped)
python main.py --load-brain curriculum_brain.pt --full-drain-steps 5000 --save-brain curriculum_brain.pt

# 3. Run RL training using the curriculum-trained brain weights
python main.py --load-brain curriculum_brain.pt --train --total-steps 5000

```

### 4. Reproducibility & Evaluation

```bash
# Any command can be made reproducible with a seed (Python, NumPy, PyTorch)
python main.py --seed 42 --load-brain pretrained_brain.pt --curriculum --save-brain curriculum_brain.pt

# Evaluate a brain on 20 fresh maps x 500 steps at 100% drain without learning
python main.py --load-brain curriculum_brain.pt --evaluate --eval-episodes 20 --seed 1

# Same maps (same seed) with argmax/mean actions, and the baselines
python main.py --load-brain curriculum_brain.pt --evaluate --eval-episodes 20 --seed 1 --deterministic
python main.py --evaluate --eval-policy expert --eval-episodes 20 --seed 1
python main.py --evaluate --eval-policy random --eval-episodes 20 --seed 1
```
The summary reports the survival rate with a 95% confidence interval over episodes, deaths with mean age and cause, and the final utility, needs and gold holdings of survivors.

---

## 🧩 Graph Neural Network (GNN) Integration

To swap the standard MLP backbone for a Graph Neural Network:
1. Pass `--encoder gnn` to `main.py`:
   ```bash
   python main.py --test --encoder gnn
   ```
2. Extend `GNNEncoder` in `network.py` by incorporating `torch_geometric.nn.GCNConv` or custom graph adjacency matrices derived from agent spatial proximity.

---

## 📐 Mathematical Formulation

- **Logarithmic Utility Function**:
  $$U_i = w_n \ln(\max(0, N_i) + 1) + w_h \ln(\max(0, H_i) + 1) + w_g \ln(\max(0, G_i) + 1)$$
  with $(w_n, w_h, w_g) = (1.0, 1.2, 0.05)$: gold carries only a small intrinsic utility, so its value stems mainly from exchange.
- **Super-Linear Spoilage Loss**:
  $$\text{Spoil}_f = \text{rate}_f \cdot f \cdot \left(1 + \left(\frac{\max(0, f - \theta_f)}{\theta_f}\right)^{\beta_f}\right)$$
- **Resource Map Hub Decay**:
  $$P_{\text{resource}}(r, c) = \max\left(\text{min\_eff}, \exp\left(-\gamma \cdot \min_{h \in \text{hubs}} d((r,c), h)\right)\right)$$
- **Observation Dimension**:
  $$\text{Input Dim} = (\text{sight}^2 \times 4) + 2 + 3 + 3 = (5^2 \times 4) + 8 = 108$$
