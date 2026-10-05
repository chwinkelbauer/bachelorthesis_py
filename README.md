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

### 3. Full Pipeline Execution (Expert Data -> Pre-Training -> Curriculum -> RL)
Execute the end-to-end curriculum and MAPPO training pipeline:
```bash
# Step 1: Collect expert dataset (25 episodes)
python main.py --collect --episodes 25

# Step 2: Run supervised pre-training on expert dataset (10 epochs)
python main.py --pretrain --epochs 10

# Step 3: Run curriculum learning (scaling drain rates)
python main.py --curriculum

# Step 4: Run full MAPPO RL experiment (20,000 steps)
python main.py --train --total-steps 20000
```

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
- **Resource Map Hub Decay**:
  $$P_{\text{resource}}(r, c) = \max\left(\text{min\_eff}, \exp\left(-\gamma \cdot \min_{h \in \text{hubs}} d((r,c), h)\right)\right)$$
- **Observation Dimension**:
  $$\text{Input Dim} = (\text{sight}^2 \times 3) + 2 + 3 + 3 = (5^2 \times 3) + 8 = 83$$
