"""
PyTorch Neural Network architecture for Multi-Head Actor-Critic Global Brain.
Includes modular Encoder abstraction to allow seamless integration of Graph Neural Networks (GNNs).
Ported and modernized from R global_brain.R and nn_engine.R to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class BaseEncoder(nn.Module):
    """Abstract Base Class for state representation encoders."""
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MLPEncoder(BaseEncoder):
    """
    Standard Multi-Layer Perceptron Encoder for flat observation vectors.
    """
    def __init__(self, input_dim: int = 83, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.activation(self.fc(x))


class GNNEncoder(BaseEncoder):
    """
    Modular Extension Placeholder for Graph Neural Networks (GNN).
    Processes agent interaction graphs / spatial grid topology graphs.
    """
    def __init__(self, input_dim: int = 83, hidden_dim: int = 64):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Note: torch_geometric GCNConv / GATConv can be easily integrated here!

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Fallback dense feature transformation if no graph structure is supplied
        x_node = self.node_mlp(x)
        if edge_index is not None:
            # Future GNN message passing step can be executed here
            pass
        return F.relu(x_node)


class MultiHeadGlobalBrain(nn.Module):
    """
    Multi-Head Actor-Critic Architecture shared across agents.
    Outputs discrete policy logits, continuous control fractions, and scalar critic values.
    """
    def __init__(self, input_dim: int = 83, hidden_dim: int = 64, encoder_type: str = "mlp"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder selection (MLP or GNN)
        if encoder_type.lower() == "gnn":
            self.encoder: BaseEncoder = GNNEncoder(input_dim, hidden_dim)
        else:
            self.encoder: BaseEncoder = MLPEncoder(input_dim, hidden_dim)

        # Policy & Value Heads
        self.head_action = nn.Linear(hidden_dim, 11)        # Discrete Action (1-11)
        self.head_barter = nn.Linear(hidden_dim, 2)         # Barter Flag (No, Yes)
        self.head_consume_type = nn.Linear(hidden_dim, 3)   # Consume Type (None, Food, Wood)
        self.head_drop_amt = nn.Linear(hidden_dim, 1)       # Continuous Drop Fraction (0..1)
        self.head_consume_amt = nn.Linear(hidden_dim, 1)    # Continuous Consume Fraction (0..1)
        self.critic_head = nn.Linear(hidden_dim, 1)         # Critic State-Value V(s)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding head outputs.
        x: Tensor of shape (batch_size, input_dim) or (input_dim,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        shared_repr = self.encoder(x, edge_index)

        logits_action = self.head_action(shared_repr)
        logits_barter = self.head_barter(shared_repr)
        logits_consume_type = self.head_consume_type(shared_repr)

        drop_amt = torch.sigmoid(self.head_drop_amt(shared_repr))
        consume_amt = torch.sigmoid(self.head_consume_amt(shared_repr))

        v_pred = self.critic_head(shared_repr)

        return {
            "logits_action": logits_action,
            "logits_barter": logits_barter,
            "logits_consume_type": logits_consume_type,
            "drop_amt": drop_amt,
            "consume_amt": consume_amt,
            "v_pred": v_pred
        }

    def evaluate_actions_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        consume_types: torch.Tensor,
        consume_amts: Optional[torch.Tensor] = None,
        drop_amts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates batch state-action pairs for PPO mini-batch updates.
        Computes new log-probabilities, policy entropies, continuous control outputs, and critic state-values.
        """
        out = self.forward(states)

        # Discrete Action Policy Distribution
        dist_act = torch.distributions.Categorical(logits=out["logits_action"])
        # If actions are 1-indexed, convert to 0-indexed
        act_idx = actions if actions.max() < 11 else (actions - 1)
        log_prob_act = dist_act.log_prob(act_idx)
        entropy_act = dist_act.entropy()

        # Discrete Consume Type Policy Distribution
        dist_ct = torch.distributions.Categorical(logits=out["logits_consume_type"])
        ct_idx = consume_types if consume_types.max() < 3 else (consume_types - 1)
        log_prob_ct = dist_ct.log_prob(ct_idx)
        entropy_ct = dist_ct.entropy()

        return {
            "log_prob_act": log_prob_act,
            "entropy_act": entropy_act,
            "log_prob_ct": log_prob_ct,
            "entropy_ct": entropy_ct,
            "consume_amt": out["consume_amt"].squeeze(-1),
            "drop_amt": out["drop_amt"].squeeze(-1),
            "v_pred": out["v_pred"].squeeze(-1)
        }

    def predict_action(self, state_vec: torch.Tensor) -> Dict[str, Any]:
        """
        Inference helper to derive argmax discrete decisions and continuous fractions.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(state_vec)
            action_idx = torch.argmax(out["logits_action"], dim=-1).item() + 1  # 1-indexed to match R
            barter_flag = (torch.argmax(out["logits_barter"], dim=-1).item() == 1)
            consume_type = torch.argmax(out["logits_consume_type"], dim=-1).item() + 1  # 1-indexed (1: None, 2: Food, 3: Wood)
            drop_fraction = out["drop_amt"].squeeze().item()
            consume_fraction = out["consume_amt"].squeeze().item()
            v_pred = out["v_pred"].squeeze().item()

        return {
            "action_idx": action_idx,
            "barter_flag": barter_flag,
            "consume_type": consume_type,
            "drop_fraction": drop_fraction,
            "consume_fraction": consume_fraction,
            "v_pred": v_pred
        }
