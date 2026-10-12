"""
PyTorch Neural Network architecture for Multi-Head Actor-Critic Global Brain.
Includes modular Encoder abstraction to allow seamless integration of Graph Neural Networks (GNNs).
Ported and modernized from R global_brain.R and nn_engine.R to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

MASK_FILL_VALUE = -1e9  # Logit assigned to masked (invalid) actions
CONSUME_AMT_EPS = 1e-3  # Consume fractions are clipped to [eps, 1 - eps] for the Beta policy (not consuming = consume type "None")


class BaseEncoder(nn.Module):
    """Abstract Base Class for state representation encoders."""
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MLPEncoder(BaseEncoder):
    """
    Standard Multi-Layer Perceptron Encoder for flat observation vectors.
    """
    def __init__(self, input_dim: int = 108, hidden_dim: int = 64):
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
    def __init__(self, input_dim: int = 108, hidden_dim: int = 64):
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
    def __init__(self, input_dim: int = 108, hidden_dim: int = 64, encoder_type: str = "mlp"):
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
        self.head_consume_amt = nn.Linear(hidden_dim, 2)    # Beta(alpha, beta) parameters of the Consume Fraction (0..1)
        self.critic_head = nn.Linear(hidden_dim, 1)         # Critic State-Value V(s)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass yielding head outputs.
        x: Tensor of shape (batch_size, input_dim) or (input_dim,)
        action_mask: Optional bool Tensor of shape (batch_size, 11) or (11,); False entries are masked out.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        shared_repr = self.encoder(x, edge_index)

        logits_action = self.head_action(shared_repr)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits_action = logits_action.masked_fill(~action_mask, MASK_FILL_VALUE)
        logits_barter = self.head_barter(shared_repr)
        logits_consume_type = self.head_consume_type(shared_repr)

        drop_amt = torch.sigmoid(self.head_drop_amt(shared_repr))
        # alpha, beta >= 1 keeps the Beta density unimodal and finite at the interval borders
        consume_ab = F.softplus(self.head_consume_amt(shared_repr)) + 1.0

        v_pred = self.critic_head(shared_repr)

        return {
            "logits_action": logits_action,
            "logits_barter": logits_barter,
            "logits_consume_type": logits_consume_type,
            "drop_amt": drop_amt,
            "consume_alpha": consume_ab[:, 0],
            "consume_beta": consume_ab[:, 1],
            "v_pred": v_pred
        }

    @staticmethod
    def consume_distribution(out: Dict[str, torch.Tensor]) -> torch.distributions.Beta:
        """Continuous consume-fraction policy Beta(alpha, beta) over (0, 1)."""
        return torch.distributions.Beta(out["consume_alpha"], out["consume_beta"])

    def evaluate_actions_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        consume_types: torch.Tensor,
        consume_amts: torch.Tensor,
        drop_amts: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates a batch of states and actions for PPO policy gradient and value updates.
        consume_amts: sampled consume fractions in (0, 1).
        """
        # Strikte Konvertierung ohne fehlerhaftes .max()-Verhalten
        act_idx = actions - 1 if actions.min() >= 1 else actions
        ct_idx = consume_types - 1 if consume_types.min() >= 1 else consume_types

        out = self.forward(states, action_mask=action_masks)

        # Aktions-Log-Probs und Entropie
        dist_act = torch.distributions.Categorical(logits=out["logits_action"])
        log_prob_act = dist_act.log_prob(act_idx)
        entropy_act = dist_act.entropy()

        # Konsumtyp-Log-Probs und Entropie
        dist_ct = torch.distributions.Categorical(logits=out["logits_consume_type"])
        log_prob_ct = dist_ct.log_prob(ct_idx)
        entropy_ct = dist_ct.entropy()

        # Konsummengen-Log-Probs und Entropie
        dist_ca = self.consume_distribution(out)
        log_prob_ca = dist_ca.log_prob(consume_amts.clamp(CONSUME_AMT_EPS, 1.0 - CONSUME_AMT_EPS))
        entropy_ca = dist_ca.entropy()

        return {
            "v_pred": out["v_pred"].squeeze(-1),
            "log_prob_act": log_prob_act,
            "entropy_act": entropy_act,
            "log_prob_ct": log_prob_ct,
            "entropy_ct": entropy_ct,
            "log_prob_ca": log_prob_ca,
            "entropy_ca": entropy_ca,
            "drop_amt": out["drop_amt"].squeeze(-1)
        }

    def predict_action(self, state_vec: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Inference helper to derive argmax discrete decisions and continuous fractions.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(state_vec, action_mask=action_mask)
            action_idx = torch.argmax(out["logits_action"], dim=-1).item() + 1  # 1-indexed to match R
            barter_flag = (torch.argmax(out["logits_barter"], dim=-1).item() == 1)
            consume_type = torch.argmax(out["logits_consume_type"], dim=-1).item() + 1  # 1-indexed (1: None, 2: Food, 3: Wood)
            drop_fraction = out["drop_amt"].squeeze().item()
            consume_fraction = self.consume_distribution(out).mean.squeeze().item()
            v_pred = out["v_pred"].squeeze().item()

        return {
            "action_idx": action_idx,
            "barter_flag": barter_flag,
            "consume_type": consume_type,
            "drop_fraction": drop_fraction,
            "consume_fraction": consume_fraction,
            "v_pred": v_pred
        }
