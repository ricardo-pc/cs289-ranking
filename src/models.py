"""
models.py - Model architectures for the sparsity analysis

Models:
  - NCF   : Neural Collaborative Filtering (He et al. 2017) - Model B
  - MF    : Matrix Factorization baseline                    - Model A  (TODO)
  - Ranker: MF retrieval + BPR-trained ranking MLP           - Model C  (TODO)

Loss:
  - confidence_weighted_bce : WMF-style BCE with per-sample confidence weights
"""

import torch
import torch.nn as nn
from typing import List


# Model B - Neural Collaborative Filtering (NCF)

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (He et al. 2017).

    Architecture:
        user_id -> user embedding (B, emb_dim) -+
                                                 +- concat -> MLP -> sigmoid -> y_hat
        item_id -> item embedding (B, emb_dim) -+

    The dot-product scoring of MF is replaced by an MLP,
    allowing the model to learn non-linear user-item interactions.

    Args:
        n_users    : number of unique users (embedding table rows)
        n_items    : number of unique items (embedding table rows)
        emb_dim    : embedding dimension k, size of p_u and q_i (default 64, from NCF paper)
        mlp_layers : hidden layer sizes, tower pattern halving each layer (default [256, 128, 64])
        dropout    : fraction of neurons randomly zeroed during training to reduce overfitting (0 = off)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        mlp_layers: List[int] = [256, 128, 64],
        dropout: float = 0.2, # dropout is a regularization technique that randomly zeros out 20% of neurons in a layer
    ):
        super().__init__()

        # Embedding tables: shape (n_entities, emb_dim).
        # embeddings capture interaction patterns, not linguistic meaning.
        # nn.Embedding(A, B) creates a learnable matrix of shape (A, B).
        # Indexing with a batch of IDs returns the corresponding rows.
        # Gradients flow back into only the rows that were looked up.
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # MLP: sequence of Linear -> ReLU -> Dropout blocks.
        # Input size is 2*emb_dim because we concatenate p_u and q_i.
        # Each block transforms z^(l-1) -> z^(l): applies Wz + b, then ReLU, then dropout.
        # nn.Sequential runs the layers in order — no need to call each one manually.
        layers = []
        input_size = 2 * emb_dim
        for hidden_size in mlp_layers:
            layers.append(nn.Linear(input_size, hidden_size))  # W^(l), b^(l)
            layers.append(nn.ReLU())                            # non-linearity
            layers.append(nn.Dropout(dropout))                  # regularization
            input_size = hidden_size                            # next layer's input = this layer's output

        # Output layer: maps last hidden layer -> single logit (no activation yet)
        # Sigmoid is applied in forward() rather than here so we can use
        # numerically stable BCE loss functions if needed later.
        layers.append(nn.Linear(input_size, 1))

        self.mlp = nn.Sequential(*layers)

        # Weight initialization: Xavier uniform for Linear layers, zeros for biases.
        # Xavier sets initial weights proportional to layer size, which keeps
        # gradients from vanishing or exploding at the start of training.
        self._init_weights()

    def _init_weights(self):
        # Normal initialization for embeddings (mean=0, std=0.01) — small random start
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        # Xavier uniform for MLP linear layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids : (B,) integer tensor of user indices
            item_ids : (B,) integer tensor of item indices
        Returns:
            y_hat    : (B,) float tensor of predicted interaction probabilities in (0, 1)
        """
        p_u = self.user_emb(user_ids)          # (B, emb_dim) — lookup rows from P
        q_i = self.item_emb(item_ids)          # (B, emb_dim) — lookup rows from Q
        x   = torch.cat([p_u, q_i], dim=1)    # (B, 2*emb_dim) — concatenate
        logit = self.mlp(x)                    # (B, 1) — pass through MLP
        return torch.sigmoid(logit).squeeze(1) # (B,) — squeeze out the last dim


# Model A - Matrix Factorization (MF)

class MF(nn.Module):
    """
    Matrix Factorization baseline.

    Architecture:
        user_id -> user embedding (B, emb_dim) -+
                                                 +- dot product -> sigmoid -> y_hat
        item_id -> item embedding (B, emb_dim) -+

    Scoring is a simple inner product: score = sigmoid(p_u · q_i).
    Compared to NCF, there is no MLP — interactions are assumed to be linear.

    Args:
        n_users : number of unique users (embedding table rows)
        n_items : number of unique items (embedding table rows)
        emb_dim : embedding dimension k (default 64, same as NCF for fair comparison)
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids : (B,) integer tensor of user indices
            item_ids : (B,) integer tensor of item indices
        Returns:
            y_hat    : (B,) float tensor of predicted interaction probabilities in (0, 1)
        """
        p_u = self.user_emb(user_ids)          # (B, emb_dim)
        q_i = self.item_emb(item_ids)          # (B, emb_dim)
        dot = (p_u * q_i).sum(dim=1)           # (B,) element-wise product then sum = dot product
        return torch.sigmoid(dot)              # (B,)


# Shared loss function (used by both NCF and MF)

def confidence_weighted_bce(
    y_hat: torch.Tensor,       # (B,) predicted probabilities in (0, 1)
    y: torch.Tensor,           # (B,) binary labels — 1 for positives, 0 for negatives
    confidence: torch.Tensor,  # (B,) per-sample weights — c_ui = 1 + alpha * rating for pos, 1 for neg
) -> torch.Tensor:
    """
    Confidence-weighted binary cross-entropy loss (Hu et al. 2008 / WMF).

    Standard BCE for one sample:
        bce = -(y * log(y_hat) + (1-y) * log(1 - y_hat))

    Weighted version — multiply each sample's BCE by its confidence weight:
        loss = mean(confidence * bce)

    For positives (y=1): confidence = c_ui = 1 + alpha * rating
        - (1-y) term vanishes: loss_i = -c_ui * log(y_hat_i)
        - higher-rated movies get penalized more if predicted incorrectly
    For negatives (y=0): confidence = 1 (unweighted)
        - y term vanishes: loss_i = -log(1 - y_hat_j)
        - all negatives treated equally

    Args:
        y_hat      : (B,) model output after sigmoid — values in (0, 1)
        y          : (B,) ground-truth labels in {0, 1}
        confidence : (B,) per-sample confidence weights (>= 1)
    Returns:
        scalar mean loss over the batch
    """
    # Clamp y_hat away from 0 and 1 to avoid log(0) = -inf
    # 1e-7 is a common choice — small enough not to distort probabilities
    eps = 1e-7
    y_hat = y_hat.clamp(min=eps, max=1.0 - eps)

    # Element-wise BCE: shape (B,)
    bce = -(y * torch.log(y_hat) + (1.0 - y) * torch.log(1.0 - y_hat))

    # Scale each sample's loss by its confidence weight, then average
    return (confidence * bce).mean()
