"""
utils.py - Shared evaluation utilities

Imported by both train.py (val set, called every epoch) and evaluate.py (test set, called once).
Keeping it here avoids duplicating the metric logic across scripts.
"""

import torch
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute NDCG@10 and HR@10 using the leave-one-out protocol (He et al. 2017).

    For each user:
        - 100 candidates: 1 positive (held-out item) + 99 pre-sampled negatives
        - Score all 100 with the model in one vectorized call
        - The positive is always at column 0 of the candidates tensor (from eval_collate)

    HR@10 (Hit Rate):
        1 if the positive item appears in the top 10, else 0.

    NDCG@10 (Normalized Discounted Cumulative Gain):
        1 / log2(rank + 1) if rank <= 10, else 0.
        Rewards ranking the positive item closer to the top.

    Args:
        model  : trained NCF or MF instance
        loader : DataLoader wrapping EvalDataset, using eval_collate
        device : torch.device
    Returns:
        (ndcg_at_10, hr_at_10) — mean over all users, values in [0, 1]
    """
    model.eval()

    ndcg_scores = []
    hr_scores   = []

    with torch.no_grad():
        for users, candidates, _ in loader:
            users      = users.to(device)
            candidates = candidates.to(device)

            B = users.size(0)

            # Score all 100 candidates per user in one forward pass
            users_expanded = users.repeat_interleave(100)  # (B*100,)
            items_flat     = candidates.view(-1)           # (B*100,)
            scores         = model(users_expanded, items_flat).view(B, 100)  # (B, 100)

            # Rank of the positive (column 0): count items scoring higher, add 1
            pos_score = scores[:, 0].unsqueeze(1)          # (B, 1)
            rank      = (scores > pos_score).sum(dim=1) + 1  # (B,) 1-indexed

            hit  = (rank <= 10).float()
            ndcg = torch.where(
                rank <= 10,
                1.0 / torch.log2(rank.float() + 1),
                torch.zeros(B, device=device),
            )

            ndcg_scores.append(ndcg.cpu())
            hr_scores.append(hit.cpu())

    return torch.cat(ndcg_scores).mean().item(), torch.cat(hr_scores).mean().item()
