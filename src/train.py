"""
train.py - Training loop for MF and NCF models

Accepts CLI flags so the same script handles every (model, density, hyperparameter) combo.
The BO outer loop calls this script repeatedly with different hyperparameter configs.

Usage:
  # single run with defaults
  python src/train.py --model ncf --density 1.0 --device cuda

  # custom hyperparameters (e.g. from BO)
  python src/train.py --model ncf --density 0.6 --emb-dim 128 --lr 5e-4 --alpha 2.0

  # CPU (local testing)
  python src/train.py --model ncf --density 1.0 --device cpu --epochs 5
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports — run from repo root so Python finds src/
from data import load_ml1m, build_eval_negatives, TrainDataset, EvalDataset, eval_collate
from models import NCF, confidence_weighted_bce
from utils import evaluate


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Every hyperparameter that BO needs to tune is exposed as a flag with a
    sensible default so the script runs without any arguments during development.
    """
    parser = argparse.ArgumentParser(description="Train MF or NCF on MovieLens 1M")

    # Data
    parser.add_argument("--data-dir",  type=str,   default="data/raw/ml-1m",
                        help="path to the raw ml-1m folder")
    parser.add_argument("--density",   type=float, default=1.0,
                        help="fraction of training interactions to keep (0 < d <= 1.0)")

    # Model
    parser.add_argument("--model",     type=str,   default="ncf", choices=["ncf", "mf"],
                        help="which model architecture to train")
    parser.add_argument("--emb-dim",   type=int,   default=64,
                        help="embedding dimension k (size of p_u and q_i)")
    parser.add_argument("--mlp-layers", type=int,  nargs="+", default=[256, 128, 64],
                        help="NCF hidden layer sizes, e.g. --mlp-layers 256 128 64")
    parser.add_argument("--dropout",   type=float, default=0.2,
                        help="dropout rate in NCF MLP layers (0 = off)")

    # Training
    parser.add_argument("--epochs",    type=int,   default=20,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int,  default=1024,
                        help="training batch size")
    parser.add_argument("--lr",        type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--l2",        type=float, default=1e-5,
                        help="L2 weight decay on embeddings (ridge regularization)")
    parser.add_argument("--alpha",     type=float, default=1.0,
                        help="WMF confidence scaling: c_ui = 1 + alpha * rating")
    parser.add_argument("--n-neg",     type=int,   default=4,
                        help="negative samples per positive in training")

    # Infrastructure
    parser.add_argument("--device",    type=str,   default="cpu",
                        help="'cuda', 'mps', or 'cpu'")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--save-dir",  type=str,   default="checkpoints",
                        help="directory to save best model weights")

    return parser.parse_args()


# Training loop for one epoch

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training data and return the mean loss.

    Each batch from TrainDataset contains:
        user   : (B,) user indices
        pos    : (B,) positive item indices (movies the user rated)
        rating : (B,) raw star rating 1-5 for the positive interaction
        neg    : (B,) negative item indices (movies the user did not rate)

    For each batch we:
        1. Score positives and negatives through the model separately
        2. Concatenate into a single (2B,) vector alongside labels and confidence weights
        3. Compute confidence-weighted BCE loss
        4. Backpropagate and update weights

    Args:
        model     : NCF or MF instance (already on device)
        loader    : DataLoader wrapping TrainDataset
        optimizer : Adam optimizer
        alpha     : WMF confidence scaling hyperparameter
        device    : torch.device to run on
    Returns:
        mean loss over all batches in this epoch (float, for logging)
    """
    model.train()  # enable dropout — only active during training, not evaluation

    total_loss = 0.0
    n_batches  = 0

    for user, pos, rating, neg in loader:

        # Move everything to GPU/CPU
        user   = user.to(device)    # (B,)
        pos    = pos.to(device)     # (B,)
        rating = rating.to(device)  # (B,)
        neg    = neg.to(device)     # (B,)

        # Forward pass: score positive pairs and negative pairs
        pos_score = model(user, pos)  # (B,) — predicted probability for (user, pos item)
        neg_score = model(user, neg)  # (B,) — predicted probability for (user, neg item)

        # Labels: positives = 1, negatives = 0
        # torch.ones / torch.zeros match the batch size and live on the same device
        B      = user.size(0)
        y      = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])   # (2B,)
        y_hat  = torch.cat([pos_score, neg_score])                                          # (2B,)

        # Confidence weights:
        #   positives get c_ui = 1 + alpha * rating (higher-rated = higher weight)
        #   negatives get c = 1 (unobserved — no rating signal, treated equally)
        pos_conf = 1.0 + alpha * rating                         # (B,)
        neg_conf = torch.ones(B, device=device)                 # (B,)
        confidence = torch.cat([pos_conf, neg_conf])            # (2B,)

        # Loss, backward pass, weight update
        loss = confidence_weighted_bce(y_hat, y, confidence)

        optimizer.zero_grad()  # clear gradients from previous batch
        loss.backward()        # compute gradients via backprop (chain rule through MLP)
        optimizer.step()       # update weights: w = w - lr * grad

        total_loss += loss.item()  # .item() converts 0-dim tensor to a plain Python float
        n_batches  += 1

    return total_loss / n_batches  # mean loss over the epoch


# Main: wires everything together

def main():
    args   = parse_args()
    device = torch.device(args.device)

    # Reproducibility: fix all random seeds so two runs with the same args give the same result
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data at the requested density level
    print(f"Loading data  (density={args.density}) ...")
    data = load_ml1m(args.data_dir, density=args.density, seed=args.seed)
    print(f"  train: {len(data.train):,}  val: {len(data.val):,}  test: {len(data.test):,}")

    # Eval negatives must be built on full-density data so no real interactions leak in.
    # If density < 1.0 we reload the full dataset just for this step, then discard it.
    if args.density < 1.0:
        full_data = load_ml1m(args.data_dir, density=1.0, seed=args.seed)
        negatives = build_eval_negatives(full_data, n_neg=99, seed=args.seed)
    else:
        negatives = build_eval_negatives(data, n_neg=99, seed=args.seed)

    # DataLoaders
    train_ds = TrainDataset(data, n_neg_per_pos=args.n_neg, seed=args.seed)
    val_ds   = EvalDataset(data, negatives, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256,             shuffle=False,
                              num_workers=0, collate_fn=eval_collate)

    # Model
    print(f"Building {args.model.upper()} (emb_dim={args.emb_dim}, mlp_layers={args.mlp_layers}) ...")
    if args.model == "ncf":
        model = NCF(
            n_users    = data.n_users,
            n_items    = data.n_items,
            emb_dim    = args.emb_dim,
            mlp_layers = args.mlp_layers,
            dropout    = args.dropout,
        ).to(device)
    else:
        raise NotImplementedError("MF not yet implemented — coming soon")

    # Adam optimizer with L2 weight decay (ridge regularization on all parameters)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # Checkpoint directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.model}_density{args.density}.pt"

    # Training loop
    print(f"\nTraining for {args.epochs} epochs on {device} ...")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'NDCG@10':>9}  {'HR@10':>7}  {'Time':>6}")

    best_ndcg = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss        = train_one_epoch(model, train_loader, optimizer, args.alpha, device)
        val_ndcg, val_hr  = evaluate(model, val_loader, device)
        elapsed           = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>8.4f}  {val_ndcg:>9.4f}  {val_hr:>7.4f}  {elapsed:>5.1f}s")

        # Save checkpoint whenever val NDCG improves
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), ckpt_path)

    # Load best checkpoint and report final val metrics
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    final_ndcg, final_hr = evaluate(model, val_loader, device)

    print(f"\nBest val  NDCG@10 = {final_ndcg:.4f}  HR@10 = {final_hr:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
