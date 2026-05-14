"""
evaluate.py - Standalone test-set evaluator

Loads a saved checkpoint and evaluates it on the test set.
This is the only script that touches the test set — never called during training.

Usage:
  python src/evaluate.py --model ncf --density 1.0 --checkpoint checkpoints/ncf_density1.0.pt
  python src/evaluate.py --model ncf --density 0.4 --checkpoint checkpoints/ncf_density0.4.pt --device cuda
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import load_ml1m, build_eval_negatives, EvalDataset, eval_collate
from models import NCF, MF, Ranker
from utils import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the test set")

    parser.add_argument("--data-dir",    type=str, default="data/raw/ml-1m")
    parser.add_argument("--density",     type=float, default=1.0,
                        help="density the model was trained on — used to load the right data split")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="path to saved .pt checkpoint, e.g. checkpoints/ncf_density1.0.pt")
    parser.add_argument("--model",       type=str, default="ncf", choices=["ncf", "mf", "ranker"])
    parser.add_argument("--emb-dim",     type=int, default=64)
    parser.add_argument("--mlp-layers",  type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--device",      type=str, default="cpu")

    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading data  (density={args.density}) ...")
    data = load_ml1m(args.data_dir, density=args.density)

    # Always build negatives on full-density data
    if args.density < 1.0:
        full_data = load_ml1m(args.data_dir, density=1.0)
        negatives = build_eval_negatives(full_data, n_neg=99, seed=42)
    else:
        negatives = build_eval_negatives(data, n_neg=99, seed=42)

    test_ds     = EvalDataset(data, negatives, split="test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0, collate_fn=eval_collate)

    # Build model with same architecture as checkpoint
    print(f"Loading {args.model.upper()} from {args.checkpoint} ...")
    if args.model == "ncf":
        model = NCF(
            n_users    = data.n_users,
            n_items    = data.n_items,
            emb_dim    = args.emb_dim,
            mlp_layers = args.mlp_layers,
            dropout    = args.dropout,
        ).to(device)
    elif args.model == "mf":
        model = MF(
            n_users = data.n_users,
            n_items = data.n_items,
            emb_dim = args.emb_dim,
        ).to(device)
    else:  # ranker
        model = Ranker(
            n_users    = data.n_users,
            n_items    = data.n_items,
            emb_dim    = args.emb_dim,
            mlp_layers = args.mlp_layers,
            dropout    = args.dropout,
        ).to(device)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # Evaluate on test set
    print("Evaluating on test set ...")
    test_ndcg, test_hr = evaluate(model, test_loader, device)

    print(f"\nModel      : {args.model.upper()}")
    print(f"Density    : {args.density}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"NDCG@10    : {test_ndcg:.4f}")
    print(f"HR@10      : {test_hr:.4f}")


if __name__ == "__main__":
    main()
