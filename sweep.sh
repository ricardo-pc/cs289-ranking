#!/bin/bash
#SBATCH --job-name=sparsity_sweep
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep_%j.log

# Best configs from Bayesian Optimization
MF_EMB=256;  MF_LR=8.65e-04;  MF_L2=1.45e-06;  MF_ALPHA=0.5
NCF_EMB=256; NCF_LR=7.14e-04; NCF_L2=1.00e-06; NCF_ALPHA=0.5
RANKER_LAYERS="64 32"   # ranking MLP on top of frozen MF embeddings

mkdir -p checkpoints logs

for D in 1.0 0.8 0.6 0.4 0.2; do
    echo ""
    echo "========================================"
    echo " density = $D"
    echo "========================================"

    # MF — fast, runs first so ranker can load its checkpoint
    python src/train.py --model mf --density $D \
        --emb-dim $MF_EMB --lr $MF_LR --l2 $MF_L2 --alpha $MF_ALPHA \
        --epochs 15 --batch-size 1024 --device cuda --seed 42

    # Ranker — loads frozen MF embeddings, trains ranking MLP with BPR
    python src/train.py --model ranker --density $D \
        --mf-checkpoint checkpoints/mf_density${D}.pt \
        --emb-dim $MF_EMB --mlp-layers $RANKER_LAYERS \
        --epochs 20 --lr 1e-3 --l2 1e-5 \
        --batch-size 1024 --device cuda --seed 42

    # NCF — independent of MF, runs last
    python src/train.py --model ncf --density $D \
        --emb-dim $NCF_EMB --mlp-layers 256 128 64 32 \
        --lr $NCF_LR --l2 $NCF_L2 --alpha $NCF_ALPHA \
        --epochs 20 --batch-size 1024 --device cuda --seed 42

done

echo ""
echo "========================================"
echo " Training complete — running evaluation"
echo "========================================"

for D in 1.0 0.8 0.6 0.4 0.2; do
    echo ""
    echo "--- density=$D ---"

    python src/evaluate.py --model mf --density $D \
        --checkpoint checkpoints/mf_density${D}.pt \
        --emb-dim $MF_EMB --device cuda

    python src/evaluate.py --model ncf --density $D \
        --checkpoint checkpoints/ncf_density${D}.pt \
        --emb-dim $NCF_EMB --mlp-layers 256 128 64 32 --device cuda

    python src/evaluate.py --model ranker --density $D \
        --checkpoint checkpoints/ranker_density${D}.pt \
        --emb-dim $MF_EMB --mlp-layers $RANKER_LAYERS --device cuda

done

echo ""
echo "All done. Copy the NDCG@10 and HR@10 values above into the report."
