# When Does Ranking Help? A Sparsity Analysis of Two-Stage Recommendation

**CS 289A Final Project · UC Berkeley · Spring 2026**

We train three recommender systems of increasing complexity on MovieLens 1M and measure how each holds up as training data shrinks. The central question: does adding a ranking stage on top of collaborative filtering justify its cost, and does data sparsity change the answer?

## Systems Compared

| System | Architecture | Complexity |
|--------|-------------|------------|
| **A: MF** | Matrix Factorization — user/item dot product scored with sigmoid | Low |
| **B: NCF** | Neural Collaborative Filtering — embeddings concatenated into an MLP | Medium |
| **C: Ranker** | MF retrieval (top-100 candidates) + BPR-trained ranking MLP | High |

All three are trained at five density levels $d \in \{1.0, 0.8, 0.6, 0.4, 0.2\}$. Hyperparameters for MF and NCF are tuned independently via Bayesian Optimization at $d = 1.0$, then held fixed across the sweep. Evaluation uses NDCG@10 and HR@10 on a fixed leave-one-out test set.

## Key Finding

MF outperforms both NCF and the two-stage Ranker on NDCG@10 at every density level — there is no crossover. A partial hit-rate crossover appears at $d \leq 0.4$, where the Ranker improves HR@10 by up to 1.2 percentage points over MF, but without a corresponding gain in ranking quality.

See `documents/report.pdf` for the full analysis.

## Repository Structure

```
cs289-ranking/
├── src/
│   ├── data.py         # Data loading, ID remapping, splits, negative sampling
│   ├── models.py       # MF, NCF, and Ranker implemented in PyTorch
│   ├── train.py        # Training loop (--model, --density, --device flags)
│   ├── evaluate.py     # Standalone test-set evaluator (NDCG@10, HR@10)
│   └── utils.py        # Shared evaluation utilities
├── notebooks/
│   ├── 01_eda.ipynb    # Exploratory data analysis of MovieLens 1M
│   └── plot_results.py # Generates figures/sparsity_results.png
├── figures/            # Result plots and EDA figures
├── documents/          # Final report (PDF)
├── sweep.sh            # SLURM script: full 3-model × 5-density sweep
├── download_data.sh    # Downloads MovieLens 1M into data/raw/ml-1m/
├── environment.yml     # Conda environment (CPU / local)
└── environment-gpu.yml # Conda environment (GPU / SCF cluster)
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ricardo-pc/cs289-ranking.git
cd cs289-ranking
```

### 2. Create the environment

**CPU (local, for EDA and data pipeline testing):**
```bash
conda env create -f environment.yml
conda activate cs289-ranking
```

**GPU (recommended for training):**
```bash
conda env create -f environment-gpu.yml
conda activate cs289-ranking-gpu

# Install PyTorch for your CUDA version (check with: nvidia-smi | grep "CUDA Version")
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download the dataset

```bash
bash download_data.sh
```

Downloads MovieLens 1M (~6 MB) into `data/raw/ml-1m/`. This is the only dataset used.

### 4. Validate the data pipeline

```bash
python src/data.py data/raw/ml-1m
```

Expected output:
```
n_users : 6,040
n_items : 3,706
train   : 988,129 interactions
val     : 6,040 rows
test    : 6,040 rows
All checks passed.
```

## Running Experiments

### Single training run

```bash
# Matrix Factorization
python src/train.py --model mf --density 1.0 \
    --emb-dim 256 --lr 8.65e-4 --l2 1.45e-6 --alpha 0.5 \
    --epochs 15 --batch-size 1024 --device cuda --seed 42

# Neural Collaborative Filtering
python src/train.py --model ncf --density 1.0 \
    --emb-dim 256 --mlp-layers 256 128 64 32 \
    --lr 7.14e-4 --l2 1e-6 --alpha 0.5 \
    --epochs 20 --batch-size 1024 --device cuda --seed 42

# Two-stage Ranker (requires a trained MF checkpoint)
python src/train.py --model ranker --density 1.0 \
    --mf-checkpoint checkpoints/mf_density1.0.pt \
    --emb-dim 256 --mlp-layers 64 32 \
    --epochs 20 --lr 1e-3 --l2 1e-5 \
    --batch-size 1024 --device cuda --seed 42
```

### Full sparsity sweep (15 runs: 3 models × 5 densities)

On a SLURM cluster with a GPU node:
```bash
sbatch sweep.sh
```

Results are written to `logs/sweep_<job_id>.log`.

### Evaluate a saved checkpoint

```bash
python src/evaluate.py --model mf --density 1.0 \
    --checkpoint checkpoints/mf_density1.0.pt \
    --emb-dim 256 --device cuda
```

### Reproduce the results figure

```bash
python notebooks/plot_results.py
# Saves figures/sparsity_results.pdf and figures/sparsity_results.png
```

## References

- Koren et al. (2009) — Matrix Factorization Techniques for Recommender Systems
- Hu et al. (2008) — Collaborative Filtering for Implicit Feedback Datasets
- He et al. (2017) — Neural Collaborative Filtering
- Rendle et al. (2009) — BPR: Bayesian Personalized Ranking from Implicit Feedback
- Rendle et al. (2020) — Neural Collaborative Filtering vs. Matrix Factorization Revisited
- Covington et al. (2016) — Deep Neural Networks for YouTube Recommendations
- Harper & Konstan (2015) — The MovieLens Datasets
