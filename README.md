# CS289 final project

# When Does Ranking Help? A Sparsity Analysis of Two-Stage Recommendation

Ricardo Perez Castillo, [Teammate 2], [Teammate 3] · CS 289A, UC Berkeley · Spring 2026

## Overview

We study how recommendation quality degrades as per-user interaction history shrinks,
comparing three systems of increasing complexity on MovieLens 1M:

| System | Description |
|--------|-------------|
| **MF** | Matrix Factorization (dot-product scoring) |
| **NCF** | Neural Collaborative Filtering (MLP scoring) |
| **MF + Ranker** | MF retrieval + BPR-trained ranking MLP |

**Core question:** Below how many interactions per user does the ranking stage stop helping?

## Key finding
- TBD


## Setup

```bash
pip install -r requirements.txt
# Download MovieLens 1M to data/ from https://grouplens.org/datasets/movielens/1m/
```

## Reproducing results

```bash
python src/train.py --model mf --density 1.0
python src/train.py --model ncf --density 0.6
python src/train.py --model ranker --density 0.4
# or run all experiments via the sparsity sweep script
python src/sparsity.py --densities 0.2 0.4 0.6 0.8 1.0
```

## Evaluation
- TBD



## References
- TBD


cs289-ranking/
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results_analysis.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── sparsity.py
├── experiments/
│   └── configs/
├── figures/
├── report/
├── requirements.txt
└── README.md
