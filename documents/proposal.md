# CS 289A Project: Internal Team Document

## When Does the Ranking Stage Help? A Sparsity Analysis of Two-Stage Recommendation

Ricardo Perez Castillo, Teammate 2, Teammate 3

---

## What we're doing and why

We're building three recommendation systems of increasing complexity and measuring how each one performs as we artificially reduce how much data each user has. The goal is to find the crossover point: the number of per-user interactions below which the more complex ranking stage stops helping and may even hurt.

### The three systems

| System | What it does | Complexity |
|--------|-------------|------------|
| **A: Matrix Factorization (MF)** | Learns a user embedding and an item embedding, scores by dot product. This is the simplest collaborative filtering baseline. | Low |
| **B: Neural Collaborative Filtering (NCF)** | Same embeddings, but replaces the dot product with a multi-layer perceptron (MLP). Can capture nonlinear user-item interactions. | Medium |
| **C: MF + Ranking MLP** | Uses MF to retrieve a candidate set, then a second MLP re-ranks those candidates. The ranker is trained with Bayesian Personalized Ranking (BPR) loss, a pairwise objective that pushes the score of observed (positive) items above unobserved (negative) items. | High |

### The sparsity experiment

For each user, we subsample their training interactions to **80%, 60%, 40%, and 20%** of their original history. We retrain all three systems at each level and evaluate. This gives us a curve of performance vs. density for each system, and we can read off where the curves cross.

### Evaluation metrics

- **NDCG@10** (Normalized Discounted Cumulative Gain at 10): measures ranking quality of the top 10 recommendations, giving more credit to relevant items ranked higher.
- **Hit Rate@10**: binary — did the held-out item appear anywhere in the top 10?

These metrics are standard in the RecSys literature and are what He et al. (2017) use.

### Dataset

**MovieLens 1M**: 1 million ratings from ~6,000 users on ~4,000 movies. Well-studied, clean, and small enough to iterate fast. We use leave-one-out evaluation (hold out each user's most recent interaction for testing).

---

## Why this project (real motivation)

### For the class
The rubric rewards a controlled comparison under matched conditions with a clear hypothesis. Our hypothesis is that more expressive models degrade faster under sparsity due to higher variance (bias-variance tradeoff from lecture). The sparsity sweep is the experiment we designed, and the crossover point is the finding.

### For interviews
This project maps directly to real data science roles at companies like Uber, Spotify, and any two-sided marketplace. The core insight "complex ranking models stop helping for cold-start users", has immediate business implications:
- Where should you invest modeling complexity vs. fall back to simpler heuristics?
- How much user data do you need before deploying a ranker for a new user segment?
- For new users entering a platform continuously, what's the right system to serve them?

The interview pitch: *"I studied when adding a ranking stage on top of collaborative filtering stops helping for users with sparse histories. Below N interactions, simpler retrieval actually wins. That crossover tells you where to invest in modeling complexity."*

---

## Course connections 

| Course topic | How it shows up |
|-------------|----------------|
| Gradient descent (Lec 10-11) | MF minimizes regularized squared-error over embedding matrices U, V via SGD |
| Cross-entropy + sigmoid (HW3 P1) | NCF and the ranking MLP use binary cross-entropy with sigmoid output |
| Backpropagation (HW3 P3) | Gradients flow through MLP layers into embeddings via the chain rule |
| L2 regularization | Applied to embedding matrices; directly analogous to regularized ERM from lecture |
| Bias-variance tradeoff | The main analytical lens: more expressive models = higher variance = worse under sparsity |
| BPR as logistic regression on pairs | L_BPR = -sum log σ(r_ui - r_uj) is logistic regression on pairwise score differences |

---

## Division of responsibilities

| Member | Responsibilities |
|--------|-----------------|
| Ricardo | TBD |
| Teammate 2 | TBD |
| Teammate 3 | TBD |
| All | TBD |

---


## Implementation notes

### Key decisions to make early
- **Embedding dimension k**: MAYBE start with k=64, might sweep {32, 64, 128}
- **Train/test split**: leave-one-out (standard for MovieLens evaluations)
- **Negative sampling**: sample 99 negatives per positive for evaluation (standard protocol)
- **Candidate set size for System C**: MF retrieves top-100 candidates, ranker reranks to top-10

### Libraries
- **PyTorch** for all models
- **NumPy/Pandas** for data processing
- **Matplotlib/Seaborn** for figures
- Consider **Weights & Biases** for experiment tracking (rubric's practical advice says "log everything")

### What the main figure should look like
An x-axis of per-user interaction density (20%, 40%, 60%, 80%, 100%) with three lines (Systems A, B, C) showing NDCG@10. The crossover point — where System A overtakes B and/or C — is the key finding. A second panel or subplot for HR@10.

---

## References to cite

- He et al. (2017), "Neural Collaborative Filtering" — the NCF paper, our primary baseline reference
- Koren et al. (2009), "Matrix Factorization Techniques for Recommender Systems" — MF foundations
- Rendle et al. (2009), "BPR: Bayesian Personalized Ranking from Implicit Feedback" — BPR loss
- Harper & Konstan (2015), "The MovieLens Datasets" — dataset citation
