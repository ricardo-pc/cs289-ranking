# CS 289A Final Project — Internal Proposal

## When Does Ranking Help? A Sparsity Analysis of Two-Stage Recommendation

Ricardo Perez Castillo · [Teammate 2] · [Teammate 3] — Spring 2026

---

## The Question

Modern recommender systems are often built in two stages: a fast **retrieval** model
generates a candidate set, then a more expensive **ranking** model re-orders it. The
ranking stage adds complexity and compute cost. Is it always worth it?

We hypothesize that the answer depends on how much data each user has. A ranking model
has more parameters and needs more signal to learn useful representations. For users with
sparse histories — few interactions — the extra complexity becomes a liability. A simpler
retrieval model with lower variance may actually win.

**The finding we are looking for:** a crossover point — a number of interactions per user
below which the ranking stage stops helping and the simpler model takes over.

---

## The Three Systems

| System | Architecture | Complexity |
|--------|-------------|------------|
| **A — MF** | User + item embeddings, dot-product score | Low |
| **B — NCF** | Same embeddings, score via MLP instead of dot product | Medium |
| **C — MF + Ranker** | MF retrieves top-100 candidates; BPR-trained MLP re-ranks to top-10 | High |

All three are trained end-to-end in PyTorch. The only thing that changes across conditions
is the amount of training data available per user (the sparsity sweep).

---

## Dataset

**MovieLens 1M** — 1,000,209 ratings from 6,040 users on 3,706 movies.

Key properties confirmed from EDA:
- All users have ≥ 20 ratings (minimum guaranteed by the dataset)
- Matrix density: ~4.47% — appropriately sparse for a RecSys benchmark
- Item popularity follows a strong power law: top 20% of items cover ~80% of ratings
- Rating distribution is right-skewed (positivity bias): peaks at 3–4 stars
- Timestamps are usable for chronological leave-one-out splitting

We use **ml-1m only**. ml-32m (32M ratings, 200K users) was considered but ruled out:
training NCF across 15 conditions would take 25+ hours of GPU time vs ~1 hour on ml-1m,
which is not feasible within the project timeline.

---

## Data Pipeline Decisions (from EDA)

**Implicit feedback with confidence weighting (WMF — Hu et al. 2008)**

All observed ratings are treated as positive preference (label = 1), but the loss
contribution is scaled by a confidence weight:

$$c_{ui} = 1 + \alpha \cdot r_{ui}$$

where $r_{ui}$ is the star rating (1–5) and $\alpha$ is a tunable hyperparameter.
This preserves the information in star ratings without discarding low-rated interactions
entirely. A 5-star rating contributes ~5× more signal than a 1-star. $\alpha$ is tuned
via Bayesian Optimization alongside the other NCF hyperparameters.

**Train / val / test split — He et al. (2017) leave-one-out protocol**
- **TEST**: last interaction per user (by timestamp)
- **VAL**: second-to-last interaction per user
- **TRAIN**: all remaining interactions

**Sparsity sweep**
- Subsample TRAIN at `d` ∈ {1.0, 0.8, 0.6, 0.4, 0.2} — VAL and TEST are never touched
- At 20% density, ~9.2% of users have fewer than 5 training interactions (cold-start regime)
- No users are dropped at any density level — all 6,040 stay in every condition

**Evaluation protocol**
- For each user: sample 99 items they have NOT rated as negatives (fixed seed=42)
- Rank the test item among the 100 (1 positive + 99 negatives)
- Report NDCG@10 and HR@10 averaged over all 6,040 users
- Negatives are fixed across models and density levels for fair comparison

**Feature engineering: none**
- Input is `(user_id, item_id, rating)` only — IDs remapped to contiguous 0-indexed integers
- User demographics (gender, age, occupation) and movie genres excluded intentionally:
  adding content features would confound the sparsity comparison

**ID remapping**
- UserIDs 1–6,040 → 0–6,039
- MovieIDs: only the 3,706 IDs present in ratings.dat are kept (246 gap IDs from movies.dat
  are skipped). Remapped to 0–3,705.

---

## Methods

### Matrix Factorization (System A)
Learns embedding matrices $U \in \mathbb{R}^{n \times k}$ and $V \in \mathbb{R}^{m \times k}$.
Score for user $u$, item $i$: $\hat{r}_{ui} = u_u^\top v_i$.
Loss: confidence-weighted BCE with online negative sampling.

### Neural Collaborative Filtering (System B)
Same user/item embeddings as MF, but the score is computed by a small MLP instead of
dot product: $\hat{r}_{ui} = \text{MLP}([u_u \| v_i])$.
Loss: same confidence-weighted BCE.
Hyperparameters (embedding dim, MLP hidden sizes, learning rate, L2 regularization, $\alpha$)
tuned via Bayesian Optimization using a Gaussian Process surrogate and Expected Improvement
acquisition function. The best configuration found by BO is used as the NCF baseline for
the sparsity sweep.

### MF + Ranking MLP (System C)
**Stage 1 (retrieval):** MF scores all items for each user; top-100 candidates selected.
**Stage 2 (ranking):** A separate MLP re-ranks the 100 candidates.
The ranker is trained with BPR loss — a pairwise objective that pushes the score of observed
items above unobserved ones: $L_{\text{BPR}} = -\sum \log \sigma(\hat{r}_{ui} - \hat{r}_{uj})$

### Sparsity sweep
All three systems are retrained from scratch at each density level `d`. This gives 15
independent training runs (3 systems × 5 densities). Results are plotted as NDCG@10
vs. density — the crossover point is the key finding.

---

## Analytical Framing (Course Connection)

The bias-variance tradeoff from CS 289A lecture is the analytical lens for interpreting
results. More expressive models (NCF, Ranker) have lower bias but higher variance — they
need more data to reliably estimate their larger parameter space. Under sparsity, variance
dominates and simpler models win. The sparsity sweep makes this concrete and measurable.

| CS 289A topic | How it appears |
|--------------|---------------|
| Gradient descent | MF/NCF trained via SGD on embedding matrices |
| Cross-entropy + sigmoid | Confidence-weighted BCE for MF and NCF |
| Backpropagation | Gradients flow through MLP into embeddings |
| L2 regularization | Applied to all embedding matrices |
| Bias-variance tradeoff | Main lens for interpreting the sparsity curves |
| BPR as logistic regression | $\mathcal{L}_\text{BPR}$ is logistic regression on pairwise score differences |

---

## Timeline

| Date | Milestone |
|------|-----------|
| May 3–4 | NCF training pipeline complete and validated |
| May 4–5 | BO sweep kicked off on SCF to tune NCF hyperparameters |
| May 5–6 | MF baseline + two-stage ranker implemented |
| May 6–8 | Full sparsity sweep running (15 conditions) |
| May 8–9 | BO results back → NCF hyperparameters locked |
| May 9–11 | Results finalized, figures generated |
| May 11–13 | Report drafted |
| May 14 | Final review and submission |

---

## Main Deliverable Figure

X-axis: per-user interaction density (20%, 40%, 60%, 80%, 100%)
Y-axis: NDCG@10 (primary) and HR@10 (secondary panel)
Three lines: MF (A), NCF (B), MF+Ranker (C)

The crossover point — where line A overtakes B and/or C — is the key finding.

---

## References

- He et al. (2017), "Neural Collaborative Filtering" — NeurIPS
- Koren et al. (2009), "Matrix Factorization Techniques for Recommender Systems" — IEEE Computer
- Rendle et al. (2009), "BPR: Bayesian Personalized Ranking from Implicit Feedback" — UAI
- Hu et al. (2008), "Collaborative Filtering for Implicit Feedback Datasets" — ICDM
- Harper & Konstan (2015), "The MovieLens Datasets" — ACM TiiS
