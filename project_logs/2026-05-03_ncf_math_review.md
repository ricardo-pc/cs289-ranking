# NCF — Conceptual & Mathematical Review

Reference log before building `src/models.py`.
Connects to CS 289A course content for finals prep.

---

## The Problem

Given user $u$ and movie $i$, predict the probability that $u$ interacts with $i$.
This is binary classification on (user, item) pairs — logistic regression with learned features.

---

## Step 1 — Embeddings

Two learned lookup tables:

$$P \in \mathbb{R}^{n_{users} \times k}, \quad Q \in \mathbb{R}^{n_{items} \times k}$$

- $p_u \in \mathbb{R}^k$ — dense representation for user $u$
- $q_i \in \mathbb{R}^k$ — dense representation for item $i$
- $k$ = embedding dimension (hyperparameter, e.g. 64)

Start random, updated by gradient descent during training.

**CS 289A:** $P$ and $Q$ are parameter matrices — same concept as $W$ in linear models.

---

## Step 2 — How MF scores (baseline)

$$\hat{y}_{ui}^{MF} = p_u^\top q_i$$

Linear interaction — assumes user-item compatibility is fully captured by vector alignment.
Same family as ridge regression — linear model on embedding features.

---

## Step 3 — NCF: replacing dot product with MLP

$$\hat{y}_{ui}^{NCF} = \sigma\left(\text{MLP}\left([p_u \| q_i]\right)\right)$$

- $\|$ means **concatenation** (not KL divergence)
- $[p_u \| q_i] \in \mathbb{R}^{2k}$ — embeddings stacked end to end
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ — sigmoid, squashes output to $(0,1)$ so it's a valid probability

### MLP layer by layer

With embedding dim $k=64$ and hidden layers $[128, 64, 32]$:

$$z^{(0)} = [p_u \| q_i] \in \mathbb{R}^{128} \quad \leftarrow \text{input, no computation}$$
$$z^{(1)} = \text{ReLU}(W^{(1)} z^{(0)} + b^{(1)}), \quad W^{(1)} \in \mathbb{R}^{128 \times 128}$$
$$z^{(2)} = \text{ReLU}(W^{(2)} z^{(1)} + b^{(2)}), \quad W^{(2)} \in \mathbb{R}^{64 \times 128}$$
$$z^{(3)} = \text{ReLU}(W^{(3)} z^{(2)} + b^{(3)}), \quad W^{(3)} \in \mathbb{R}^{32 \times 64}$$
$$\hat{y}_{ui} = \sigma(W^{out} z^{(3)} + b^{out}), \quad W^{out} \in \mathbb{R}^{1 \times 32}$$

**Notation:**
- $l$ — layer index (0 = input, L = last hidden)
- $z^{(l)}$ — activation vector at layer $l$ (output of that layer)
- $W^{(l)}$ — weight matrix at layer $l$, shape: (current layer size) × (previous layer size)
- $b^{(l)}$ — bias vector at layer $l$
- $Wz + b$ — same affine transformation as in linear models from lecture

**Why ReLU?** $\text{ReLU}(x) = \max(0, x)$
- Without any non-linearity, stacking multiple $Wz + b$ collapses to one linear transformation — defeats the purpose of using MLP over dot product
- ReLU avoids vanishing gradients (gradient is 0 or 1, not squashed like sigmoid/tanh)
- Sigmoid only at the output layer — the one place we need (0,1) range

**CS 289A:** Backprop through each layer via chain rule — HW3 P3.

---

## Step 4 — Loss function

### Where BCE comes from

Model outputs $\hat{y}_{ui} \in (0,1)$, true label $y \in \{0,1\}$.
Probability of one observation under Bernoulli:

$$P(y|\hat{y}_{ui}) = \hat{y}_{ui}^y (1 - \hat{y}_{ui})^{1-y}$$

Maximize log-likelihood over all observations:

$$\log \mathcal{L} = \sum_i \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

Flip sign to minimize (gradient descent):

$$\mathcal{L}_{BCE} = -\sum_i \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**CS 289A:** Identical to logistic regression loss from HW3 P1.

### WMF confidence weighting

WMF = Weighted Matrix Factorization (Hu et al. 2008).
All observed ratings are positive preference, but weighted by confidence:

$$c_{ui} = 1 + \alpha \cdot r_{ui}$$

- $r_{ui}$ — raw star rating (1–5) from `rating` column in train
- $\alpha$ — scaling hyperparameter tuned by Bayesian Optimization
- Example with $\alpha=1$: 1-star → $c=2$, 3-star → $c=4$, 5-star → $c=6$

### Full batch loss (why two summations)

For **positives** ($y=1$, $(1-y)$ term vanishes):
$$\mathcal{L}_{pos} = -c_{ui} \log \hat{y}_{ui}$$

For **negatives** ($y=0$, $c=1$, $y$ term vanishes):
$$\mathcal{L}_{neg} = -\log(1 - \hat{y}_{uj})$$

Full loss:
$$\mathcal{L} = \underbrace{\sum_{(u,i) \in \text{pos}} -c_{ui} \log \hat{y}_{ui}}_{\text{push positives toward 1}} + \underbrace{\sum_{(u,j) \in \text{neg}} -\log(1 - \hat{y}_{uj})}_{\text{push negatives toward 0}}$$

Plus L2 regularization on embeddings (= ridge regression from lecture):
$$\mathcal{L}_{total} = \mathcal{L}_{WMF} + \lambda(\|P\|_F^2 + \|Q\|_F^2)$$

---

## Step 5 — How BO connects

BO is the **outer loop**. NCF training is the **inner loop**.

```
BO picks hyperparameter config θ = (k, mlp_layers, η, λ, α)
    → train NCF from scratch with θ  (~4 min on GPU)
    → evaluate val NDCG@10
    → report score back to BO
BO fits GP surrogate over all (θ, NDCG@10) pairs seen so far
BO picks next θ via Expected Improvement
... repeat 25-30 times ...
→ θ* = best config found
→ retrain NCF at θ* for all 5 density levels
```

BO optimizes: $\theta^* = \arg\max_\theta \text{NDCG@10}_{val}(θ)$

**CS 289A / STAT 238:** GP surrogate = GP regression from Lecture 23.
EI acquisition = principled explore/exploit tradeoff.

---

## Bias-Variance Connection (key finding)

| Model | Parameters | Bias | Variance |
|-------|-----------|------|----------|
| MF | $n_u k + n_i k$ | Higher (linear only) | Lower |
| NCF | Same + MLP weights | Lower (non-linear) | Higher |

NCF needs more data per user to estimate its extra parameters reliably.
Under sparsity, variance dominates → MF wins at low density.
The crossover point is the key finding of the project.

**CS 289A:** Bias-variance tradeoff from lecture — made measurable on the x-axis of the main figure.
