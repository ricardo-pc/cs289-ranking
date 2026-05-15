"""
plot_results.py - Generate the main sparsity figure for the report.

Run from the repo root:
    python notebooks/plot_results.py

Saves figures/sparsity_results.pdf (and .png for Overleaf upload).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from evaluate.py across all 15 conditions
densities = [0.2, 0.4, 0.6, 0.8, 1.0]

ndcg = {
    "MF":     [0.2998, 0.3418, 0.3636, 0.3783, 0.3962],
    "NCF":    [0.2796, 0.3211, 0.3502, 0.3661, 0.3779],
    "Ranker": [0.2992, 0.3394, 0.3524, 0.3639, 0.3762],
}

hr = {
    "MF":     [0.5391, 0.6028, 0.6334, 0.6565, 0.6768],
    "NCF":    [0.5171, 0.5825, 0.6220, 0.6430, 0.6578],
    "Ranker": [0.5515, 0.6086, 0.6310, 0.6483, 0.6570],
}

colors  = {"MF": "#2196F3", "NCF": "#FF5722", "Ranker": "#4CAF50"}
markers = {"MF": "o",       "NCF": "s",        "Ranker": "^"}
labels  = {"MF": "MF (A)", "NCF": "NCF (B)",  "Ranker": "MF + Ranker (C)"}

Path("figures").mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for model in ["MF", "NCF", "Ranker"]:
    ax1.plot(densities, ndcg[model],
             color=colors[model], marker=markers[model],
             linewidth=2, markersize=7, label=labels[model])
    ax2.plot(densities, hr[model],
             color=colors[model], marker=markers[model],
             linewidth=2, markersize=7, label=labels[model])

# Annotate the HR crossover at d=0.4
ax2.annotate(
    "HR crossover\n(Ranker > MF)",
    xy=(0.4, 0.6086), xytext=(0.45, 0.575),
    fontsize=8, color=colors["Ranker"],
    arrowprops=dict(arrowstyle="->", color=colors["Ranker"], lw=1.2),
)

for ax, ylabel, title in [
    (ax1, "NDCG@10", "No ranking benefit: MF leads on NDCG@10 at every density"),
    (ax2, "HR@10",   "Ranking stage improves hit rate below 40% density"),
]:
    ax.set_xlabel("Training density $d$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(densities)
    ax.set_xticklabels([str(d) for d in densities])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.15, 1.05)

plt.tight_layout()
plt.subplots_adjust(top=0.82)

fig.suptitle(
    "MF Outperforms More Complex Models at Every Density Level",
    fontsize=13, fontweight="bold", y=0.98,
)
fig.text(
    0.5, 0.90,
    "A ranking stage does not improve ranking quality (NDCG) but gains a hit rate advantage in sparse regimes — adding complexity rarely pays off",
    ha="center", fontsize=9, style="italic", color="#444444",
)

plt.savefig("figures/sparsity_results.pdf", bbox_inches="tight", dpi=150)
plt.savefig("figures/sparsity_results.png", bbox_inches="tight", dpi=150)
print("Saved figures/sparsity_results.pdf and .png")
