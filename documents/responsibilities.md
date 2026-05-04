# Task Checklist

Last updated: 2026-05-03

For full context on each task see [`proposal.md`](proposal.md).
Check off tasks as they are merged into `main`.

---

## Infrastructure
- [x] Repo structure, `.gitignore`, `environment.yml` / `environment-gpu.yml`
- [x] `download_data.sh`
- [x] SLURM job script (`jobs/train_scf.sh`)
- [x] HuggingFace checkpoint repo configured
- [x] SCF environment set up (conda, PyTorch cu121 wheel, git SSH auth)

## Data & EDA
- [x] EDA notebook (`notebooks/01_eda.ipynb`)
- [x] `src/data.py` — load, remap IDs, leave-one-out split, confidence weights, negative sampling
- [x] `src/data.py` self-test passes all checks

## Models
- [ ] `src/models.py` — MF (embedding → dot product) ← teammate
- [x] `src/models.py` — NCF (embedding → MLP + confidence-weighted BCE loss)
- [ ] `src/models.py` — two-stage ranker (MF retrieval + BPR ranking MLP)

## Training & Evaluation
- [x] `src/train.py` — training loop with confidence-weighted BCE, checkpoint saving
- [x] `src/utils.py` — shared `evaluate()` function (NDCG@10, HR@10) ← to be created
- [ ] `src/evaluate.py` — standalone test-set evaluator (imports from utils.py) ← in progress
- [x] Validate NCF trains end-to-end — smoke test passed (density=0.2, 1 epoch, CPU + GPU)
- [x] NCF baseline run complete (density=1.0, 20 epochs, GPU) — val NDCG@10=0.3944, HR@10=0.6834
- [ ] Validate MF trains end-to-end (`python src/train.py --model mf --density 1.0`) ← after teammate adds MF
- [ ] Validate ranker trains end-to-end (`python src/train.py --model ranker --density 1.0`)

## Hyperparameter Tuning
- [ ] Kick off Bayesian Optimization sweep on SCF (STAT 238 repo) — 25-30 trials, tunes emb_dim, mlp_layers, lr, l2, alpha
- [ ] Lock in best NCF configuration θ* from BO results

## Sparsity Sweep
- [ ] `src/sparsity.py` — orchestrates all 15 runs (3 models × 5 densities)
- [ ] Run full sweep and collect results
- [ ] Validate results (NDCG@10 per model per density level)

## Results & Report
- [ ] `notebooks/02_results_analysis.ipynb` — main figure (NDCG@10 vs density, 3 curves)
- [ ] HR@10 subplot
- [ ] Report: Problem Definition + Related Work + Course Connections
- [ ] Report: Approach + Contribution (sparsity experiment)
- [ ] Report: Results + Discussion
- [ ] Final edit and unification pass
- [ ] Submission check (format, page count, references, code zip)

## Working Conventions (all teammates)
- [ ] Set up environment and verify with `python src/data.py data/raw/ml-1m`
- [ ] Configure HuggingFace before running any GPU job (see `documents/huggingface_setup.md`)
- [ ] Add a `project_logs/YYYY-MM-DD_<name>.md` entry after each work session
- [ ] Open PRs for all code — do not push directly to `main`
