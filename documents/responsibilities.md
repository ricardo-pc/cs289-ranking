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

## Data & EDA
- [x] EDA notebook (`notebooks/01_eda.ipynb`)
- [x] `src/data.py` — load, remap IDs, leave-one-out split, confidence weights, negative sampling
- [ ] Run `src/data.py` self-test and confirm all checks pass

## Models
- [ ] `src/models.py` — MF (embedding → dot product)
- [ ] `src/models.py` — NCF (embedding → MLP)
- [ ] `src/models.py` — two-stage ranker (MF retrieval + BPR ranking MLP)

## Training & Evaluation
- [ ] `src/train.py` — training loop with confidence-weighted BCE, checkpoint saving
- [ ] `src/evaluate.py` — NDCG@10 and HR@10
- [ ] Validate MF trains end-to-end (`python src/train.py --model mf --density 1.0`)
- [ ] Validate NCF trains end-to-end (`python src/train.py --model ncf --density 1.0`)
- [ ] Validate ranker trains end-to-end (`python src/train.py --model ranker --density 1.0`)

## Hyperparameter Tuning
- [ ] Kick off Bayesian Optimization sweep on SCF to tune NCF hyperparameters
- [ ] Lock in best NCF configuration from BO results

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
