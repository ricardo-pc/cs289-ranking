# Responsibilities & Task Checklist

Last updated: 2026-05-03

For full context on each task, see [`proposal.md`](proposal.md).
Check off tasks as they are merged into `main`.

---

## Ricardo (project lead)

### Infrastructure (done)
- [x] Repo structure, `.gitignore`, `environment.yml` / `environment-gpu.yml`
- [x] `download_data.sh`
- [x] SLURM job script (`jobs/train_scf.sh`)
- [x] EDA notebook (`notebooks/01_eda.ipynb`)
- [x] `src/data.py` — data loading, ID remapping, leave-one-out split, confidence weights, negative sampling

### Models & training (in progress)
- [ ] `src/models.py` — NCF architecture (embedding → MLP → score)
- [ ] `src/train.py` — training loop with confidence-weighted BCE loss, checkpoint saving
- [ ] `src/evaluate.py` — NDCG@10 and HR@10
- [ ] Validate NCF trains end-to-end on ml-1m (`python src/train.py --model ncf --density 1.0`)
- [ ] Kick off BO sweep on SCF for STAT 238 (depends on NCF working)
- [ ] Lock in NCF hyperparameters from BO results

### Sparsity sweep
- [ ] Run NCF sweep: all 5 density levels on SCF
- [ ] Collect and validate results (NDCG@10 per density level)

### Report
- [ ] Sections 4–5: Approach and Contribution (sparsity experiment)
- [ ] Final edit and unification pass
- [ ] Submission check (format, page count, references, code zip)

### HuggingFace
- [ ] Create HF repo for checkpoint storage
- [ ] Add team members as collaborators
- [ ] Configure token in SCF environment (see `documents/huggingface_setup.md`)

---

## Teammate 2

### Data pipeline (in progress)
- [ ] Review and test `src/data.py` — run the self-test, confirm all checks pass
- [ ] `src/models.py` — MF architecture (embedding → dot product → score)
- [ ] Validate MF trains end-to-end (`python src/train.py --model mf --density 1.0`)

### Sparsity sweep
- [ ] Run MF sweep: all 5 density levels (local or GCP)
- [ ] Collect and validate results

### Report
- [ ] Sections 1–3: Problem Definition, Related Work, Course Connections
- [ ] Dataset and evaluation protocol description

---

## Teammate 3

### Models
- [ ] `src/models.py` — two-stage ranker (MF retrieval → BPR-trained ranking MLP)
- [ ] Validate ranker trains end-to-end (`python src/train.py --model ranker --density 1.0`)

### Sparsity sweep
- [ ] Run ranker sweep: all 5 density levels
- [ ] Collect and validate results

### Results & figures
- [ ] `src/sparsity.py` — orchestrates all 15 runs, saves results to `experiments/`
- [ ] `notebooks/02_results_analysis.ipynb` — main figure (NDCG@10 vs density, 3 curves)
- [ ] HR@10 subplot
- [ ] Section 6: Results and Discussion

---

## All teammates

- [ ] Set up environment and verify with `python src/data.py data/raw/ml-1m`
- [ ] Configure HuggingFace checkpointing before running GPU jobs
- [ ] Add a `project_logs/YYYY-MM-DD_<name>.md` entry after each work session
- [ ] Open PRs for all code changes — do not push directly to `main`
- [ ] Final report review before submission (May 14)

---

## Key Dates

| Date | Milestone | Owner |
|------|-----------|-------|
| May 3–4 | NCF training validated | Ricardo |
| May 4–5 | MF training validated | Teammate 2 |
| May 5–6 | Ranker training validated | Teammate 3 |
| May 6–8 | Full 15-run sweep complete | All |
| May 9 | NCF hyperparameters from BO locked | Ricardo |
| May 9–11 | Results finalized, figures done | Teammate 3 |
| May 11–13 | Report drafted | All |
| May 14 | Final review | Ricardo |
| **May 15** | **Submission** | **Ricardo** |
