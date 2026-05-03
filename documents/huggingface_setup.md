# HuggingFace Checkpoint Setup

GPU jobs on SCF are **preemptable** — your job can be stopped mid-run. Checkpoints are
pushed to a private HuggingFace repo so you can resume without starting over.
Ricardo has already created the repo and will share the token with you directly.

---

## Your setup (one time only)

**Step 1 — Get the token from Ricardo** via a secure channel (not Slack, not the repo).

**Step 2 — Add it to your shell config.**

On Mac (laptop):
```bash
nano ~/.zshrc
```

On SCF / GCP:
```bash
nano ~/.bashrc
```

Add these two lines at the bottom:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"        # replace with the token Ricardo gave you
export HF_REPO_ID="Berkeley-statistics/cs289-ranking-checkpoints"
```

Save (`Ctrl+O` → Enter) and exit (`Ctrl+X`).

**Step 3 — Reload your shell:**
```bash
source ~/.zshrc    # Mac
source ~/.bashrc   # SCF / GCP
```

**Step 4 — Verify:**
```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami(token='$HF_TOKEN')['name'])"
```
You should see your HuggingFace username. If you get an error, the token is wrong or not set.

---

## How it works during training

`src/train.py` automatically pushes a checkpoint to HF at the end of each epoch
when `HF_TOKEN` is set. Only the 2 most recent checkpoints per run are kept.
If your job gets preempted, just resubmit the same command — training resumes from
the latest checkpoint automatically.

---

## Important

**Never paste the token into the repo, a script, or a chat message.** If it gets
exposed, tell Ricardo immediately so it can be revoked and regenerated on HuggingFace.
