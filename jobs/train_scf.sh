#!/bin/bash
# ─── Berkeley SCF SLURM job — one (model, density) training run ──────────────
#
# Usage (submit from repo root):
#   sbatch jobs/train_scf.sh --model mf    --density 1.0
#   sbatch jobs/train_scf.sh --model ncf   --density 0.6
#   sbatch jobs/train_scf.sh --model ranker --density 0.4
#
# To launch the full sparsity sweep (15 jobs) at once:
#   for model in mf ncf ranker; do
#     for density in 1.0 0.8 0.6 0.4 0.2; do
#       sbatch jobs/train_scf.sh --model $model --density $density
#     done
#   done
#
# Check job status:  squeue -u $USER
# Cancel all yours:  scancel -u $USER
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cs289-train
#SBATCH --partition=jsteinhardt        # recommended: rainbowquartz / smokyquartz / sunstone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # data loading workers
#SBATCH --gres=gpu:1                  # one GPU per run
#SBATCH --mem=16G
#SBATCH --time=00:30:00               # ml-1m NCF < 10 min; 30 min is safe
#SBATCH --output=jobs/logs/%x_%j.out  # stdout: jobname_jobid.out
#SBATCH --error=jobs/logs/%x_%j.err   # stderr: jobname_jobid.err

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate cs289-ranking-gpu

# Double-check GPU is visible
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA     : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "Python   : $(python --version)"
echo "PyTorch  : $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
echo "Args     : $@"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
# Move to repo root (assumes job was submitted from there)
cd $SLURM_SUBMIT_DIR

python src/train.py "$@" --device cuda --num-workers 4
