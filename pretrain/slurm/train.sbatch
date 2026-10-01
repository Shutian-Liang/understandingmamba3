#!/usr/bin/env bash
#SBATCH --cpus-per-task=20
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch pretrain/slurm/train.sbatch CONFIG.json [training overrides]" >&2
    exit 2
fi
: "${SLURM_NTASKS:?Missing Slurm task allocation}"
: "${EXPECTED_GPUS:?Submit with pretrain/slurm/submit.sh}"
if (( SLURM_NTASKS != EXPECTED_GPUS )); then
    echo "Expected $EXPECTED_GPUS tasks, got $SLURM_NTASKS" >&2
    exit 2
fi

config="$1"
shift
project="${MAMBA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MAMBA_PYTHON="${MAMBA_PYTHON:-python}"
python_prefix="$("$MAMBA_PYTHON" -c 'import sys; print(sys.prefix)')"
export PATH="$python_prefix/bin:$PATH"
if [[ -z "${CUDA_HOME:-}" && -x "$python_prefix/bin/nvcc" ]]; then
    export CUDA_HOME="$python_prefix"
fi
export PYTHONPATH="$project${MAMBA_EXTRA_PYTHONPATH:+:$MAMBA_EXTRA_PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$project/.cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$project/.cache/tilelang}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

mkdir -p "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR"
cd "$project"
echo "job=$SLURM_JOB_ID config=$config nodes=$SLURM_JOB_NUM_NODES world=$SLURM_NTASKS"

srun \
    --ntasks="$SLURM_NTASKS" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --kill-on-bad-exit=1 \
    "$MAMBA_PYTHON" -m pretrain.train --config "$config" "$@"
