#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash pretrain/slurm/submit.sh CONFIG.json|all [training overrides]" >&2
    exit 2
fi

config="$1"
shift
if [[ "$config" == all ]]; then
    while IFS= read -r config; do
        [[ -z "$config" || "$config" == \#* ]] && continue
        bash pretrain/slurm/submit.sh "$config" "$@"
    done < pretrain/configs/experiments.txt
    exit 0
fi

if [[ ! -f "$config" && -f "pretrain/configs/$config" ]]; then
    config="pretrain/configs/$config"
fi
if [[ ! -f "$config" ]]; then
    echo "Config not found: $config" >&2
    exit 2
fi

name="$(basename "$config")"
case "$name" in
    mamba3_mimo_187m_*) nodes=4; tasks=16; gpus_per_node=4; time=72:00:00 ;;
    mamba3_mimo_444m_*) nodes=8; tasks=32; gpus_per_node=4; time=72:00:00 ;;
    *) echo "Unknown resource profile: $name" >&2; exit 2 ;;
esac

sbatch \
    --job-name="${name%.json}" \
    --nodes="$nodes" \
    --ntasks="$tasks" \
    --ntasks-per-node="$gpus_per_node" \
    --gpus-per-node="$gpus_per_node" \
    --time="$time" \
    --export="ALL,EXPECTED_GPUS=$tasks" \
    pretrain/slurm/train.sbatch "$config" "$@"
