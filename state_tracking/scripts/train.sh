#!/usr/bin/env bash
# Start one recipe/seed/learning-rate run in the background.
set -euo pipefail
STATE_TRACKING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FOREGROUND=0
for argument in "$@"; do
    case "$argument" in
        -h|--help)
            printf '%s\n' 'Single-run defaults: --task parity --model no_rotation --seed 42 --lr 0.001.' \
                'Validation runs during training; the best checkpoint is tested at the end.' \
                'Pass options below to override them; use sweep.sh for configured sweeps.'
            FOREGROUND=1
            ;;
        --list|--dry-run)
            FOREGROUND=1
            ;;
    esac
done
cd -- "$STATE_TRACKING_ROOT"

COMMAND=(
    "${PYTHON:-python}" "$STATE_TRACKING_ROOT/run.py"
    --task parity --model no_rotation --seed 42 --lr 0.001 "$@"
)
if (( FOREGROUND )); then
    exec env PYTHONUNBUFFERED=1 "${COMMAND[@]}"
fi

LOG_DIR="${LOG_DIR:-$STATE_TRACKING_ROOT/logs}"
[[ "$LOG_DIR" == /* ]] || LOG_DIR="$STATE_TRACKING_ROOT/$LOG_DIR"
mkdir -p -- "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_$$.log}"
[[ "$LOG_FILE" == /* ]] || LOG_FILE="$STATE_TRACKING_ROOT/$LOG_FILE"
mkdir -p -- "$(dirname -- "$LOG_FILE")"

nohup setsid env PYTHONUNBUFFERED=1 "${COMMAND[@]}" \
    >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf 'Started state-tracking training (PID %s).\nLog: %s\nMonitor: tail -f %q\n' \
    "$PID" "$LOG_FILE" "$LOG_FILE"
