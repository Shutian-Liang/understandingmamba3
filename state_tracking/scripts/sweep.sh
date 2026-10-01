#!/usr/bin/env bash
# Start configured sweeps in the background.
set -euo pipefail
STATE_TRACKING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FOREGROUND=0
for argument in "$@"; do
    case "$argument" in
        -h|--help)
            printf '%s\n' 'Defaults: all 18 recipes and their configured sweeps (275 runs).' \
                'Use --task/--model to limit scope and --dry-run to preview commands.'
            FOREGROUND=1
            ;;
        --list|--dry-run)
            FOREGROUND=1
            ;;
    esac
done
cd -- "$STATE_TRACKING_ROOT"

COMMAND=("${PYTHON:-python}" "$STATE_TRACKING_ROOT/run.py" "$@")
if (( FOREGROUND )); then
    exec env PYTHONUNBUFFERED=1 "${COMMAND[@]}"
fi

LOG_DIR="${LOG_DIR:-$STATE_TRACKING_ROOT/logs}"
[[ "$LOG_DIR" == /* ]] || LOG_DIR="$STATE_TRACKING_ROOT/$LOG_DIR"
mkdir -p -- "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S)_$$.log}"
[[ "$LOG_FILE" == /* ]] || LOG_FILE="$STATE_TRACKING_ROOT/$LOG_FILE"
mkdir -p -- "$(dirname -- "$LOG_FILE")"

nohup setsid env PYTHONUNBUFFERED=1 "${COMMAND[@]}" \
    >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf 'Started state-tracking sweep (PID %s).\nLog: %s\nMonitor: tail -f %q\n' \
    "$PID" "$LOG_FILE" "$LOG_FILE"
