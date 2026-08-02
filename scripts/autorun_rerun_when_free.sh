#!/usr/bin/env bash
# Wait until an A5000 is genuinely free, then auto-launch the full July re-run
# (matrix -> hires -> resolution, both species) pinned to that card. Runs the
# whole batch sequentially on the freed card. Meant to be launched DETACHED
# (setsid + nohup) so it survives the session:
#
#   setsid nohup bash scripts/autorun_rerun_when_free.sh >outputs/autorun_nohup.log 2>&1 </dev/null &
#
# Tunables (env): FREE_UTIL (%, default 20), FREE_MEM (MiB free, default 16000),
#   STABLE (consecutive passing checks, default 3), INTERVAL (seconds, default 180).
set -u
cd "$(dirname "$0")/.." || exit 1

FREE_UTIL="${FREE_UTIL:-20}"
FREE_MEM="${FREE_MEM:-16000}"
STABLE="${STABLE:-3}"
INTERVAL="${INTERVAL:-180}"

# TARGET (optional): a script to run as `bash $TARGET <gpu>` once a card frees.
# Default (unset) runs the built-in July re-run chain below. LABEL names the logs.
TARGET="${TARGET:-}"
LABEL="${LABEL:-rerun}"
WATCH_LOG="outputs/autorun_watch_${LABEL}.log"
BATCH_LOG="outputs/${LABEL}.log"
: > "$WATCH_LOG"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$WATCH_LOG"; }

log "WATCHER started; waiting for a free A5000 (util<=${FREE_UTIL}%, free>=${FREE_MEM}MiB, x${STABLE} checks @${INTERVAL}s). GPUs currently held by other users."

pick_free_gpu() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.total,memory.used \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx util total used; do
      idx="${idx// /}"; util="${util// /}"; total="${total// /}"; used="${used// /}"
      free=$(( total - used ))
      if [ "${util:-100}" -le "$FREE_UTIL" ] && [ "$free" -ge "$FREE_MEM" ]; then
        echo "$idx"
      fi
    done | head -1
}

streak=0; gpu=""
while :; do
  cand="$(pick_free_gpu)"
  if [ -n "$cand" ]; then
    if [ "$cand" = "$gpu" ]; then streak=$((streak+1)); else gpu="$cand"; streak=1; fi
    log "GPU $gpu looks free (streak ${streak}/${STABLE})"
    [ "$streak" -ge "$STABLE" ] && break
  else
    [ -n "$gpu" ] && log "GPU $gpu no longer free; resetting"
    streak=0; gpu=""
  fi
  sleep "$INTERVAL"
done

log "LAUNCHING ${LABEL} on GPU ${gpu} -> ${BATCH_LOG}"
{
  echo "=== ${LABEL} started $(date '+%F %H:%M:%S') on GPU ${gpu} ==="
  if [ -n "$TARGET" ]; then
    bash "$TARGET" "$gpu"
  else
    scripts/multiseed_matrix.sh     rousettus "$gpu"
    scripts/multiseed_matrix.sh     mauritius "$gpu"
    scripts/multiseed_hires.sh      rousettus "$gpu"
    scripts/multiseed_hires.sh      mauritius "$gpu"
    scripts/multiseed_resolution.sh rousettus "$gpu"
    scripts/multiseed_resolution.sh mauritius "$gpu"
  fi
  echo "=== ${LABEL} finished $(date '+%F %H:%M:%S') ==="
} >> "$BATCH_LOG" 2>&1
log "${LABEL} COMPLETE (see ${BATCH_LOG})"
