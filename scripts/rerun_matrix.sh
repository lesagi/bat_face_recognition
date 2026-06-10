#!/usr/bin/env bash
# PR11 driver: re-run the experiment matrix on the fixed code, on GPU.
# Usage: rerun_matrix.sh <gpu_id> <group>   where group in {all,a,b}
#   a = first half, b = second half (for 2-GPU parallelism).
# Each run logs to MLflow experiment "bat-rerun-gpu"; run IDs + status are
# appended to outputs/rerun_logs/summary_<group>.tsv. Safe to run unattended.
set +u

GPU_ID="${1:-0}"
GROUP="${2:-all}"

# The project .venv (torch 2.5.1+cu121) reaches the GPUs directly once the
# command sandbox is disabled; no conda activation needed.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="$GPU_ID"

LOG_DIR="outputs/rerun_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_${GROUP}.tsv"
echo -e "experiment\tstatus\trun_id\tlog" > "$SUMMARY"

ALL_A="
siamese_rousettus_random_bg_video
siamese_rousettus_green_bg_video
siamese_mauritius_random_bg_video
siamese_mauritius_green_bg_video
siamese_mauritius_original_bg_video
arcface_rousettus_random_bg_video_tuned
arcface_rousettus_green_bg_video_tuned
arcface_mauritius_random_bg_video_tuned
"
ALL_B="
arcface_mauritius_green_bg_video_tuned
arcface_mauritius_original_bg_video_tuned
adaface_rousettus_random_bg_video_tuned
adaface_rousettus_green_bg_video_tuned
adaface_mauritius_random_bg_video_tuned
adaface_mauritius_green_bg_video_tuned
adaface_mauritius_original_bg_video_tuned
"

case "$GROUP" in
  a) EXPS="$ALL_A" ;;
  b) EXPS="$ALL_B" ;;
  *) EXPS="$ALL_A $ALL_B" ;;
esac

for e in $EXPS; do
  log="$LOG_DIR/${e}.log"
  echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] START $e"
  bat-cli train --experiment "$e" \
      --permutation-n 100 \
      --mlflow-experiment bat-rerun-gpu > "$log" 2>&1
  rc=$?
  run_id=$(grep -oE "MLflow run: [0-9a-f]+" "$log" | tail -1 | awk '{print $3}')
  if [ "$rc" -eq 0 ]; then
    echo -e "${e}\tOK\t${run_id:-NA}\t${log}" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] DONE  $e (run_id=${run_id:-NA})"
  else
    echo -e "${e}\tFAIL_rc${rc}\t${run_id:-NA}\t${log}" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] FAIL  $e (rc=$rc)"
  fi
done

echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] GROUP $GROUP COMPLETE"
