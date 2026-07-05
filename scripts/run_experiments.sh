#!/usr/bin/env bash
# Run the FULL bat-cli train flow for an experiment matrix (3 model families ×
# 3 backgrounds = 9 configs): each config → train → test eval → curves →
# saliency + t-SNE/UMAP → unified 8-section PDF → MLflow logging → permutation
# test. Cross-run summary is produced separately by scripts/extract_results.py.
#
# Usage: run_experiments.sh <species> <gpu_id> <group>
#   species in {mauritius, rousettus};  group in {all,a,b} (a=GPU-A half, b=GPU-B half)
# Env overrides: MLFLOW_EXP (default "<species>-<UTC-date>"), PERM_N (default 100).
# Every run is tagged batch=$MLFLOW_EXP + species=<species> (filter the MLflow UI
# by tags.batch to find the latest set). Per-experiment logs + a summary land in
# outputs/<species>_logs/. Safe to run unattended in the background.
set +u

SPECIES="${1:?usage: run_experiments.sh <species> <gpu_id> <group>}"
GPU_ID="${2:-0}"
GROUP="${3:-all}"
PERM_N="${PERM_N:-100}"

cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
MLFLOW_EXP="${MLFLOW_EXP:-${SPECIES}-$(date -u +%Y%m%d)}"

# 3 model families × 3 backgrounds. Siamese uses base configs; ArcFace/AdaFace
# use the _tuned configs (matching the established setup).
case "$SPECIES" in
  mauritius|rousettus) : ;;
  *) echo "unknown species: $SPECIES (expected mauritius|rousettus)"; exit 2 ;;
esac
GROUP_A="
siamese_${SPECIES}_random_bg_video
siamese_${SPECIES}_green_bg_video
siamese_${SPECIES}_original_bg_video
arcface_${SPECIES}_random_bg_video_tuned
arcface_${SPECIES}_green_bg_video_tuned
"
GROUP_B="
arcface_${SPECIES}_original_bg_video_tuned
adaface_${SPECIES}_random_bg_video_tuned
adaface_${SPECIES}_green_bg_video_tuned
adaface_${SPECIES}_original_bg_video_tuned
"
case "$GROUP" in
  a) EXPS="$GROUP_A" ;;
  b) EXPS="$GROUP_B" ;;
  *) EXPS="$GROUP_A $GROUP_B" ;;
esac

LOG_DIR="outputs/${SPECIES}_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_${GROUP}.tsv"
echo -e "experiment\tstatus\trun_id\tlog" > "$SUMMARY"
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES group $GROUP → MLflow experiment '$MLFLOW_EXP'"

for e in $EXPS; do
  log="$LOG_DIR/${e}.log"
  echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] START $e"
  uv run bat-cli train --experiment "$e" \
      --permutation-n "$PERM_N" \
      --mlflow-experiment "$MLFLOW_EXP" \
      --run-name "$e" \
      --run-tag "batch=$MLFLOW_EXP" \
      --run-tag "species=$SPECIES" > "$log" 2>&1
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

echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES GROUP $GROUP COMPLETE"
