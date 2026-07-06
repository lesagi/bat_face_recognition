#!/usr/bin/env bash
# Multi-seed the full 3x3 matrix (Siamese base + ArcFace/AdaFace tuned ×
# green/original/random) for one species, N seeds each, into one MLflow
# experiment — for mean±std reporting on the small (2-3 identity) test splits
# where single runs are noise-dominated. deterministic=true + a fixed data
# split_seed so ONLY the training seed varies; --no-explanations/--no-permutation
# keep each run fast (metrics are logged regardless).
#
# Usage: multiseed_matrix.sh <species> <gpu_id>   species in {mauritius, rousettus}
# Env: MLFLOW_EXP (default multiseed-full-20260706), SEEDS (default "42 43 44 45 46").
set +u

SPECIES="${1:?usage: multiseed_matrix.sh <species> <gpu_id>}"
GPU_ID="${2:-0}"
EXP="${MLFLOW_EXP:-multiseed-full-20260706}"
SEEDS="${SEEDS:-42 43 44 45 46}"

cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

case "$SPECIES" in
  mauritius|rousettus) : ;;
  *) echo "unknown species: $SPECIES"; exit 2 ;;
esac

CONFIGS="
siamese_${SPECIES}_green_bg_video
siamese_${SPECIES}_original_bg_video
siamese_${SPECIES}_random_bg_video
arcface_${SPECIES}_green_bg_video_tuned
arcface_${SPECIES}_original_bg_video_tuned
arcface_${SPECIES}_random_bg_video_tuned
adaface_${SPECIES}_green_bg_video_tuned
adaface_${SPECIES}_original_bg_video_tuned
adaface_${SPECIES}_random_bg_video_tuned
"

LOG_DIR="outputs/multiseed_${SPECIES}_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
echo -e "experiment\tseed\tstatus\trun_id" > "$SUMMARY"
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES full matrix × seeds ($SEEDS) → '$EXP'"

for e in $CONFIGS; do
  for s in $SEEDS; do
    log="$LOG_DIR/${e}_s${s}.log"
    uv run bat-cli train --experiment "$e" \
      --hydra trainer.deterministic=true --hydra seed="$s" \
      --no-explanations --no-permutation \
      --mlflow-experiment "$EXP" --run-name "${e}-s${s}" \
      --run-tag "batch=$EXP" --run-tag "species=$SPECIES" --run-tag "seed=$s" > "$log" 2>&1
    rc=$?
    rid=$(grep -oE "MLflow run: [0-9a-f]+" "$log" | tail -1 | awk '{print $3}')
    st=OK; [ "$rc" -ne 0 ] && st="FAIL_rc${rc}"
    echo -e "${e}\t${s}\t${st}\t${rid:-NA}" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $e s=$s -> $st"
  done
done

echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES multiseed COMPLETE"
