#!/usr/bin/env bash
# Higher-resolution retrain of the embedding models (ArcFace/AdaFace only —
# ResNet50 is size-agnostic; Siamese's 4-conv net is not). Same tuned configs,
# just model.input_edge_length overridden. The 224px aligned crops already
# exist, so at EDGE=224 the loader does an identity resize (no downsizing).
# Multi-seed for mean±std, into one MLflow experiment tagged edge=<EDGE>.
#
# Usage: multiseed_hires.sh <species> <gpu_id>   species in {mauritius, rousettus}
# Env: EDGE (default 224), MLFLOW_EXP (default multiseed-hires<EDGE>-20260706),
#      SEEDS (default "42 43 44 45 46"),
#      EXPLAIN (default 1; set 0 to skip explanations+permutation),
#      KEEP_CKPT (default none; all|roc_auc|f1|none).
set +u

SPECIES="${1:?usage: multiseed_hires.sh <species> <gpu_id>}"
GPU_ID="${2:-0}"
EDGE="${EDGE:-224}"
EXP="${MLFLOW_EXP:-multiseed-hires${EDGE}-20260706}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EXPLAIN="${EXPLAIN:-1}"
KEEP_CKPT="${KEEP_CKPT:-none}"
EXPL_FLAGS=""
[ "$EXPLAIN" = "0" ] && EXPL_FLAGS="--no-explanations --no-permutation"

cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

case "$SPECIES" in
  mauritius|rousettus) : ;;
  *) echo "unknown species: $SPECIES"; exit 2 ;;
esac

CONFIGS="
arcface_${SPECIES}_green_bg_video_tuned
arcface_${SPECIES}_original_bg_video_tuned
arcface_${SPECIES}_random_bg_video_tuned
adaface_${SPECIES}_green_bg_video_tuned
adaface_${SPECIES}_original_bg_video_tuned
adaface_${SPECIES}_random_bg_video_tuned
"

LOG_DIR="outputs/hires${EDGE}_${SPECIES}_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
echo -e "experiment\tseed\tstatus\trun_id" > "$SUMMARY"
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES ArcFace/AdaFace @ ${EDGE}px × seeds ($SEEDS) → '$EXP'"

for e in $CONFIGS; do
  for s in $SEEDS; do
    log="$LOG_DIR/${e}_s${s}.log"
    uv run bat-cli train --experiment "$e" \
      --hydra model.input_edge_length="$EDGE" \
      --hydra trainer.deterministic=true --hydra seed="$s" \
      $EXPL_FLAGS --keep-checkpoints "$KEEP_CKPT" \
      --mlflow-experiment "$EXP" --run-name "${e}-e${EDGE}-s${s}" \
      --run-tag "batch=$EXP" --run-tag "species=$SPECIES" --run-tag "seed=$s" \
      --run-tag "edge=$EDGE" > "$log" 2>&1
    rc=$?
    rid=$(grep -oE "MLflow run: [0-9a-f]+" "$log" | tail -1 | awk '{print $3}')
    st=OK; [ "$rc" -ne 0 ] && st="FAIL_rc${rc}"
    echo -e "${e}\t${s}\t${st}\t${rid:-NA}" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $e s=$s -> $st"
  done
done

echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES hires${EDGE} COMPLETE"
