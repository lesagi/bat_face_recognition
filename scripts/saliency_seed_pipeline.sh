#!/usr/bin/env bash
# Train N folds of one species, analyse each checkpoint's saliency, then DELETE
# the checkpoint before training the next one.
#
# The delete is not tidiness. /home is 100% full with ~18 GB free, and one
# arcface-320 checkpoint is 376 MB (model 94 + EMA 94 + optimizer ~188). Keeping
# all 40 would need ~15 GB and would very likely fill the disk mid-crawl on a
# machine shared with another user. Interleaving caps the peak at one checkpoint
# per job. Total wall-clock is unchanged: the IG analysis dominates either way.
#
# Usage: saliency_seed_pipeline.sh <species> <gpu_id>
# Env: FOLDS (default 0..9), MAX_PER_ID (default 4), BACKGROUNDS (default "green original")
set +u
SPECIES="${1:?usage: saliency_seed_pipeline.sh <species> <gpu_id>}"
GPU="${2:-0}"
FOLDS="${FOLDS:-$(seq 0 9)}"
MAX_PER_ID="${MAX_PER_ID:-4}"
BACKGROUNDS="${BACKGROUNDS:-green original}"
EXP_MLFLOW="saliency-seeds-20260825"

cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
LOG_DIR="outputs/saliency_seeds_${SPECIES}_logs"; mkdir -p "$LOG_DIR"
OUT_DIR="outputs/saliency/folds"; mkdir -p "$OUT_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
[ -f "$SUMMARY" ] || echo -e "species\tbackground\tfold\tstatus\tckpt_deleted\tseconds" > "$SUMMARY"

for BG in $BACKGROUNDS; do
  for F in $FOLDS; do
    OUT="$OUT_DIR/${SPECIES}_${BG}_f${F}.json"
    if [ -s "$OUT" ]; then
      echo "[$(date +%H:%M:%S)] skip ${SPECIES}/${BG}/f${F} (already analysed)"
      continue
    fi
    T0=$(date +%s)
    EXP="arcface_${SPECIES}_${BG}_a320_video_tuned"
    TLOG="$LOG_DIR/${EXP}_f${F}.train.log"

    echo "[$(date +%H:%M:%S)] [gpu$GPU] train ${SPECIES}/${BG}/f${F}"
    uv run bat-cli train --experiment "$EXP" --fold "$F" \
      --no-explanations --no-permutation --keep-checkpoints roc_auc \
      --mlflow-experiment "$EXP_MLFLOW" --run-name "${EXP}-f${F}" \
      --run-tag "batch=$EXP_MLFLOW" --run-tag "species=$SPECIES" > "$TLOG" 2>&1
    rc=$?

    # Resolve the run dir from the leaf the CLI builds: the fold index is the
    # training seed, so it lands in _s<fold>.
    CKPT=$(ls -1dt outputs/runs/*/"${SPECIES}_video_${BG}_arcface_arcface_e320_s${F}"* 2>/dev/null \
           | head -1 | sed 's:$:/best_model_roc_auc.pt:')
    if [ "$rc" -ne 0 ] || [ ! -f "$CKPT" ]; then
      echo -e "${SPECIES}\t${BG}\t${F}\tFAIL_train_rc${rc}\tno\t$(( $(date +%s) - T0 ))" >> "$SUMMARY"
      echo "[$(date +%H:%M:%S)]   !! train failed (rc=$rc), no checkpoint at ${CKPT:-<none>}"
      continue
    fi

    echo "[$(date +%H:%M:%S)] [gpu$GPU] analyse ${SPECIES}/${BG}/f${F}"
    ALOG="$LOG_DIR/${EXP}_f${F}.saliency.log"
    uv run python scripts/species_saliency_maps.py \
      --model arcface --method ig --model-edge 320 --max-per-identity "$MAX_PER_ID" \
      --randomise-control --background "$BG" --species "$SPECIES" \
      --checkpoint "${SPECIES}=${CKPT}" --device cuda:0 \
      --out "$OUT" --fig-dir "$LOG_DIR/figures" > "$ALOG" 2>&1
    arc=$?

    DEL=no
    if [ -f "$CKPT" ]; then rm -f "$CKPT" && DEL=yes; fi
    ST=OK; [ "$arc" -ne 0 ] && ST="FAIL_saliency_rc${arc}"
    [ ! -s "$OUT" ] && ST="FAIL_no_output"
    echo -e "${SPECIES}\t${BG}\t${F}\t${ST}\t${DEL}\t$(( $(date +%s) - T0 ))" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)]   -> $ST ($(( $(date +%s) - T0 ))s, ckpt deleted=$DEL)"
  done
done
echo "[$(date +%H:%M:%S)] [gpu$GPU] $SPECIES pipeline COMPLETE"
awk -F'\t' 'NR>1{c[$4]++} END{for(k in c) printf "  %s: %d\n", k, c[k]}' "$SUMMARY"
