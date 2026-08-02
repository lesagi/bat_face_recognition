#!/usr/bin/env bash
# Shape-vs-texture controls, pinned to one GPU. Two experiments:
#   (1) Silhouette-only matrix  - train Siamese/ArcFace/AdaFace on pure face-shape
#       images (white mask on black) for both species x 5 seeds, to measure the
#       shape channel against green/original. Explanations off (not meaningful on
#       silhouettes); permutation kept (tests above-chance); checkpoints pruned.
#   (2) Occlusion probe         - retrain ArcFace/original (seed 42, keep roc_auc)
#       per species, then evaluate it on texture-removed and shape-removed test
#       sets (scripts/occlusion_probe.py) to see which cue it causally relies on.
#
# Usage: run_controls.sh <gpu_id>
set +u
GPU_ID="${1:-0}"
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
SEEDS="${SEEDS:-42 43 44 45 46}"
SIL_EXP="${SIL_EXP:-controls-silhouette-20260729}"
OCC_EXP="${OCC_EXP:-controls-occlusion-20260801}"
LOG_DIR="outputs/controls_logs"; mkdir -p "$LOG_DIR"
SUM="$LOG_DIR/summary.tsv"; echo -e "phase\texperiment\tspecies\tseed\tstatus\trun_id" > "$SUM"
log(){ echo "[$(date '+%F %H:%M:%S')] [gpu$GPU_ID] $*"; }

log "CONTROLS START"

# ---- (1) silhouette-only matrix -------------------------------------------------
if [ "${SKIP_SILH:-0}" = "1" ]; then
  log "SKIP_SILH=1 -> skipping silhouette matrix"
else
for sp in rousettus mauritius; do
  MAN="data/manifests/${sp}_silhouette_bg_manifest.csv"
  for e in "siamese_${sp}_green_bg_video" "arcface_${sp}_green_bg_video_tuned" "adaface_${sp}_green_bg_video_tuned"; do
    for s in $SEEDS; do
      lg="$LOG_DIR/silh_${e}_s${s}.log"
      uv run bat-cli train --experiment "$e" \
        --hydra data.manifest_path="$MAN" --hydra data.background=silhouette \
        --hydra trainer.deterministic=true --hydra seed="$s" \
        --no-explanations --keep-checkpoints none \
        --mlflow-experiment "$SIL_EXP" --run-name "${e}-silh-s${s}" \
        --run-tag batch="$SIL_EXP" --run-tag species="$sp" --run-tag seed="$s" \
        --run-tag variant=silhouette > "$lg" 2>&1
      rc=$?; rid=$(grep -oE "MLflow run: [0-9a-f]+" "$lg" | tail -1 | awk '{print $3}')
      st=OK; [ "$rc" -ne 0 ] && st="FAIL_rc${rc}"
      echo -e "silhouette\t${e}\t${sp}\t${s}\t${st}\t${rid:-NA}" >> "$SUM"
      log "silhouette $e s=$s -> $st"
    done
  done
done
log "silhouette matrix COMPLETE"
fi

# ---- (2) occlusion probe (multi-seed; 2-identity test is too noisy for one seed) ---
rm -f outputs/controls/occlusion_results.csv   # fresh accumulation across seeds
for sp in rousettus mauritius; do
  for s in $SEEDS; do
    run_out="outputs/controls/occlusion_${sp}_arcface_s${s}"
    lg="$LOG_DIR/occl_retrain_${sp}_s${s}.log"
    uv run bat-cli train --experiment "arcface_${sp}_original_bg_video_tuned" \
      --hydra trainer.deterministic=true --hydra seed="$s" \
      --no-explanations --no-permutation --keep-checkpoints roc_auc \
      --output-dir "$run_out" \
      --mlflow-experiment "$OCC_EXP" --run-name "occl-base-${sp}-s${s}" \
      --run-tag batch="$OCC_EXP" --run-tag species="$sp" --run-tag seed="$s" > "$lg" 2>&1
    rc=$?; rid=$(grep -oE "MLflow run: [0-9a-f]+" "$lg" | tail -1 | awk '{print $3}')
    st=OK; [ "$rc" -ne 0 ] && st="FAIL_rc${rc}"
    echo -e "occlusion_retrain\tarcface_${sp}_original\t${sp}\t${s}\t${st}\t${rid:-NA}" >> "$SUM"
    log "occlusion retrain $sp s=$s -> $st"
    if [ -e "$run_out/best_model_roc_auc.pt" ]; then
      uv run python scripts/occlusion_probe.py --species "$sp" --seed "$s" \
        --checkpoint "$run_out/best_model_roc_auc.pt" >> "$LOG_DIR/occl_probe_${sp}.log" 2>&1
      log "occlusion probe $sp s=$s -> rc=$?"
    else
      log "occlusion probe $sp s=$s SKIPPED (no checkpoint)"
    fi
  done
done

log "CONTROLS COMPLETE"
