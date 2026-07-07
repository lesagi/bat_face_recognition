#!/usr/bin/env bash
# Resolution study: retrain the two CNN embedding heads (ArcFace/AdaFace) on the
# freshly-built genuine higher-resolution aligned crops, at input_edge_length =
# {224, 320}, for one species. The 224/320 crops live in dimension-named dirs and
# each has its own manifest + data config (manifest_<species>_<bg>_aligned_<edge>),
# so training at edge=E reads the E-px crops with an identity resize (no info loss).
# Multi-seed for mean±std on the small (2-3 identity) test splits. Recognition is
# already known not to improve with resolution; the payoff here is a finer GradCAM
# grid (112->4x4, 224->7x7, 320->10x10) computed later from these checkpoints.
#
# Usage: multiseed_resolution.sh <species> <gpu_id>   species in {mauritius, rousettus}
# Env: MLFLOW_EXP (default multiseed-res-20260707), SEEDS (default "42 43 44 45 46"),
#      EDGES (default "224 320").
# Note: per-background ArcFace margin/scale were tuned at 112px (partial confound,
# consistent with the prior resolution study).
set +u

SPECIES="${1:?usage: multiseed_resolution.sh <species> <gpu_id>}"
GPU_ID="${2:-0}"
EXP="${MLFLOW_EXP:-multiseed-res-20260707}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EDGES="${EDGES:-224 320}"

cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

case "$SPECIES" in
  mauritius|rousettus) : ;;
  *) echo "unknown species: $SPECIES"; exit 2 ;;
esac

MODELS="${MODELS:-arcface adaface}"
BGS="${BGS:-green original random}"

# Leave exactly one *real* best_model_roc_auc.pt in a run dir. The trainer saves
# several best_model_<metric>.pt and symlinks metrics that peak at the same epoch
# (e.g. arcface's roc_auc → f1), so a naive "rm f1 loss" would delete the file
# the roc_auc symlink points at. Materialize the target first, then drop siblings.
keep_roc_auc() {
  local d=$1
  [ -e "$d/best_model_roc_auc.pt" ] || return 0
  local tgt; tgt=$(readlink -f "$d/best_model_roc_auc.pt")
  cp --remove-destination "$tgt" "$d/roc_auc.keep" 2>/dev/null || return 0
  find "$d" -maxdepth 1 -name "best_model_*.pt" -delete
  mv "$d/roc_auc.keep" "$d/best_model_roc_auc.pt"
}

LOG_DIR="outputs/res_${SPECIES}_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
echo -e "experiment\tedge\tseed\tstatus\trun_id" > "$SUMMARY"
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES ArcFace/AdaFace × {$BGS} × {$EDGES}px × seeds ($SEEDS) → '$EXP'"

for m in $MODELS; do
  for bg in $BGS; do
    exp="${m}_${SPECIES}_${bg}_bg_video_tuned"
    for edge in $EDGES; do
      data="manifest_${SPECIES}_${bg}_aligned_${edge}"
      for s in $SEEDS; do
        log="$LOG_DIR/${m}_${bg}_e${edge}_s${s}.log"
        run_out="outputs/res_runs/${SPECIES}_${m}_${bg}_e${edge}_s${s}"
        uv run bat-cli train --experiment "$exp" \
          --hydra "data=$data" \
          --hydra model.input_edge_length="$edge" \
          --hydra trainer.deterministic=true --hydra seed="$s" \
          --no-explanations --no-permutation \
          --output-dir "$run_out" \
          --mlflow-experiment "$EXP" --run-name "${m}-${SPECIES}-${bg}-e${edge}-s${s}" \
          --run-tag "batch=$EXP" --run-tag "species=$SPECIES" --run-tag "seed=$s" \
          --run-tag "edge=$edge" --run-tag "background=$bg" --run-tag "model=$m" > "$log" 2>&1
        rc=$?
        # Keep only a real roc_auc checkpoint; the other best_model_*.pt files are
        # ~376MB each and unused downstream (disk pressure on the 120-run matrix).
        keep_roc_auc "$run_out"
        rid=$(grep -oE "MLflow run: [0-9a-f]+" "$log" | tail -1 | awk '{print $3}')
        st=OK; [ "$rc" -ne 0 ] && st="FAIL_rc${rc}"
        echo -e "${exp}\t${edge}\t${s}\t${st}\t${rid:-NA}" >> "$SUMMARY"
        echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $exp e=$edge s=$s -> $st"
      done
    done
  done
done

echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES resolution matrix COMPLETE"
