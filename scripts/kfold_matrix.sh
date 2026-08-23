#!/usr/bin/env bash
# 20-fold the full 3x3 matrix (Siamese base + ArcFace/AdaFace tuned x
# green/original/random) for one species, into one MLflow experiment.
#
# Difference from multiseed_matrix.sh, and the whole point of this script: that
# one pins the data split and varies only the training seed, so all 5 runs of a
# cell hold out the SAME bats and the spread measures optimizer noise. Here each
# fold re-partitions the identities in memory (`--fold N`), so the spread
# measures what actually matters -- which individuals landed in test. Fold N also
# varies the number of held-out identities (mauritius 2-6, rousettus 2-5, drawn
# from the fold seed), which makes "how many test identities do you need?" a
# measurable question rather than a fixed assumption.
#
# Explanations are OFF by default: 180 runs per species would emit 180 PDFs and
# saliency galleries nobody opens. Permutation IS kept -- it is cheap
# (inference-mode) and is what reviewer comment 4 asks for per species.
#
# Usage: kfold_matrix.sh <species> <gpu_id>   species in {mauritius, rousettus}
# Env: MLFLOW_EXP (default kfold20-<today>), FOLDS (default "0..19"),
#      EXPLAIN (default 0; set 1 to also emit explanations + PDF per fold),
#      PERM (default 1; set 0 to skip the permutation test),
#      KEEP_CKPT (default none), PERM_N (default 1000).
set +u

SPECIES="${1:?usage: kfold_matrix.sh <species> <gpu_id>}"
GPU_ID="${2:-0}"
EXP="${MLFLOW_EXP:-kfold20-$(date +%Y%m%d)}"
FOLDS="${FOLDS:-$(seq 0 19)}"
EXPLAIN="${EXPLAIN:-0}"
PERM="${PERM:-1}"
KEEP_CKPT="${KEEP_CKPT:-none}"
PERM_N="${PERM_N:-1000}"

FLAGS=""
[ "$EXPLAIN" = "0" ] && FLAGS="$FLAGS --no-explanations"
[ "$PERM" = "0" ] && FLAGS="$FLAGS --no-permutation"

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

N_CONFIGS=$(echo "$CONFIGS" | grep -c .)
N_FOLDS=$(echo "$FOLDS" | wc -w)
TOTAL=$((N_CONFIGS * N_FOLDS))

# LOG_DIR is overridable so the fold range can be sharded across concurrent
# jobs. One training process tops out around 40 of this box's 112 cores — giving
# it the whole machine only bought 1.3x — so several jobs over disjoint FOLDS
# finish sooner than one job doing everything. Each shard needs its own
# directory: they would otherwise interleave appends into one summary.tsv.
# Seed a shard's directory with a copy of an existing summary.tsv and its resume
# logic will skip work already completed elsewhere.
LOG_DIR="${LOG_DIR:-outputs/kfold_${SPECIES}_logs}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"

# Resume support. A 180-run crawl gets interrupted -- by a shared-GPU conflict, a
# restart, or a deliberate re-plan -- and redoing completed folds wastes hours.
# Keep the existing summary and skip any (experiment, fold) already recorded OK.
# FORCE=1 starts clean.
DONE_LIST="$LOG_DIR/.completed"
: > "$DONE_LIST"
if [ "${FORCE:-0}" = "1" ] || [ ! -f "$SUMMARY" ]; then
  echo -e "experiment\tfold\tstatus\trun_id\tn_test_ids\twarnings" > "$SUMMARY"
else
  # Only clean OK counts as done; OK_WITH_WARNINGS may have produced no metrics.
  awk -F'\t' 'NR>1 && $3=="OK" {print $1"\t"$2}' "$SUMMARY" | sort -u > "$DONE_LIST"
  N_DONE=$(wc -l < "$DONE_LIST")
  [ "$N_DONE" -gt 0 ] && echo "[resume] skipping $N_DONE run(s) already OK in $SUMMARY"
fi
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES 3x3 matrix x $N_FOLDS folds = $TOTAL runs -> '$EXP'"

# Warn if the pinned card is already busy -- the workstation is shared, and a
# 180-run crawl on a contended GPU is worse than waiting.
if command -v nvidia-smi >/dev/null 2>&1; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null)
  [ -n "$USED" ] && [ "$USED" -gt 4000 ] && \
    echo "[warn] gpu$GPU_ID already has ${USED} MiB in use -- another job may be running"
fi

DONE=0
SKIPPED=0
for e in $CONFIGS; do
  for f in $FOLDS; do
    if grep -qxF "${e}	${f}" "$DONE_LIST" 2>/dev/null; then
      SKIPPED=$((SKIPPED + 1))
      DONE=$((DONE + 1))
      continue
    fi
    log="$LOG_DIR/${e}_f${f}.log"
    uv run bat-cli train --experiment "$e" \
      --fold "$f" \
      $FLAGS --keep-checkpoints "$KEEP_CKPT" --permutation-n "$PERM_N" \
      --mlflow-experiment "$EXP" --run-name "${e}-f${f}" \
      --run-tag "batch=$EXP" --run-tag "species=$SPECIES" > "$log" 2>&1
    rc=$?
    rid=$(grep -oE "MLflow run: [0-9a-f]+" "$log" | tail -1 | awk '{print $3}')
    # The [split] line the CLI prints reports the realised identity counts.
    nte=$(grep -oE "test=[0-9]+ids" "$log" | tail -1 | grep -oE "[0-9]+")

    # A pipeline step that throws is caught by the CLI's _safe(), appended to the
    # run's warning list, and the process still exits 0. A run can therefore
    # "succeed" having produced no test metrics at all -- which is what an eval
    # OOM on this shared GPU looks like. Record it, or the batch reports 180/180
    # OK while silently yielding fewer usable results.
    warn=$(grep -cE "failed:" "$log")
    st=OK
    [ "$rc" -ne 0 ] && st="FAIL_rc${rc}"
    [ "$st" = "OK" ] && [ "$warn" -gt 0 ] && st="OK_WITH_WARNINGS"
    echo -e "${e}\t${f}\t${st}\t${rid:-NA}\t${nte:-NA}\t${warn}" >> "$SUMMARY"
    DONE=$((DONE + 1))
    echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] ($DONE/$TOTAL) $e fold=$f n_test_ids=${nte:-?} -> $st"
    [ "$warn" -gt 0 ] && grep -E "failed:" "$log" | head -3 | sed 's/^/      /'
  done
done

FAILED=$(awk -F'\t' 'NR>1 && $3 ~ /^FAIL/' "$SUMMARY" | wc -l)
WARNED=$(awk -F'\t' 'NR>1 && $3=="OK_WITH_WARNINGS"' "$SUMMARY" | wc -l)
CLEAN=$((DONE - FAILED - WARNED))
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES kfold COMPLETE: $CLEAN/$TOTAL clean, $WARNED with warnings, $FAILED failed, $SKIPPED resumed-skipped"
if [ "$FAILED" -gt 0 ] || [ "$WARNED" -gt 0 ]; then
  echo "  see $SUMMARY; a warned run may have produced NO test metrics --"
  echo "  aggregate_kfold.py reports per-cell coverage, check it before analysing."
fi
exit 0
