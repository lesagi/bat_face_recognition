#!/usr/bin/env bash
# Run an arbitrary list of experiment configs x N folds for one species, into one
# MLflow experiment.
#
# This is scripts/kfold_matrix.sh with one change: CONFIGS comes from the
# environment instead of being hardcoded. That script's 3x3 matrix is the
# published 360-run sweep and is referenced throughout docs/ -- it stays frozen.
# Phase 3 adds five new arms (degraded / band-restricted / background-only /
# intersection / dilated), each with its own config list, so the list has to be a
# parameter. Everything else here -- resume-skip, OK_WITH_WARNINGS, the
# n_test_ids scrape, the GPU-contention warning -- is carried over verbatim
# because each of those exists to catch a failure that actually happened.
#
# Usage: CONFIGS="exp_a exp_b" phase3_matrix.sh <species> <gpu_id>
# Env: CONFIGS (REQUIRED, whitespace-separated experiment names),
#      MLFLOW_EXP (default phase3-<today>), FOLDS (default "0..19"),
#      EXPLAIN (default 0; set 1 to also emit explanations + PDF per fold),
#      PERM (default 1; set 0 to skip the permutation test),
#      KEEP_CKPT (default none), PERM_N (default 1000),
#      LOG_DIR (default outputs/phase3_<species>_logs), FORCE (default 0).
set +u

SPECIES="${1:?usage: CONFIGS=... phase3_matrix.sh <species> <gpu_id>}"
GPU_ID="${2:-0}"
CONFIGS="${CONFIGS:?CONFIGS is required -- whitespace-separated experiment names}"
EXP="${MLFLOW_EXP:-phase3-$(date +%Y%m%d)}"
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

N_CONFIGS=$(echo "$CONFIGS" | tr ' ' '\n' | grep -c .)
N_FOLDS=$(echo "$FOLDS" | wc -w)
TOTAL=$((N_CONFIGS * N_FOLDS))

# LOG_DIR is overridable so the fold range can be sharded across concurrent
# jobs. One training process tops out around 40 of this box's 112 cores -- giving
# it the whole machine only bought 1.3x -- so several jobs over disjoint FOLDS
# finish sooner than one job doing everything. Each shard needs its own
# directory: they would otherwise interleave appends into one summary.tsv.
# Seed a shard's directory with a copy of an existing summary.tsv and its resume
# logic will skip work already completed elsewhere.
LOG_DIR="${LOG_DIR:-outputs/phase3_${SPECIES}_logs}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"

# Resume support. A long crawl gets interrupted -- by a shared-GPU conflict, a
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
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES $N_CONFIGS config(s) x $N_FOLDS folds = $TOTAL runs -> '$EXP'"

# Warn if the pinned card is already busy -- the workstation is shared, and a
# long crawl on a contended GPU is worse than waiting.
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
    # OOM on this shared GPU looks like. Record it, or the batch reports N/N
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
echo "[$(date +%H:%M:%S)] [gpu$GPU_ID] $SPECIES phase3 COMPLETE: $CLEAN/$TOTAL clean, $WARNED with warnings, $FAILED failed, $SKIPPED resumed-skipped"
if [ "$FAILED" -gt 0 ] || [ "$WARNED" -gt 0 ]; then
  echo "  see $SUMMARY; a warned run may have produced NO test metrics --"
  echo "  aggregate_kfold.py reports per-cell coverage, check it before analysing."
fi
exit 0
