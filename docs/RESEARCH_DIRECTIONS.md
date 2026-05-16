# Research directions

Post-`v2.0.1-pytorch` priorities for moving the **research** forward.
This is a university research project; the goal is publishable results
on open-set bat face recognition, not production deployment. Items
that don't directly produce analyzable results (serving API, ONNX
inference deployment, WandB, SLURM templates) are not on this list —
see `docs/PHASE_4_CHECKLIST.md` "Out-of-scope" section if curious.

## State at the start of this list (2026-05-16)

- Identity-disjoint 3-way splits on rousettus (12 ids → 6/3/3) and
  mauritius (11 ids → ~5/3/3) manifests.
- Three model families wired and verified end-to-end on rousettus:
  Siamese (pair), ArcFace (embedding), AdaFace (embedding). Tuned HPs
  for the embedding pair are recorded in `docs/PHASE_4_CHECKLIST.md`
  Step 3.
- Reproducibility: bit-exact across runs with
  `trainer.deterministic=true` (v2.0.1, #34).
- Stats: permutation-test infrastructure
  (`bat-cli permutation-test`), 8-section unified PDF, MLflow
  comparison report.

## The central research result so far

**Identity-disjoint generalization stalls at ≤ ~12 train identities**
regardless of model family. On the rousettus manifest the best
`test/roc_auc` is from **Siamese (0.80)**, with ArcFace (0.54) and
AdaFace (0.51) at near-chance. On mauritius (11 ids) ArcFace lands at
0.53 — same wall. This is a meaningful negative result on its own and
is the natural lead-in to whichever direction you push next.

---

## Priority A — cheap wins on the existing manifests

These take hours, not days, and use infrastructure that's already
landed. They produce data points you can put in a thesis chapter.

### A1. Re-enable augmentation

Currently every experiment runs with `data.augmented=false`. The
legacy TF pipeline (`refactor-foundation-pytorch:app/data_augmentation/`)
used per-image HorizontalFlip + RandomBrightnessContrast + RandomGamma
+ RGBShift + VerticalFlip + AdvancedBlur, generating ~60 augmented
copies per source image. Augmentation is a textbook small-data remedy
and the most likely single-knob improvement for Siamese / arc-margin
on a 12-identity dataset.

Concrete next step:
- Run the augmentation pipeline against `data/processed/rousettus/`
  to produce an augmented variant (the TF augmenter code is preserved
  at `git show refactor-foundation-pytorch:app/data_augmentation/`).
- Build a parallel manifest (`rousettus_augmented_manifest.csv`).
- Repeat ArcFace v2 + Siamese on it; compare `test/roc_auc` vs the
  no-aug baselines from Phase 4. Use `trainer.deterministic=true` so
  the comparison is clean.

Expected outcome: meaningful lift on Siamese (which is already the
best on this dataset); modest lift on arc-margin (limited by identity
count regardless).

### A2. Combined-species manifest

12 rousettus + 11 mauritius = 23 identities. That's still small in
absolute terms but ~2× the current per-species count, which is
exactly the regime where arc-margin starts being able to use its
angular-separation objective.

Concrete next step:
- Build a single manifest containing both species' images. The
  manifest builder doesn't combine across directories today; either
  call `build_manifest` twice and concatenate, or extend
  `bat-cli build-manifest` to accept multiple `--input-dir` flags.
- Decide explicitly whether `species` becomes a label feature or just
  metadata. Cleanest: keep species as metadata, identity as the only
  label. This tests "can the model recognize *this individual bat*
  irrespective of species."
- Train ArcFace v2 on the combined manifest. Compare to the per-
  species baselines.

This is the most interesting cheap experiment — a positive result
here is a real contribution; a negative result is still publishable
("identity-disjoint open-set face recognition needs N ≥ … training
identities").

### A3. Cross-species transfer

Train on rousettus, evaluate on the mauritius test split (and vice
versa). True open-set generalization across species.

Concrete next step:
- `bat-cli evaluate` can already point at a different manifest via
  `--hydra data.manifest_path=...`. Train ArcFace v2 on rousettus,
  then call `bat-cli evaluate` against the mauritius manifest's test
  split.
- Repeat in the reverse direction.

Expected outcome: substantially worse than within-species
identity-disjoint already is — but the *gap* is the publishable
quantity. Quantifies how species-specific the learned embedding is.

---

## Priority B — statistical rigor (paper-quality numbers)

A single training run is one data point. Phase-4 results are mostly
single runs; for publication every reported number needs an
uncertainty estimate.

### B1. Multi-seed runs for the key configurations

For each model × dataset combination you want to claim numbers for:
- Run 3–5 seeds (e.g. `cfg.seed=0,1,2,3,4`), all with
  `trainer.deterministic=true`.
- Aggregate: mean ± std on `test/roc_auc`, `test/f1`, `test/top1`.
- Use `bat-cli compare` for pairwise reports; bigger comparisons can
  be done by querying MLflow directly.

The single-tag deterministic mode is now bit-exact (#34), so each
seed gives you a true independent sample with no spurious variance
from infrastructure non-determinism.

### B2. Permutation tests

`bat-cli permutation-test` is already wired; we used it for the auto-
post-training pipeline. For paper-grade claims, run it on the *best*
seed of each model and report p-values for the metrics that matter
(`test/f1`, `test/roc_auc`, identification accuracy). The permutation
null distribution is what makes "ArcFace is better than chance on
this dataset" a defensible claim rather than a vibes-based one.

### B3. Effect sizes

Don't just report p-values. Compute Cohen's d (or paired-difference
equivalent) between configurations you're comparing. The `bat_stats`
package has the building blocks; if a helper isn't there yet, it's
~50 lines of numpy.

---

## Priority C — methods that target small-data face recognition

These are larger lifts, but they're the kinds of methods that
literature suggests *should* close the generalization wall. Pick at
most one or two depending on thesis scope.

### C1. Triplet loss with hard negative mining

The original refactor plan deferred this ("baseline only — not the
recommended path"). On small datasets the literature consensus is
the opposite: arc-margin shines on large datasets (MS1M etc.) and
underperforms triplet + hard mining on small-N regimes — exactly our
regime.

The infrastructure is mostly there:
- `bat_losses.TripletLoss` already exists.
- `bat_data.HardNegativeMiner` and `SemiHardMiner` already exist.
- Wiring them into the train loop is the remaining work — currently
  the PairTrainer only handles BCE/Focal, and the miners aren't
  invoked.

If you only do one Priority-C item, do this one — it's the most
likely method-side win.

### C2. Face-domain pre-training

Currently the ResNet50 backbone is ImageNet-pretrained. That's good
for general visual features but far from face-recognition-specific.
Two options:

- Replace the ImageNet weights with a public face-recognition
  pre-trained checkpoint (e.g. ArcFace-MS1M weights). The
  `bat_models.resnet50_backbone` constructor accepts a
  `pretrained` knob; extending it to load arbitrary state-dicts is
  a small change.
- Self-supervised pre-training (DINO / MAE / SimCLR) on the union
  of bat datasets *without* identity labels, then supervised
  arc-margin fine-tuning on the labeled split. This is the more
  scientifically interesting path; also more work.

### C3. Few-shot / metric-learning approaches

Prototypical Networks, Matching Networks, Relation Networks — designed
exactly for the "few identities, many shots per identity" regime we're
in. None are implemented today; if one is interesting from a paper
perspective, it's a new `bat_models` + `bat_training` family.

---

## Priority D — experimental design + documentation

Most of these are 1–2 day items that don't move metrics directly but
make the *paper* substantially easier to write.

### D1. Codify the experiment matrix

Pin down which (model × loss × dataset × augmentation) combinations
are "the experiments we report" and write them as Hydra experiment
YAMLs in `configs/experiment/`. That makes the paper's experiments
section reproducible by anyone with the repo + manifests.

### D2. Optuna HP sweep on the best architecture

`bat_sweeps` is wired up but unused in this session. Once you've
picked a model family to take to the paper (Siamese is the current
front-runner for this dataset size), run an Optuna sweep with
`bat-cli sweep` on (lr, weight_decay, margin/scale or focal alpha/
gamma, embedding_dim) — val-only objective, won't leak into test.
A few hours of GPU produces defensible HPs to report.

### D3. Write up the paradigm-shift framing

The identity-disjoint vs random-pair distinction (the reason the
legacy 0.9872 is not directly comparable to our 0.7420) is a
methodological point worth putting in the paper's methods section.
The narrative is already in
`docs/PHASE_4_CHECKLIST.md` Step 2 + `docs/PHASE_4_STEP2_FIXES.md`;
it just needs to migrate from internal docs to a paper draft.

---

## Recommended path (opinionated)

If forced to pick a sequence:

1. **A1 augmentation** — cheapest, highest-likelihood win, paper-
   compatible (you'd have to report aug results anyway).
2. **B1 multi-seed + B2 permutation tests** — turns existing
   single-run numbers into reportable mean ± std with p-values.
3. **A2 combined-species manifest** — substantively new data point.
4. **C1 triplet + hard-mining** — likely the best method on this
   dataset size; closes a literature gap in the comparison.
5. **A3 cross-species transfer** — methodologically interesting;
   produces a publishable gap measurement.
6. **D1 experiment-matrix YAMLs + D2 sweep** — once the answers
   are stabilizing, lock them in.

Skip C2/C3 unless the thesis scope explicitly needs them. They are
big lifts that can produce one strong section each but may not be
worth the time vs the cheaper items above.
