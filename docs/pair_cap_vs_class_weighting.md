# Interaction Between `max_samples_per_class` Cap and Per-Class Weighting

## Summary

When `max_samples_per_class` is set, the pair-generation cap **structurally equalizes**
the number of training pairs contributed by each class — making per-class loss weighting
(ISNS/INS/ENS) largely redundant for classes at or above the cap.
Only classes whose *training* image count falls **below** the cap still have reduced
pair-level representation, and per-class weighting only helps those.

---

## How the Cap Works During Pair Generation

### Anchor (positive) pairs

```
effective_samples = min(train_images, max_samples_per_class)
anchor_pairs      = P(effective_samples, 2)   # permutation mode
                  = effective_samples × (effective_samples − 1)
```

### Negative pairs (per class-pair)

```
min_eff = min(eff_class_a, eff_class_b)
negative_pairs = 2 × min_eff²                 # permutation mode (both directions)
```

Because both formulas depend only on `effective_samples`, any class with
`train_images ≥ max_samples_per_class` contributes **identically**.

---

## Concrete Numbers — Mauritius Video Dataset (11 classes)

| Class    | Raw imgs | Train imgs (70%) | Eff @ cap=20 | Eff @ cap=30 |
|----------|----------|-------------------|--------------|--------------|
| 060427   | 228      | 160               | 20           | 30           |
| 055535   | 158      | 111               | 20           | 30           |
| 060839   | 123      | 86                | 20           | 30           |
| 141252   | 122      | 85                | 20           | 30           |
| 135124   | 98       | 69                | 20           | 30           |
| 142118   | 96       | 67                | 20           | 30           |
| 135909   | 75       | 52                | 20           | 30           |
| 140541   | 73       | 51                | 20           | 30           |
| 143044   | 72       | 50                | 20           | 30           |
| 133234   | 44       | 31                | 20           | 30           |
| **155428** | **30** | **21**            | **20**       | **21** ← not capped |

### With `max_samples_per_class = 20` (old value)

Every class has ≥ 21 train images → all capped to 20.

| Metric | Value |
|--------|-------|
| Anchor pairs per class | 380 (all identical) |
| Negative pairs per class-pair | 800 (all identical) |
| Pair-level appearances per class | 8,380 (all identical) |
| **Per-class weight ratio (ISNS)** | **1.00× — weighting is a complete no-op** |

### With `max_samples_per_class = 30` (new value)

All classes have ≥ 31 train images **except** class 155428 (21 train images).

| Class | Eff | Anchor pairs | Pair appearances | ISNS weight |
|-------|-----|-------------|-----------------|-------------|
| 10 capped classes | 30 | 870 | 17,952 | 0.965 |
| **155428** | **21** | **420** | **9,240** | **1.346** |

- Pair-level imbalance ratio: **1.94×** (only for the one uncapped class)
- ISNS weight ratio: **1.39×** — a modest correction
- Compare to raw image count ratio: **7.6×** (228 vs 30 images)

**The cap compresses 7.6× original imbalance down to 1.94× at the pair level,
and ISNS only adds a further 1.39× correction on top of that.**

---

## Why This Happens

The pair-generation cap acts as a **structural balancer**: by limiting how many
images each class contributes to pair formation, it forces large and small classes
(above the cap) to produce the same number of pairs. This is the dominant
balancing mechanism.

Per-class weighting, computed from the `sampled` global distribution strategy,
measures class frequency in the **actual pair dataset** — which is already
nearly flat due to the cap. So the weights come out nearly equal and don't do
much additional work.

### Comparison of distribution strategies

| Strategy | What it measures | Cap-aware? | Effect with cap |
|----------|-----------------|------------|-----------------|
| `sampled` (default) | Class frequency in actual pair dataset | Yes (implicitly) | Weights ≈ 1.0 for capped classes |
| `full_scan` | Same as sampled but exhaustive | Yes (implicitly) | Same as sampled |
| `file_based` | Estimated pairs from raw file counts | **No** | Weights reflect original imbalance |

With `file_based`, the ISNS weight ratio would be **2.76×** (largest to smallest
class), reflecting the actual image-count imbalance even though the cap has
already equalized pair counts.

---

## Implications and Trade-offs

### Current behavior (sampled + cap)

- The cap provides structural balance — effective and simple.
- Per-class weighting adds almost nothing for classes above the cap.
- Only classes below the cap get modest upweighting (1.39× for class 155428).
- **Extra images beyond the cap are completely unused** during training.
  A class with 228 images and one with 30 images contribute identically
  if both have ≥ cap training images.

### Alternative: `file_based` + cap

Using `file_based` distribution strategy with the cap would mean:
"All classes contribute equal pairs, but we still upweight rare classes in
the loss because they have less underlying diversity."
This is a defensible approach — fewer source images means less visual
variety, so heavier loss emphasis compensates.

### Alternative: no cap + per-class weighting only

Removing the cap and relying entirely on per-class weighting would let
class sizes vary naturally in the pair dataset and use weighting to
rebalance. This risks very large pair counts for big classes and slower
training unless managed.

### Alternative: higher cap

Raising the cap (e.g., to 50 or 100) would let more of the original
imbalance show through, giving per-class weighting more to work with.
However, it also increases total pair count and training time.

---

## Recommendation

The cap is a useful structural equalizer and should be kept. To decide
whether per-class weighting adds value on top of it:

1. **If all classes are above the cap** — per-class weighting is a no-op
   with `sampled`/`full_scan`. Either disable `per_class_balance` or
   switch to `file_based` if you want diversity-aware reweighting.

2. **If some classes fall below the cap** (current situation with
   cap=30, class 155428 has only 21 train images) — per-class weighting
   with `sampled` provides a small correction. Switching to `file_based`
   would give a stronger correction.

3. **Consider the goal**: the cap already ensures each class gets equal
   "air time" in the training loop. Per-class weighting on top of that
   should target *diversity compensation*, not pair-count balance — which
   favours `file_based` over `sampled` when a cap is active.

---

## Additional Loss-Level Balancing Mechanisms

The pair cap and per-class weighting (sections above) address **inter-class**
imbalance. Two additional mechanisms address the **anchor vs. negative** imbalance
and the dominance of easy examples in the loss.

### Asymmetric Anchor/Negative Ratio (`anchor_target_ratio`)

By default, anchor/negative balancing equalizes total gradient contribution
(50/50 split). The `anchor_target_ratio` parameter shifts this balance.

| `anchor_target_ratio` | Anchor contribution | Negative contribution | Effect |
|----------------------|--------------------|-----------------------|--------|
| 0.5 (default)        | 50%                | 50%                   | Equal contribution |
| 0.6                   | 60%                | 40%                   | Moderate positive emphasis |
| 0.7                   | 70%                | 30%                   | Strong positive emphasis |

Use values > 0.5 when the model has a **negative bias** — i.e., it predicts
"different" too often and misses same-bat pairs. This directly increases the
per-sample weight for anchor pairs so each false negative contributes more
gradient than each false positive.

**Interaction with the cap**: The cap equalizes *class counts* in the pair
dataset, but the anchor/negative ratio operates at the *pair type* level
(positive vs negative). They are orthogonal and stack multiplicatively.

### Binary Focal Loss

Focal loss replaces BinaryCrossentropy and automatically down-weights easy
examples while focusing on hard ones:

```
FL(p_t) = -alpha_t × (1 - p_t)^gamma × log(p_t)
```

| Parameter | Purpose | Typical value |
|-----------|---------|---------------|
| `focal_alpha` | Positive class weight (>0.5 favors same-bat pairs) | 0.75 |
| `focal_gamma` | Focusing strength (0 = standard BCE) | 2.0 |

With gamma=2.0, a well-classified negative pair (confidence 0.9) receives
**100× less loss** than a misclassified positive pair. This naturally shifts
training focus to the hard cases where the model incorrectly predicts "different"
for same-bat pairs.

**Interaction with per-class weighting**: Focal loss and per-class weights stack
multiplicatively. Focal loss handles the *easy vs. hard example* axis while
per-class weights handle the *rare vs. common class* axis. Both can be active
simultaneously.

**Interaction with anchor_target_ratio**: The focal `alpha` parameter and the
`anchor_target_ratio` both shift weight toward positives, but via different
mechanisms — alpha acts at the loss level per sample, while anchor_target_ratio
acts at the batch-level contribution normalization. When using focal loss, the
`alpha` parameter often subsumes the need for `anchor_target_ratio > 0.5`, but
both can be combined for more aggressive positive emphasis.

### Threshold Optimization

After training, the system automatically finds the optimal decision threshold
using Youden's J statistic (maximizing sensitivity + specificity - 1) on the
test set. This threshold is saved to `training_summary.json` and used
automatically during evaluation instead of the default 0.5.

---

## Configuration Reference

```yaml
siamese_network:
  training:
    max_samples_per_class: 30          # Structural pair-count balancer
    loss:
      type: "BinaryFocalLoss"          # or "BinaryCrossentropy"
      focal_alpha: 0.75                # Positive class weight (focal loss only)
      focal_gamma: 2.0                 # Focusing parameter (focal loss only)
  class_balancing:
    anchor_negative_balance: true      # Per-batch anchor vs negative balancing
    anchor_target_ratio: 0.5           # 0.5=equal, >0.5 favors anchors
    per_class_balance: true            # Loss-level per-class weighting
    weighting_scheme: "isns"           # 1/√(count)
    global_distribution_strategy: "sampled"  # Measures post-cap pair distribution
```
