# Issues Found -- Severity-Ranked Debug Report

> Generated: 2026-04-10 (updated 2026-04-11 after Koch et al. paper review and figure analysis)  
> Scope: Full Siamese network training pipeline  
> Reference: Koch et al. (2015) "Siamese Neural Networks for One-shot Image Recognition"

---

## Paper Verification Note

All issues have been cross-referenced against the original Koch et al. (2015) paper.
Key paper quotes used for verification:

- Section 4.1: *"max-pooling with a filter size and stride of 2"*
- Section 4.1: *"We use exclusively rectified linear (ReLU) units in the first L-2 layers and sigmoidal units in the remaining layers."*
- Section 4.2: *"L2 regularization weights lambda_j defined layer-wise"*
- Section 4.2: *"learning rates were decayed uniformly across the network by 1 percent per epoch"*

---

## Summary Table

| # | Severity | Component | Issue | File | Paper-Verified? |
|---|----------|-----------|-------|------|-----------------|
| 1 | **CRITICAL** | Network | MaxPooling2D pool_size=64 instead of 2 | `siamese_core/network.py:58-68` | YES -- paper says pool size = 2 |
| 2 | **CRITICAL** | Training | train_loss = last batch only, not epoch average | `siamese_training/trainer.py:860` | N/A (implementation bug) |
| 3 | **HIGH** | Data | load_config() called per-image in tf.data.map | `siamese_data/data_splitter.py:36` | N/A (implementation bug) |
| 4 | **HIGH** | Data | try/except in graph-mode tf.data pipeline | `siamese_data/data_splitter.py:30-33` | N/A (implementation bug) |
| 5 | **DESIGN NOTE** | Network | Sigmoid embedding activation | `siamese_core/network.py:71` | Paper intentionally uses sigmoid |
| 6 | **MEDIUM** | Network | No L2 regularization (paper uses it) | `siamese_core/network.py:45-73` | YES -- paper has per-layer L2 |
| 7 | **MEDIUM** | Data | Positive/negative pair count imbalance | `siamese_data/data_splitter.py` | Paper samples balanced pairs |
| 8 | **MEDIUM** | Training | test() uses model.predict() (slow) instead of __call__ | `siamese_training/trainer.py:972` | N/A (implementation bug) |
| 9 | **MEDIUM** | Training | No learning rate schedule (paper decays 1%/epoch) | `siamese_training/trainer.py` | YES -- paper decays LR |
| 10 | **LOW** | Data | preprocess fallback returns black image silently | `siamese_data/data_splitter.py:57` | N/A |
| 11 | **LOW** | Training | Nested MLflow runs per epoch (overhead) | `siamese_training/trainer.py:472` | N/A |

---

## Issue 1: CRITICAL -- MaxPooling2D pool_size=64

**STATUS: FIXED** (2026-04-11)

### Location
`app/siamese_core/network.py` lines 58, 62, 66

### Original Code (before fix)
```python
c1 = Conv2D(64, (10, 10), activation="relu")(inp)
m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

c2 = Conv2D(128, (7, 7), activation="relu")(m1)
m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

c3 = Conv2D(128, (4, 4), activation="relu")(m2)
m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)
```

### Root Cause

The paper's Figure 4 labels max-pooling operations as **"64 @ 2x2"**. This notation describes the **output**: 64 feature maps flowing through a 2x2 pooling window. The "64" is the channel count, NOT a pooling parameter. The developer misread "64" as the first argument to `MaxPooling2D`.

The figure itself also contains a copy-paste error: all three max-pooling labels say "64 @ 2x2", but Pool 2 and Pool 3 actually output 128 channels (matching their preceding Conv2 and Conv3 layers). Only Pool 1 correctly shows 64 channels.

Since `MaxPooling2D(pool_size, strides, padding)` takes `pool_size` as its first positional argument, passing `64` creates a **64x64 pooling window** instead of the intended 2x2.

### Paper Evidence (Section 4.1)

The text is unambiguous:
> *"The network applies a ReLU activation function to the output feature maps, optionally followed by **max-pooling with a filter size and stride of 2**."*

And in the mathematical formulation:
> *a(k)_{1,m} = max-pool(max(0, W(k)_{l-1,l} * h_{1,(l-1)} + b_l), **2**)*

### Verification via Paper Dimensions (105x105 grayscale input)

| Layer | Operation | Paper Output | Math |
|---|---|---|---|
| Conv1 | 64 filters, 10x10, stride 1 | 64 @ 96x96 | 105-10+1=96 |
| Pool1 | pool_size=(2,2), strides=(2,2) | 64 @ 48x48 | 96/2=48 |
| Conv2 | 128 filters, 7x7, stride 1 | 128 @ 42x42 | 48-7+1=42 |
| Pool2 | pool_size=(2,2), strides=(2,2) | 128 @ 21x21 | 42/2=21 |
| Conv3 | 128 filters, 4x4, stride 1 | 128 @ 18x18 | 21-4+1=18 |
| Pool3 | pool_size=(2,2), strides=(2,2) | 128 @ 9x9 | 18/2=9 |
| Conv4 | 256 filters, 4x4, stride 1 | 256 @ 6x6 | 9-4+1=6 |
| Flatten | | 9216 | 6×6×256 |
| Dense | sigmoid | 4096 | |

All dimensions match the figure perfectly, confirming pool_size=(2,2) with strides defaulting to pool_size.

### Impact
- A 64x64 pooling window takes the max over 4,096 input pixels per output pixel, destroying all spatial detail
- Network can only learn coarse global features (color distribution, overall brightness)
- The model effectively operates as a "color histogram comparator" rather than a face recognizer

### Applied Fix
All layer calls now use explicit named parameters:
```python
c1 = Conv2D(filters=64, kernel_size=(10, 10), activation="relu")(inp)
m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)

c2 = Conv2D(filters=128, kernel_size=(7, 7), activation="relu")(m1)
m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c2)

c3 = Conv2D(filters=128, kernel_size=(4, 4), activation="relu")(m2)
m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c3)
```

**Note on strides:** When `strides` is omitted, Keras defaults it to `pool_size`. Specifying it explicitly here for clarity — the paper says "filter size and stride of 2".

**Note on padding:** The paper uses valid padding (default). The code retains `padding="same"` — a secondary discrepancy to revisit later.

### Expected Impact of Fix
Major improvement in the model's ability to distinguish individual bats. The network will learn local features (eye spacing, ear shape, fur patterns) rather than just global color statistics.

---

## Issue 2: CRITICAL -- Training Loss = Last Batch Only

### Location
`app/siamese_training/trainer.py` lines 840-860

### Current Code
```python
for idx, batch in enumerate(self.train_batches):
    loss, yhat = self.train_step(batch)
    r.update_state(batch[2], yhat)
    p.update_state(batch[2], yhat)
    ...
train_loss = self._to_float(loss)  # <-- ONLY the last batch!
```

### Problem
The `loss` variable is overwritten each iteration. After the loop, it contains only the final batch's loss value. Recall and Precision use Keras stateful metrics that correctly accumulate, but loss does not.

### Evidence
- `r = Recall()` and `p = Precision()` are stateful metrics that accumulate via `.update_state()`
- `loss` is a plain scalar tensor, overwritten each batch
- Contrast with `test()` method which correctly accumulates: `total_loss = total_loss + batch_loss`

### Impact
- Logged `train_loss` is noisy and unrepresentative (one batch out of hundreds)
- Loss curves in MLflow will appear erratic
- Cannot reliably detect training divergence from train_loss alone
- Model selection still works (based on test_loss), so models are not corrupted

### Proposed Fix
```python
total_train_loss = 0.0
num_train_batches = 0

for idx, batch in enumerate(self.train_batches):
    loss, yhat = self.train_step(batch)
    total_train_loss += self._to_float(loss)
    num_train_batches += 1
    r.update_state(batch[2], yhat)
    p.update_state(batch[2], yhat)
    ...

train_loss = total_train_loss / num_train_batches
```

### Expected Impact of Fix
Smooth, representative train_loss curves. Easier to diagnose overfitting (diverging train/test loss).

---

## Issue 3: HIGH -- load_config() Called Per-Image

### Location
`app/siamese_data/data_splitter.py` lines 36-37 (and 55-56 in fallback)

### Current Code
```python
def preprocess_siamese_input(file_path):
    try:
        byte_img = tf.io.read_file(file_path)
        ...
        cfg = load_config()  # YAML parse every single image!
        input_edge = cfg.siamese_network.model.get("input_edge_length", 224)
        img = tf.image.resize(img, (input_edge, input_edge))
        ...
```

### Problem
`load_config()` parses the YAML configuration file from disk. This function is called inside `tf.data.Dataset.map()`, which means it executes for every image, every epoch. With thousands of pairs (2 images each), this is tens of thousands of YAML parses per epoch.

### Evidence
- `preprocess_twin_input_function` calls `preprocess_siamese_input` twice (once per image in the pair)
- Dataset is mapped via `.map(self.preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)`
- `load_config()` in `config/loader.py` reads and parses YAML from disk each call

### Impact
- Massive I/O overhead during training
- CPU bottleneck that can starve the GPU
- Each epoch spends significant time on redundant config parsing
- The config value (`input_edge_length = 224`) never changes during training

### Proposed Fix
Read the config once at module level or pass the value as a closure:
```python
_INPUT_EDGE = None

def _get_input_edge():
    global _INPUT_EDGE
    if _INPUT_EDGE is None:
        cfg = load_config()
        _INPUT_EDGE = cfg.siamese_network.model.get("input_edge_length", 224)
    return _INPUT_EDGE

def preprocess_siamese_input(file_path):
    input_edge = _get_input_edge()
    ...
```

Or better yet, use the constant already defined in `network.py`:
```python
from siamese_core.network import SIAMESE_INPUT_EDGE_LENGTH
```

### Expected Impact of Fix
Significant training speed improvement, especially with large datasets.

---

## Issue 4: HIGH -- try/except in Graph-Mode tf.data Pipeline

### Location
`app/siamese_data/data_splitter.py` lines 30-33

### Current Code
```python
try:
    img = tf.io.decode_png(byte_img)
except:
    img = tf.io.decode_jpeg(byte_img)
```

### Problem
When `tf.data.Dataset.map()` traces functions into a TensorFlow graph (which it does with `num_parallel_calls=tf.data.AUTOTUNE`), Python `try/except` does not intercept TensorFlow operation errors. The error would occur at graph execution time, not at Python trace time.

### Evidence
- TF documentation explicitly warns against Python control flow in graph-traced functions
- `tf.io.decode_png` on a JPEG file will raise an `InvalidArgumentError` at execution time, which Python `try/except` cannot catch in graph mode
- The function is used via `.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)`

### Impact
- JPEG images may fail to decode, causing pipeline errors or silent data corruption
- If all images happen to be PNG, this bug is dormant but will break when JPEG images are introduced
- The outer try/except on lines 26-57 has the same graph-mode problem

### Proposed Fix
Use `tf.io.decode_image()` which handles both PNG and JPEG automatically:
```python
byte_img = tf.io.read_file(file_path)
img = tf.io.decode_image(byte_img, channels=3, expand_animations=False)
img.set_shape([None, None, 3])
```

### Expected Impact of Fix
Robust image loading regardless of format. No silent failures.

---

## Issue 5: DESIGN NOTE -- Sigmoid Embedding Activation (Faithful to Paper)

### Location
`app/siamese_core/network.py` line 71

### Current Code
```python
d1 = Dense(4096, activation="sigmoid")(f1)
```

### Paper Verification: THIS IS CORRECT PER THE PAPER

Koch et al. (2015) Section 4.1 explicitly states:
> *"We use exclusively **rectified linear (ReLU) units in the first L-2 layers** and **sigmoidal units in the remaining layers**."*

The architecture uses ReLU for all conv layers (first L-2 layers) and sigmoid for the fully-connected embedding layer and the final output layer. The code correctly follows this design.

### Why it still matters (but is NOT a bug)

Sigmoid in the embedding layer is the paper's intentional design from 2015. It is not a code error. However, it's worth documenting that:

1. Sigmoid squashes all 4096 embedding dimensions to [0, 1]
2. This can cause vanishing gradients for saturated neurons (sigmoid'(x) -> 0 for large |x|)
3. Modern Siamese/metric-learning networks (FaceNet 2015, ArcFace 2019) moved to ReLU or linear embeddings with L2 normalization

### Recommendation (optional future improvement, NOT a fix)

If training converges poorly even after fixing the MaxPooling2D bug (Issue 1), consider experimenting with ReLU embeddings as a modernization. But this should be tested as a deliberate architectural change, not treated as a bug fix:
```python
# Option A: ReLU (modern practice)
d1 = Dense(4096, activation="relu")(f1)

# Option B: Keep sigmoid (faithful to paper -- current code is correct)
d1 = Dense(4096, activation="sigmoid")(f1)
```

### Verdict
**Not a bug.** The code correctly implements the paper's architecture. Downgraded from HIGH to DESIGN NOTE.

---

## Issue 6: MEDIUM -- No L2 Regularization (Paper Uses Per-Layer L2)

### Location
`app/siamese_core/network.py` lines 45-73, `app/siamese_training/trainer.py` line 67

### Problem
The embedding network has zero regularization. The paper explicitly uses **per-layer L2 regularization**.

### Paper Evidence (Section 4.2)
Koch et al. specifies a regularized cross-entropy objective:
> *L(x1, x2) = y * log(p) + (1-y) * log(1-p) + **lambda_T * |w|^2***

And the optimization includes:
> *"L2 regularization weights **lambda_j** defined **layer-wise**"*
> *"lambda_j in [0, 0.1]"*

The paper uses L2 weight decay on every layer, tuned via Bayesian hyperparameter optimization. The code has none.

### Additional Context
- The paper does NOT use BatchNorm (it predates widespread BatchNorm adoption)
- The paper does NOT mention Dropout
- But L2 regularization is explicitly part of the paper's loss function
- 462M parameters in Dense(4096) alone, severe overfitting risk without regularization

### Impact
- Model likely overfits to training data
- Training metrics look good but test metrics stagnate or degrade
- The network memorizes training pairs rather than learning generalizable features

### Proposed Fix (faithful to paper -- add L2 regularization)
```python
from tensorflow.keras.regularizers import l2

REG = l2(1e-4)  # Paper says lambda in [0, 0.1], tune via hyperparameter search

c1 = Conv2D(filters=64, kernel_size=(10, 10), activation="relu", kernel_regularizer=REG)(inp)
m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)

c2 = Conv2D(filters=128, kernel_size=(7, 7), activation="relu", kernel_regularizer=REG)(m1)
m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c2)

c3 = Conv2D(filters=128, kernel_size=(4, 4), activation="relu", kernel_regularizer=REG)(m2)
m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c3)

c4 = Conv2D(filters=256, kernel_size=(4, 4), activation="relu", kernel_regularizer=REG)(m3)
f1 = Flatten()(c4)
d1 = Dense(units=4096, activation="sigmoid", kernel_regularizer=REG)(f1)
```

### Optional Modern Enhancement (beyond the paper)
If overfitting persists after adding L2, consider adding BatchNorm and Dropout:
```python
c1 = Conv2D(filters=64, kernel_size=(10, 10), activation="relu", kernel_regularizer=REG)(inp)
c1 = BatchNormalization()(c1)
# ...
f1 = Dropout(0.5)(f1)
```

### Expected Impact of Fix
Better generalization, reduced overfitting, more stable training. Aligns code with the paper's actual regularization strategy.

---

## Issue 7: MEDIUM -- Positive/Negative Pair Count Imbalance

**STATUS: INVESTIGATED (2026-04-11) -- Existing weighting is largely adequate**

### Location
`app/siamese_data/data_splitter.py` -- `__create_anchor_pairs` vs `__create_negative_pairs`

### Raw Count Imbalance
- **Positive pairs**: combinations or permutations of images within a class -> O(n^2) pairs from n images
- **Negative pairs**: Cartesian product across two classes -> O(n * m) pairs
- With 10 classes of 25 images each: ~9.4:1 negative to positive ratio

### Existing Weighting Mechanisms (3 layers)

After thorough investigation, the codebase has a **multi-layered** weighting system that addresses this:

1. **Anchor/Negative per-batch balance** (`anchor_negative_weights.py` + TF reimplementation in `trainer.py`):
   - Computes per-batch counts of positives vs negatives
   - Sets `anchor_weight = 0.5 * total / n_anchors`, `negative_weight = 0.5 * total / n_negatives`
   - **Effect:** Total weighted loss contribution from positives equals negatives within each batch
   - Config: `class_balancing.anchor_negative_balance: true`

2. **Per-class frequency weighting** (`class_weights.py`):
   - Supports INS (1/n), ISNS (1/sqrt(n)), and ENS (Cui et al. CVPR 2019) schemes
   - Uses global class distribution computed via configurable strategy (file_based/sampled/full_scan)
   - Config: `class_balancing.per_class_balance: true`, `weighting_scheme: "isns"`

3. **Negative pair combination** (`class_weights.py`):
   - For negative pairs (two classes), combines both class weights using sum/geometric_mean/product
   - Config: `negative_pair_combination: "sum"`

4. **Mean normalization**: All weights are divided by their mean to keep loss scale stable.

5. **TF hash table lookup**: Precomputed for GPU-efficient weight application in `train_step`.

### Assessment

The anchor/negative balance mechanism **directly solves the core concern**: regardless of the 9:1 raw count ratio, each batch's loss is split 50/50 between positive and negative contributions. Combined with per-class ISNS weighting for rare classes and the combination rule for negative pairs, this is a solid multi-level approach.

### Residual Limitations (minor)

1. **Joint pair-frequency imbalance**: Weights depend on marginal class frequencies, not specific (A,B) pair frequencies
2. **Per-batch vs global**: 50/50 split is per-batch, not enforced globally
3. **TF edge case**: `_compute_anchor_negative_weights_tf` uses `tf.maximum(n, 1e-6)` for single-label batches instead of the NumPy special cases — rare with shuffled data but technically inconsistent

### Verdict
**No code change needed.** The existing weighting system adequately addresses the pair imbalance. The residual limitations are minor and theoretical rather than practical blockers.

---

## Issue 8: MEDIUM -- test() Uses model.predict() Instead of Direct Call

### Location
`app/siamese_training/trainer.py` line 972

### Current Code
```python
yhat = self.siamese_model.predict([test_input, test_val])
```

### Problem
`model.predict()` has significant overhead per call:
- Creates a new execution graph each call
- Cannot be used inside `@tf.function`
- Much slower than `model(inputs, training=False)`

### Evidence
Contrast with `train_step` which correctly uses direct call:
```python
yhat = self.siamese_model(x, training=True)  # Fast, graph-compatible
```

### Impact
- Test evaluation is slower than necessary
- Noticeable on large test sets

### Proposed Fix
```python
yhat = self.siamese_model([test_input, test_val], training=False)
```

### Expected Impact of Fix
Faster test evaluation. Could also wrap the test loop in `@tf.function`.

---

## Issue 9: MEDIUM -- No Learning Rate Schedule (Paper Decays 1% Per Epoch)

### Location
`app/siamese_training/trainer.py` line 67 (optimizer creation)

### Current Code
```python
optimizer=tf.keras.optimizers.Adam(1e-4)
```

Fixed learning rate throughout training. No schedule.

### Paper Evidence (Section 4.2)
Koch et al. explicitly specifies learning rate annealing:
> *"learning rates were decayed uniformly across the network by **1 percent per epoch**, so that eta(T)_j = **0.99 * eta(T-1)_j**"*

The paper also specifies:
> *"layer-wise learning rate eta_j in [10^-4, 10^-1]"*
> *"momentum to start at 0.5 in every layer, increasing linearly each epoch"*

Note: The paper uses SGD with momentum, not Adam. Adam has built-in adaptive learning rates, which partially compensates for the lack of an explicit schedule. However, a decay schedule is still beneficial to refine convergence in later epochs.

### Impact
- Without LR decay, the optimizer may oscillate around the minimum in later epochs
- The paper found LR annealing critical: *"the network was able to converge to local minima more easily without getting stuck"*
- Fixed LR with Adam is acceptable but not optimal

### Proposed Fix
```python
initial_lr = 5e-5  # from config
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=len(train_batches),  # per epoch
    decay_rate=0.99,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Expected Impact of Fix
Better convergence in later epochs. Aligns with the paper's training procedure.

---

## Issue 10: LOW -- Silent Black Image Fallback

### Location
`app/siamese_data/data_splitter.py` lines 52-57

### Current Code
```python
except Exception as e:
    tf.print(f"Error processing {file_path}: {e}")
    cfg = load_config()
    input_edge = cfg.siamese_network.model.get("input_edge_length", 224)
    return tf.zeros((input_edge, input_edge, 3), dtype=tf.float32)
```

### Problem
If an image fails to load, a black (all-zeros) tensor is silently substituted. This means:
- Corrupted or missing images produce valid-looking tensors
- The model trains on meaningless black images as if they were real data
- No aggregated count of how many images failed
- The outer try/except also won't work properly in graph mode (Issue 4)

### Impact
- Silently corrupted training data
- Black images paired with real images produce nonsensical training signal
- Difficult to diagnose data quality issues

### Proposed Fix
Log failures to a counter and raise if too many fail. Or use `tf.debugging.assert` to flag issues.

---

## Issue 11: LOW -- Nested MLflow Runs Per Epoch

### Location
`app/siamese_training/trainer.py` lines 472-489

### Current Code
```python
with mlflow.start_run(run_name=f"epoch_{epoch}", nested=True):
    mlflow.log_param("epoch", int(epoch))
    mlflow.log_metrics({...})
```

### Problem
Creating a nested MLflow run for every epoch adds I/O overhead and clutters the MLflow UI. Metrics are already logged to the parent run with `step=epoch` (lines 457-469), making the nested runs redundant.

### Impact
- MLflow UI shows 80+ child runs per training run
- Additional file I/O per epoch
- Redundant data storage

### Proposed Fix
Remove the nested run block. The parent run step-wise metrics are sufficient:
```python
# Keep only the parent run logging (already present at lines 457-469)
mlflow.log_metrics({...}, step=epoch)
```

---

## Priority Ranking for Fixes

### Must fix immediately (blocks correct training)
1. **Issue 1**: MaxPooling2D pool_size=64 -> 2 -- fundamentally broken feature extraction (CONFIRMED BY PAPER)
2. **Issue 2**: Train loss logging -- cannot diagnose training without correct loss

### Should fix before next training run (correctness + performance)
3. **Issue 4**: try/except in graph mode -- data loading correctness
4. **Issue 3**: load_config() per image -- major perf bottleneck

### Align with paper's training procedure
5. **Issue 6**: Add L2 regularization -- paper uses per-layer L2, code has none
6. **Issue 9**: Add LR schedule -- paper decays 1%/epoch, code uses fixed LR

### Fix for better results (optimization)
7. **Issue 7**: Pair imbalance -- balanced training signal
8. **Issue 8**: model.predict() -> model() -- test speed

### Not a bug (design note only)
9. **Issue 5**: Sigmoid embedding -- correct per paper, optional modern upgrade

### Nice to have
10. **Issue 10**: Silent fallback -- data quality monitoring
11. **Issue 11**: Nested MLflow runs -- UI cleanliness
