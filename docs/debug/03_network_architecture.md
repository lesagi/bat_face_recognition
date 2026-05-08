# Siamese Network Architecture -- Deep Analysis

> Generated: 2026-04-10  
> Source: `app/siamese_core/network.py`

---

## 1. Architecture Overview

The network follows the Koch et al. (2015) "Siamese Neural Networks for One-shot Image Recognition" paper -- a twin-network architecture with shared weights that produces embeddings, computes L1 distance, and classifies same/different via sigmoid.

```mermaid
flowchart LR
    subgraph twinA [Twin A]
        A1[Input Image A<br/>224x224x3] --> A2[Shared Embedding CNN]
    end
    subgraph twinB [Twin B]  
        B1[Input Image B<br/>224x224x3] --> B2[Shared Embedding CNN<br/>same weights]
    end
    A2 --> C["|emb_A - emb_B|<br/>L1 Distance"]
    B2 --> C
    C --> D["Dense(1, sigmoid)<br/>P(same)"]
```

---

## 2. Embedding Network -- Layer-by-Layer Analysis

Source: `network.py` lines 45-73

### Layer Table (CURRENT implementation with bugs)

| # | Layer | Config | Input Shape | Output Shape | Params | Notes |
|---|-------|--------|-------------|--------------|--------|-------|
| 0 | Input | - | - | (224, 224, 3) | 0 | RGB image |
| 1 | Conv2D | 64 filters, 10x10, relu, no padding | (224, 224, 3) | (215, 215, 64) | 19,264 | Kernel: 10*10*3*64 + 64 bias |
| 2 | MaxPooling2D | **pool_size=64**, stride=2, same | (215, 215, 64) | (108, 108, 64) | 0 | **BUG: pool_size=64** |
| 3 | Conv2D | 128 filters, 7x7, relu, no padding | (108, 108, 64) | (102, 102, 128) | 401,536 | Kernel: 7*7*64*128 + 128 |
| 4 | MaxPooling2D | **pool_size=64**, stride=2, same | (102, 102, 128) | (51, 51, 128) | 0 | **BUG: pool_size=64** |
| 5 | Conv2D | 128 filters, 4x4, relu, no padding | (51, 51, 128) | (48, 48, 128) | 262,272 | Kernel: 4*4*128*128 + 128 |
| 6 | MaxPooling2D | **pool_size=64**, stride=2, same | (48, 48, 128) | (24, 24, 128) | 0 | **BUG: pool_size=64** |
| 7 | Conv2D | 256 filters, 4x4, relu, no padding | (24, 24, 128) | (21, 21, 256) | 131,328 | Kernel: 4*4*128*256 + 256 |
| 8 | Flatten | - | (21, 21, 256) | (112,896) | 0 | 21*21*256 = 112,896 |
| 9 | Dense | 4096, sigmoid | (112,896) | (4096) | 462,426,112 | **462M params!** |
| **Total** | | | | | **~463.2M** | |

### Layer Table (CORRECTED -- pool_size=2)

| # | Layer | Config | Input Shape | Output Shape | Params | Notes |
|---|-------|--------|-------------|--------------|--------|-------|
| 0 | Input | - | - | (224, 224, 3) | 0 | RGB image |
| 1 | Conv2D | 64 filters, 10x10, relu | (224, 224, 3) | (215, 215, 64) | 19,264 | |
| 2 | MaxPooling2D | pool_size=2, stride=2, same | (215, 215, 64) | (108, 108, 64) | 0 | Correct 2x2 pool |
| 3 | Conv2D | 128 filters, 7x7, relu | (108, 108, 64) | (102, 102, 128) | 401,536 | |
| 4 | MaxPooling2D | pool_size=2, stride=2, same | (102, 102, 128) | (51, 51, 128) | 0 | Correct 2x2 pool |
| 5 | Conv2D | 128 filters, 4x4, relu | (51, 51, 128) | (48, 48, 128) | 262,272 | |
| 6 | MaxPooling2D | pool_size=2, stride=2, same | (48, 48, 128) | (24, 24, 128) | 0 | Correct 2x2 pool |
| 7 | Conv2D | 256 filters, 4x4, relu | (24, 24, 128) | (21, 21, 256) | 131,328 | |
| 8 | Flatten | - | (21, 21, 256) | (112,896) | 0 | Same flatten size |
| 9 | Dense | 4096, sigmoid | (112,896) | (4096) | 462,426,112 | |
| **Total** | | | | | **~463.2M** | |

**Key observation**: The output shapes are identical between buggy and corrected versions because `padding="same"` with `stride=(2,2)` always halves the spatial dimensions regardless of pool_size. The difference is in **what information is preserved**.

---

## 3. The MaxPooling2D Bug -- Detailed Impact

### What happens with pool_size=64

With `MaxPooling2D(64, (2, 2), padding="same")`:
- The max operation covers a **64x64 window** for each output pixel
- With stride 2, adjacent output pixels share almost their entire receptive field (62 of 64 rows/cols overlap)
- This creates extreme spatial blurring -- fine-grained features (edges, textures, small regions) are destroyed
- Each output neuron sees the maximum value across a massive region, so all local spatial variation is lost

### What should happen with pool_size=2

With `MaxPooling2D((2, 2), (2, 2), padding="same")`:
- The max operation covers a **2x2 window** -- standard downsampling
- Adjacent output pixels have **no overlap** in their pooling windows
- Local features are preserved; only modest spatial resolution is lost
- This is the standard architecture from Koch et al. and virtually every CNN in practice

### Visual comparison of receptive fields

```
pool_size=2 (CORRECT):          pool_size=64 (BUGGY):
+--+--+--+--+--+--+            +--+--+--+--+--+--+--+--+
|##|##|  |  |  |  |            |##|##|##|##|##|##|##|##|
|##|##|  |  |  |  |            |##|##|##|##|##|##|##|##|
+--+--+--+--+--+--+            |##|##|##|##|##|##|##|##|
|  |  |  |  |  |  |            |##|##|##|##|##|##|##|##|
|  |  |  |  |  |  |            |##|##|##|##|##|##|##|##|
+--+--+--+--+--+--+            |##|##|##|##|##|##|##|##|
                                |##|##|##|##|##|##|##|##|
## = one output pixel's         |##|##|##|##|##|##|##|##|
     receptive field            +--+--+--+--+--+--+--+--+

4 input pixels                  4096 input pixels (!)
contribute to 1 output          contribute to 1 output
```

### Why the network still trains (somewhat)

Despite pool_size=64, training doesn't completely fail because:
1. The stride is still (2, 2), so spatial dimensions are halved as expected
2. The network can still learn coarse color/brightness patterns via the max values
3. The Dense(4096) layer has enormous capacity (462M params) that partially compensates
4. But the network **cannot learn fine spatial features** (face details, eye positions, ear shapes) which are critical for individual bat identification

---

## 4. Full Siamese Model Architecture

### Twin-input model structure

```
Input A: (batch, 224, 224, 3) ─┐
                                ├─> Shared Embedding ─> emb_A: (batch, 4096)
Input B: (batch, 224, 224, 3) ─┤                                    │
                                └─> Shared Embedding ─> emb_B: (batch, 4096)
                                                                     │
                                                    L1Dist: |emb_A - emb_B|
                                                            (batch, 4096)
                                                                     │
                                                    Dense(1, sigmoid, float32)
                                                            (batch, 1)
                                                                     │
                                                    Output: P(same identity)
```

### Parameter count breakdown

| Component | Parameters |
|-----------|-----------|
| Embedding CNN (shared, counted once) | ~463.2M |
| L1Dist layer | 0 (pure computation) |
| Final Dense(1) classifier | 4,097 (4096 weights + 1 bias) |
| **Total trainable** | **~463.2M** |

The model is **extremely parameter-heavy** due to the Dense(4096) layer taking 112,896 inputs from Flatten. This is consistent with the original Koch et al. paper but is an overfitting risk without regularization.

---

## 5. Distance Layer Analysis

### Current: L1 Distance (`network.py` lines 9-14)

```python
class L1Dist(Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
```

- Computes element-wise absolute difference: |emb_A - emb_B|
- Output shape: same as embedding (4096)
- No learnable parameters
- Followed by Dense(1, sigmoid) for binary classification

### Considerations

The L1 distance + sigmoid classifier is correct per Koch et al. Other options used in modern Siamese networks:
- **L2 distance** (Euclidean) with contrastive loss
- **Cosine similarity** with cross-entropy
- **Learned distance metric** (additional dense layers on concatenated embeddings)

For this binary verification task (same bat / different bat), L1 + sigmoid is a valid approach.

---

## 6. Loss Function Analysis

### Training loss (`trainer.py` lines 162-164)

```python
self.loss_function = tf.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
self.test_loss_function = tf.losses.BinaryCrossentropy()
```

- **Train**: `reduction=NONE` returns per-sample losses for weighted loss computation
- **Test**: Default reduction (mean over batch)
- BCE is appropriate for binary same/different classification with sigmoid output
- This is NOT contrastive loss or triplet loss -- it's a direct classification approach

### Weight application in train_step

```python
weights = weights / tf.reduce_mean(weights)       # Normalize to mean=1
weighted_loss = per_sample_loss * weights_casted   # Apply weights
loss = tf.reduce_mean(weighted_loss)               # Final scalar loss
```

Weight normalization to mean=1.0 ensures loss magnitude remains comparable across batches.

---

## 7. Activation Function Concerns

### Sigmoid in embedding (Dense 4096)

The embedding layer uses `sigmoid`:
- Squashes all 4096 dimensions to [0, 1]
- Gradient: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)), max value 0.25 at x=0
- For saturated neurons (output near 0 or 1), gradients vanish
- The L1 distance of two sigmoid embeddings is bounded to [0, 1] per dimension
- Maximum total L1 distance: 4096; minimum: 0

### ReLU alternative (modern practice)

Using `relu` for the embedding:
- Unbounded positive activations
- Gradient = 1 for positive inputs (no vanishing gradient)
- L1 distance is unbounded, giving more dynamic range
- But requires careful initialization and possibly normalization

### Sigmoid in classifier (Dense 1)

The final `Dense(1, activation="sigmoid")` is correct -- it outputs a probability P(same).

---

## 8. Mixed Precision Interaction

When `mixed_precision=true` (config default):
- Global policy: `mixed_float16`
- Compute dtype: float16 (conv, dense forward/backward)
- Variable dtype: float32 (weight storage)
- Final classifier: `Dense(1, sigmoid, dtype='float32')` -- explicitly float32 to avoid numerical issues in loss computation
- Optimizer wrapped in `LossScaleOptimizer` for gradient scaling

This is correctly implemented and should not cause issues.
