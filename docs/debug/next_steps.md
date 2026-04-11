# Next Steps -- Future Considerations

> Created: 2026-04-11

---

## 1. Experiment with ReLU Embedding Activation (Currently Sigmoid)

**Current:** `Dense(units=4096, activation="sigmoid")` — faithful to Koch et al. (2015).

**Rationale for change:** Modern metric-learning networks (FaceNet 2015, ArcFace 2019) use ReLU or linear + L2-normalize for embeddings. Sigmoid squashes all values to [0, 1], which can cause vanishing gradients at saturation and limits representational capacity.

**Why NOT a bug:** The paper explicitly specifies sigmoid for the embedding layer (Section 4.1: *"sigmoidal units in the remaining layers"*). The current code is correct per the paper.

**Recommendation:** After fixing all critical bugs (MaxPooling, loss accumulation, etc.) and establishing a baseline, run an A/B experiment:
- **Baseline:** sigmoid embedding (current, paper-faithful)
- **Experiment:** ReLU embedding, same architecture otherwise

```python
# Option A: ReLU (modern practice)
d1 = Dense(units=4096, activation="relu")(f1)

# Option B: ReLU + L2 normalize (FaceNet-style)
d1 = Dense(units=4096, activation="relu")(f1)
d1 = tf.math.l2_normalize(d1, axis=1)
```

**Priority:** Low — only after all bugs are fixed and baseline is established.

---

## 2. Padding: "same" vs "valid" in MaxPooling2D

**Current:** `padding="same"` on all MaxPooling2D layers.

**Paper:** Uses valid padding (default) — Section 4.1 says convolutions use *"the valid convolutional operation"*.

**Impact:** With 224x224 input (vs paper's 105x105), valid padding may reduce spatial dimensions differently. Worth testing both to see which produces better feature maps for bat face recognition.

---

## 3. Add BatchNormalization (Beyond the Paper)

The paper predates widespread BatchNorm adoption. Adding BatchNorm after each Conv2D could stabilize training and allow higher learning rates.

---

## 4. Add Dropout Before Embedding

Adding `Dropout(0.5)` before the Dense(4096) layer could reduce overfitting, especially given the large parameter count.

---

## 5. Online Data Augmentation

The paper uses affine distortions (8x augmentation). Adding online augmentation during training (random flip, crop, color jitter) could improve generalization without pre-generating augmented images.
