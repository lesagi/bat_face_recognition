# Permutation Test Performance Fix

## How We Discovered the Problem

We launched a **permutation test** -- a statistical method to check if our bat face recognition model is genuinely learning, or just getting lucky. The test was supposed to run 100 iterations, each training a small model from scratch. We kicked it off with `nohup` (a way to run long tasks in the background) and checked back later.

When we looked at the log file (`nohup.out`), we saw something alarming:

- The process had been running for **over 30 hours**
- The log file hadn't been updated since the first 2 minutes
- It was still stuck on **Permutation 1 out of 100**
- The process was eating **127 GB of RAM** and 100% CPU

Something was clearly wrong. But *what*, exactly?

---

## The Three Problems We Found

### Problem 1: Invisible Output (stdout Buffering)

**What happened:** The log file stopped updating after 2 minutes, even though the program was still running. It looked like the program was frozen.

**Why it happened:** When you run a Python program normally in a terminal, every `print()` statement shows up immediately. But when you redirect output to a file (like with `nohup`), Python switches to something called **"full buffering"**. Instead of writing each line to the file immediately, Python collects output in a memory buffer and only writes it to the file when the buffer is full (or the program exits).

Think of it like writing a letter: instead of mailing each sentence as you write it, you wait until you've filled an entire page before putting it in the mailbox. If your program is slow and doesn't produce much output, that "page" might never fill up, and the file stays empty for hours.

**The fix:** We added `flush=True` to every `print()` statement. This is like saying "mail this sentence immediately, don't wait for a full page." We also started running Python with the `-u` flag (`python -u`), which disables buffering globally.

```python
# Before (buffered -- output may not appear for hours)
print(f"Epoch {epoch}/{total}")

# After (flushed -- output appears immediately)
print(f"Epoch {epoch}/{total}", flush=True)
```

---

### Problem 2: Reading Images from Disk Over and Over

**What happened:** Training was absurdly slow -- about **3 seconds per batch** of 16 images. A single epoch took nearly 5 hours.

**Why it happened:** In machine learning, training works in **epochs** -- each epoch means going through your entire dataset once. Our dataset has ~90,000 image pairs. Every time the model needed to look at an image, it would:

1. Open the file on disk
2. Read the raw bytes
3. Decode the image format (PNG/JPEG)
4. Resize it to 224x224 pixels
5. Convert it to numbers the model can use

This happened for **every image, every batch, every epoch, every permutation**. With 10 epochs per permutation and 100 permutations, that's the same images being read from disk **1,000 times each**. Disk I/O (reading files from a hard drive) is one of the slowest operations a computer can do.

**The fix:** We added `.cache()` to the data pipeline. This is a built-in TensorFlow feature that says "after you read an image the first time, store it in RAM so you never have to read it from disk again." Since our server has 754 GB of RAM, this was a no-brainer.

```python
# Before (reads from disk every time)
self.train_batches = (
    train_data
    .shuffle(buffer_size=shuffle_buffer)
    .batch(self.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# After (reads from disk once, then serves from RAM)
train_data = train_data.cache()  # <-- this one line makes epochs 2-10 dramatically faster
self.train_batches = (
    train_data
    .shuffle(buffer_size=shuffle_buffer)
    .batch(self.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
```

The first epoch is still slow (it has to read everything once), but epochs 2 through 10 now run from memory and are 10-50x faster.

---

### Problem 3: Tiny Batches and Too Much Evaluation

This was actually three smaller issues combined:

#### a) Batch size was too small

**What happened:** The model was processing 16 images at a time (batch size = 16), resulting in 5,606 batches per epoch.

**Why it happened:** A GPU is like a factory with 1,000 workstations. If you only give it 16 items to work on at a time, 984 workstations sit idle. Our NVIDIA RTX A5000 GPUs have 24 GB of memory each -- they can easily handle batches of 128 or more. The original batch size of 16 was inherited from the main training config and was never tuned for the permutation test.

**The fix:** We increased the batch size to 128 for the permutation test. This means 8x fewer batches per epoch (351 instead of 5,606), and the GPU stays much busier.

#### b) Gradient accumulation added unnecessary overhead

**What happened:** The code was using "gradient accumulation" with 4 steps -- a technique that processes 4 small batches in a row before updating the model.

**Why it happened:** Gradient accumulation is useful when your GPU doesn't have enough memory for large batches. But since we were already using tiny batches (16), this just added Python loop overhead without any benefit. It would have been better to just use a batch of 64 (= 16 x 4) directly.

**The fix:** We removed gradient accumulation entirely and just use the larger batch size of 128. Simpler code, faster execution.

#### c) Evaluating the model after every epoch (unnecessarily)

**What happened:** After every training epoch, the code ran a full evaluation pass over the entire test set (~60,000 pairs). With 10 epochs per permutation, that's 11 evaluation passes (10 per-epoch + 1 final). Each evaluation pass called `model.predict()` thousands of times.

**Why it happened:** In normal model training, you want to track how your model improves epoch by epoch. But in a permutation test, you only care about the **final** performance -- you just need one number at the end to build your null distribution. Intermediate evaluations are pure waste.

**The fix:** We added an `eval_last_only` option. When enabled, the model only evaluates once at the very end (after all 10 epochs), skipping 10 unnecessary evaluation passes. We also replaced `model.predict()` (which has high per-call overhead) with a direct `@tf.function` model call, which is much faster.

---

## The Combined Impact

| What Changed | Before | After | Speedup |
|---|---|---|---|
| Batch size | 16 (5,606 batches/epoch) | 128 (351 batches/epoch) | ~16x fewer batches |
| Dataset size | 89,700 pairs (full) | ~44,850 pairs (50% sample) | 2x less data |
| Disk I/O | Read from disk every epoch | Read once, cached in RAM | ~10x for epochs 2+ |
| Gradient accumulation | 4-step accumulation loop | Direct training step | Simpler, faster |
| Evaluation passes | 11 per permutation | 1 per permutation | ~11x less eval work |
| Output visibility | Buffered (invisible for hours) | Flushed (real-time) | Debuggable |

**Estimated total runtime:**
- Before: **~200 days** (yes, really)
- After: **~4-6 days**

That's roughly a **30-50x speedup** from relatively simple changes -- no algorithmic breakthroughs, just removing waste.

---

## Key Takeaways

1. **Always check your logs are actually being written.** When running long jobs with `nohup` or in the background, use `flush=True` or `python -u` so you can actually monitor progress.

2. **Cache your data when possible.** If you're reading the same images multiple times (across epochs), `.cache()` is almost always worth it. RAM is fast; disks are slow.

3. **Tune your batch size for your hardware.** A batch size of 16 on a 24 GB GPU is like driving a truck to deliver a single envelope. Check your GPU utilization and increase batch size until the GPU is actually busy.

4. **Don't do unnecessary work.** If you only need the final metric, don't evaluate after every epoch. This seems obvious, but it's easy to copy-paste from a training script where per-epoch evaluation makes sense.

5. **Profile before optimizing.** The logging fix was the first thing we did -- without it, we couldn't even tell what was happening. Always make the problem observable before trying to fix it.
