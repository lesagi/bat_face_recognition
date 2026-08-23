# Siamese saliency regression: why the current maps look worse than the old ones

**Date:** 2026-07-06
**Author:** investigation note (read-only; no code changed)
**Scope:** why Siamese (`family == "pair"`) saliency maps from the current
PyTorch pipeline resemble the bat's face *less* than the older maps.

---

## TL;DR

The two sets of maps were produced by **two different attribution methods**:

| | "Previous" (looks good) | "Current" (looks worse) |
|---|---|---|
| Produced by | Legacy **TensorFlow** stack | PyTorch `bat_interpretability` |
| Example artifact | `models/siamese/mauritius/20260412_..._5c44941a/saliency/*.png` | `mlruns/.../post_training_report.pdf` p.7 (saliency composite) |
| Attribution method | **Integrated Gradients** (20 steps) | **Vanilla single-pass gradient** |
| Title on the image | "Saliency Map (Integrated Gradients)" | "Saliency overlays" (no method shown) |

Everything downstream of the gradient (channel aggregation = L2 magnitude,
Gaussian σ=1 smoothing, min–max normalisation, input resolution 105², [0,1]
pixel scaling) is **the same** in both stacks. The counterpart is a random
image in both. So the visual gap is almost entirely explained by the
**method downgrade from Integrated Gradients → vanilla gradients** that
happened silently during the TF→PyTorch port.

**Does it make sense?** Yes — and it's mechanistically expected, not a
mystery. See [§4](#4-why-this-makes-sense-the-mechanism). It is a
*behavioural regression*, not a broken pipeline: the port kept an
`integrated_gradients` option but defaults to `"vanilla"`, and nothing in the
dispatcher/CLI/configs ever selects it.

---

## 1. Provenance of the two artifacts

### "Previous" (the good-looking maps)

The example you were looking at —
`models/siamese/mauritius/20260412_siamese_mauritius_video_no_aug_random_bg_5c44941a/saliency/20230807_055421_m--20230807_055421--155_saliency.png` —
was written by the **legacy TF** code, not the current pipeline. Evidence:

- That run directory also contains `evaluation_*.csv` files. CSV output is a
  legacy-TF hallmark (the PyTorch stack renders to the PDF and emits no CSVs —
  see `CLAUDE.md`, "No CSV output from new code").
- The filenames (`<class>_<filename-stem>_saliency.png`) match exactly the TF
  method `SiameseModelSaliencyMapCreator.generate_per_bat_saliency_images`
  (recover with `git show refactor-foundation-pytorch:app/visualization/saliency.py`).
- The image title literally reads **"Saliency Map (Integrated Gradients)"**.

That TF method's signature default is `method="integrated_gradients"`.

### "Current" (the worse-looking maps)

Produced by the PyTorch pipeline: `bat_cli` → `bat_interpretability` →
`bat_reporting`. The saliency page is a 5×2 composite (`original | overlay`)
embedded on p.7 of `post_training_report.pdf`. Example run inspected:
`mlruns/550014979119175420/42e44078b92642998bffdcbe6199b91d`
(`siamese_rousettus_random_bg_video`, 2026-07-06).

> Note on a fair comparison: the *current mauritius* siamese runs
> (multiseed, green-bg) **skip the saliency page entirely** — mauritius has
> too few identities to fill the required 5 rows
> (`bat_reporting/sections/saliency.py:31`, `DEFAULT_ROWS = 5`). So the
> closest current example is the rousettus random-bg run. Species differs
> from the mauritius reference, but the method difference below is
> species-independent.

---

## 2. The code path that fixes the method to "vanilla"

1. `SiameseSaliencyAdapter` defaults to vanilla:
   `packages/bat_interpretability/bat_interpretability/siamese_saliency.py:67`
   ```python
   method: SaliencyMethod = "vanilla"
   ```

2. The dispatcher constructs it with **defaults only**:
   `packages/bat_interpretability/bat_interpretability/dispatcher.py:32`
   ```python
   if family == "pair":
       return SiameseSaliencyAdapter()      # method="vanilla"
   ```

3. The CLI overrides **only `input_size`**, never the method:
   `packages/bat_cli/bat_cli/runtime.py:846`
   ```python
   adapter = replace(select_adapter(model), input_size=image_size)
   ```

4. No Hydra config selects a saliency method (grep of `configs/` for
   `salien|integrated_grad|vanilla` returns nothing).

So the current pipeline **always** runs `method="vanilla"` for Siamese. The
`integrated_gradients` branch exists (`siamese_saliency.py:126`,
`_integrated_gradients` at `:164`) but is unreachable in practice.

---

## 3. What is *not* the cause (ruled out)

These were checked and are equivalent across the two stacks, so they don't
explain the gap:

- **Input resolution.** Both run at 105×105 (`SIAMESE_INPUT_EDGE_LENGTH = 105`;
  TF inferred the same from the model input shape). Displayed maps are
  upscaled afterward.
- **Pixel normalisation.** Both scale to `[0,1]` by `/255`
  (TF `preprocess_siamese_input`; current `_load_and_preprocess`,
  `siamese_saliency.py:262`). No ImageNet normalisation in either (that fix,
  commit `2608f7d`, targets the embedding ResNet backbone, not siamese_4conv).
- **Channel aggregation.** Both use L2 magnitude across channels
  (`_aggregate_channels(..., "magnitude")`, `siamese_saliency.py:244`; TF
  `process_saliency_map(method="magnitude")`).
- **Post-smoothing / normalisation.** Both apply Gaussian σ=1 then min–max to
  `[0,1]`.
- **Counterpart is random in both.** TF `create_random_image` vs current
  `_random_counterpart` (`torch.rand_like`, `siamese_saliency.py:271`).
  *Minor caveat:* TF's random counterpart is `uniform[0,256)` (i.e. ~[0,255]
  scale — it is **not** divided by 255), while the current one is
  `uniform[0,1)`. Both are "a random, maximally-dissimilar partner", so this
  changes the counterpart's embedding but not the qualitative structure of the
  anchor's gradient. It is a second-order difference at most, not the cause.

---

## 4. Why this makes sense (the mechanism)

Two independent properties of Integrated Gradients explain why the old maps
"look like the bat's face" and the vanilla maps look blobby/noisy.

### (a) IG multiplies the gradient by `(image − baseline)` → intensity gating

IG's final attribution is:

```
IG(x) = mean_over_path( ∂output/∂x ) ⊙ (image − baseline),   baseline = 0
      = mean_grad ⊙ image
```

(`_integrated_gradients`, `siamese_saliency.py:183`; TF identical,
`compute_integrated_gradients`). Because the baseline is a **black image**,
the attribution is *element-wise multiplied by the pixel values themselves*.
After the per-channel L2 magnitude, each location's saliency is scaled by how
bright that pixel is.

Consequence: **dark regions get ~zero attribution automatically, bright
regions get amplified.** In these crops the bat's illuminated face/fur is the
bright content and much of the surround is darker, so IG's map is *spatially
gated onto the face by construction*. It resembles the face partly because it
is literally weighted by the face.

Vanilla saliency is just `|∂output/∂x|` (`_vanilla_grads`,
`siamese_saliency.py:145`) with **no `(image − baseline)` term** — no
intensity gating. It responds to local sensitivity anywhere, including
background edges and texture, so it spreads out and looks blobbier and less
face-shaped.

### (b) IG averages gradients over 20 points → denoising

Vanilla saliency reads the gradient at the **single** operating point `x`.
For a trained net this point often sits in a saturated / locally-jagged region
of the function, so the raw gradient is high-frequency and noisy (the
well-known saturation problem, Sundararajan et al. 2017). IG instead averages
`∂output/∂x` over 21 interpolation points from black→image
(`steps=20`, `siamese_saliency.py:174`), which smooths out that
per-point noise before the Gaussian blur ever runs. Result: cleaner, more
connected, more "structural" heat.

Together, (a) + (b) are exactly the reasons IG was introduced over vanilla
gradients, and exactly the qualities you're describing as "resembles the
bat's face."

### (c) Secondary: rendering differs (overlay vs standalone)

The TF per-bat images show the saliency **alone** as a `hot` heatmap
(`imshow(..., cmap="hot")`, no overlay), so the attribution structure is seen
directly. The current composite **alpha-blends** the heat over the original at
`alpha=0.55` (`composite.py:30`, `_make_overlay` at `:114`,
`Image.blend(original, heat, 0.55)`). Overlaying would, if anything, make the
map look *more* face-like (you see the face underneath) — so the fact that you
perceive the current ones as *less* face-like reinforces that the underlying
attribution (the method) is the real regression, not the rendering.

---

## 5. Verdict

- The gap is **real and expected**, driven by a silent
  **Integrated Gradients → vanilla** method change during the TF→PyTorch port.
- It is a **behavioural regression in the interpretability output**, not a
  numerical bug and not a sign the model changed. The same checkpoint would
  produce the old-style maps if run with `method="integrated_gradients"`.
- The intensity-gating property (§4a) is the single biggest reason the old
  maps trace the face so cleanly; the path-averaging (§4b) is why they're
  smoother. Both are lost with vanilla.

## 6. How it could be restored (not done here — code left unchanged)

Any one of these would bring back IG-style maps; the first is the smallest:

1. Flip the dispatcher default:
   `SiameseSaliencyAdapter(method="integrated_gradients")`
   (`dispatcher.py:32`).
2. Or flip the dataclass default (`siamese_saliency.py:67`).
3. Or (cleanest, matches project style) add a Hydra knob under
   `configs/evaluation/` and thread it through
   `runtime.py:846`'s `replace(...)`.

Caveat if restored: IG is ~21× the forward/backward passes per image (20
steps), so the explanation phase gets proportionally slower. Also worth
labelling the method on the composite title so this can't silently regress
again.

---

## Appendix: how to reproduce this inspection

```bash
# Old (TF, Integrated Gradients) maps on disk:
ls models/siamese/mauritius/20260412_siamese_mauritius_video_no_aug_random_bg_5c44941a/saliency/

# Recover the TF source that made them:
git show refactor-foundation-pytorch:app/visualization/saliency.py

# Current (PyTorch, vanilla) saliency composite — extract p.7 of the report:
pdfimages -f 7 -l 7 -png \
  mlruns/550014979119175420/42e44078b92642998bffdcbe6199b91d/artifacts/reports/post_training_report.pdf \
  /tmp/cur_sal
```
