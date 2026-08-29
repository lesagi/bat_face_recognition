"""Report assets: Integrated-Gradients saliency maps + (later) results table/figure.

Integrated Gradients (Sundararajan 2017, 20 steps) for each model family on the
SAME bat image, rendered as overlay + on-black.

- Siamese  → the packaged `SiameseSaliencyAdapter(method="integrated_gradients")`
             (IG of the pair similarity vs a seeded random counterpart).
- ArcFace/AdaFace → embedding IG. Target selectable:
    * "class_logit" (default): cosine to the top predicted identity
      (`emb · normalize(head.weight)[argmax]`) — class-discriminative.
    * "emb_mag": `||forward_embedding(x)||^2` — diffuse, magnitude-driven.
  Input is ImageNet-normalized (on-distribution); baseline = normalized black;
  L2-aggregate channels, Gaussian-smooth, min-max normalize.

`--mode preview` renders all three families for the chosen bat, per species, for
human review before the full report is built.

    python scripts/gen_report_assets.py --mode preview --target class_logit --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import nn

from bat_cli.runtime import build_model, compose_config
from bat_data import manifest_from_csv
from bat_data.dataset import IMAGENET_MEAN, IMAGENET_STD, default_image_loader

HOT = matplotlib.colormaps["hot"]
DISPLAY = 224  # render size for the grid

BATS = {
    "mauritius": {
        "manifest": "data/manifests/mauritius_original_bg_manifest.csv",
        "path": "data/processed/mauritius/video/not_augmented/base/original_bg/"
        "m--20230902_052035--20230902_052035.1408.jpg",
    },
    "rousettus": {
        "manifest": "data/manifests/rousettus_original_bg_manifest.csv",
        "path": "data/processed/rousettus/video/not_augmented/base/original_bg/"
        "r--arrowhead--VID_20250909_144129.100.jpg",
    },
}
# (display name, curve-PNG head token, family, experiment template, input edge)
SETTINGS = [
    ("Siamese", "binary-focal", "pair", "siamese_{sp}_original_bg_video", 105),
    ("ArcFace", "arcface", "embedding", "arcface_{sp}_original_bg_video_tuned", 112),
    ("AdaFace", "adaface", "embedding", "adaface_{sp}_original_bg_video_tuned", 112),
]


def resolve_ckpt(d: Path) -> Path | None:
    """Best available checkpoint. Embedding runs save best_model_roc_auc.pt;
    the pair (Siamese) trainer saves f1/loss/recall/precision instead."""
    for name in ("best_model_roc_auc.pt", "best_model_f1.pt", "best_model_loss.pt"):
        if (d / name).exists():
            return d / name
    rest = sorted(d.glob("best_model_*.pt"))
    return rest[0] if rest else None


def find_run_dir(species: str, head_token: str, *, require_ckpt: bool = True) -> Path | None:
    base = Path("outputs/runs")
    # Support both the legacy flat layout (outputs/runs/<run>/) and the current
    # day-nested layout (outputs/runs/<YYYY-MM-DD>/<run>/). Day dirs themselves hold
    # no roc_curve PNGs, so they are harmlessly skipped by the match below.
    # require_ckpt=False finds runs kept under --keep-checkpoints none (which still
    # carry their report/figures/explanations, just no best_model_*.pt).
    candidates = list(base.glob("*/")) + list(base.glob("*/*/"))
    runs = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for d in runs:
        if list(d.glob(f"roc_curve__{species}_video_original_*{head_token}*.png")) and (
            not require_ckpt or resolve_ckpt(d)
        ):
            return d
    return None


def render_projection(out: Path) -> None:
    """Copy per-run UMAP/t-SNE embedding projections into the report figures dir.

    Checkpoint-free: reuses the projection PNGs each ArcFace run already writes to
    ``<run>/explanations/embedding_projection_{umap,tsne}.png`` when trained with
    explanations enabled. No model reload, so this works under
    ``--keep-checkpoints none``.
    """
    import shutil

    for sp in BATS:
        rd = find_run_dir(sp, "arcface", require_ckpt=False)
        if rd is None:
            print(f"  !! no ArcFace run dir for {sp} — skipping projection")
            continue
        found = False
        for method in ("umap", "tsne"):
            src = rd / "explanations" / f"embedding_projection_{method}.png"
            if src.exists():
                dst = out / f"proj_{sp}_{method}.png"
                shutil.copyfile(src, dst)
                print(f"  ok {sp}/{method}: {rd.name} -> {dst.name}")
                found = True
            else:
                print(f"  !! missing {src}")
        if not found:
            print(f"  !! {sp}: {rd.name} has no projection PNGs (train with explanations enabled)")


def load_model(experiment: str, ckpt: Path, device: str, family: str):
    cfg = compose_config(experiment=experiment)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    num_classes = int(state["model"]["head.weight"].shape[0]) if family == "embedding" else 2
    model = build_model(cfg, num_classes=num_classes)
    model.load_state_dict(state["model"])
    if isinstance(state, dict) and state.get("ema"):
        try:
            from bat_training import ExponentialMovingAverage

            ema = ExponentialMovingAverage(model, decay=0.999)
            ema.load_state_dict(state["ema"])
            ema.apply_to(model)
        except Exception as exc:  # noqa: BLE001 - EMA is best-effort
            print(f"    (EMA not applied: {exc})")
    return model.to(device).eval()


def _smooth(sal: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return sal
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(sal, sigma=sigma).astype(np.float32)
    except ImportError:
        return sal


def _finish(sal_t: torch.Tensor, sigma: float) -> np.ndarray:
    """(3,H,W) IG attribution -> (H,W) [0,1] via L2, smooth, min-max."""
    sal = sal_t.pow(2).sum(0).sqrt().detach().cpu().numpy().astype(np.float32)
    sal = _smooth(sal, sigma)
    lo, hi = float(sal.min()), float(sal.max())
    return (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)


def embedding_ig(model, x: torch.Tensor, steps: int, target: str, sigma: float = 1.0) -> np.ndarray:
    """20-step IG for an embedding model. target: 'class_logit' or 'emb_mag'."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    baseline = (torch.zeros_like(x) - mean) / std

    cls = None
    if target == "class_logit":
        with torch.no_grad():
            w = F.normalize(model.head.weight, dim=1)  # (C, D)
            emb = model.forward_embedding(x)  # (1, D) normalized
            cls = int((emb @ w.t()).argmax(1).item())

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    grad_sum = torch.zeros_like(x)
    for a in alphas:
        interp = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        emb = model.forward_embedding(interp)
        if target == "class_logit":
            w = F.normalize(model.head.weight, dim=1)
            scalar = emb @ w[cls]  # cosine to the top class
        else:
            scalar = emb.pow(2).sum()
        grad_sum = grad_sum + torch.autograd.grad(scalar, interp)[0]
    ig = (grad_sum / float(len(alphas))) * (x - baseline)
    return _finish(ig[0], sigma)


def embedding_gradcam(model, x: torch.Tensor, target: str) -> np.ndarray:
    """Grad-CAM on the last conv layer, weighting activations by the gradient of
    the top-class cosine logit (or ||emb||^2). Smooth, localized CNN attention.
    Returns a (H,W) [0,1] map (upsampled from the conv grid)."""
    convs = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
    target_layer = convs[-1]
    store: dict[str, torch.Tensor] = {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        # Target the PRE-normalization embedding: forward_embedding returns an
        # L2-normalized vector, so ||emb||^2 == 1 (constant -> zero gradient).
        # The projection output before normalize is what varies spatially.
        emb_raw = model.projection(model.backbone(x))
        if target == "class_logit":
            w = F.normalize(model.head.weight, dim=1)
            cls = int((F.normalize(emb_raw, dim=1) @ w.t()).argmax(1).item())
            scalar = F.normalize(emb_raw, dim=1) @ w[cls]
        else:
            scalar = emb_raw.pow(2).sum()
        scalar.backward()
        act = store["a"][0]  # (C,h,w)
        grad = store["g"][0]  # (C,h,w)
    finally:
        h1.remove()
        h2.remove()
    weights = grad.mean(dim=(1, 2))  # (C,)
    cam = torch.relu((weights[:, None, None] * act).sum(0))  # (h,w)
    cam = cam.detach().cpu().numpy().astype(np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def embedding_smoothgrad_ig(
    model,
    x: torch.Tensor,
    steps: int,
    target: str,
    n: int = 12,
    noise: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    """SmoothGrad-IG (Smilkov 2017): average the 20-step IG attribution over n
    noisy copies of the input. Denoises deep-net gradients -> smooth maps.

    ``seed`` makes the noise draw reproducible. Without it two runs of the same
    command on the same data differ by ~6 percentage points on the pointing game
    (measured: 76.6% vs 82.8%), which is a third of the effect the species
    analysis is trying to detect. ``siamese_ig`` below already seeded its
    counterpart; this one did not. Seeding is per image at the call site, so the
    noise is reproducible without being correlated across images. A local
    Generator is used rather than ``torch.manual_seed`` so global RNG state --
    and therefore anything else in the process -- is left alone.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    baseline = (torch.zeros_like(x) - mean) / std
    sigma = noise * float((x.max() - x.min()).item())
    cls = None
    if target == "class_logit":
        with torch.no_grad():
            w = F.normalize(model.head.weight, dim=1)
            cls = int((model.forward_embedding(x) @ w.t()).argmax(1).item())
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    accum = torch.zeros_like(x)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
    for _ in range(n):
        noise_draw = (
            torch.randn_like(x)
            if gen is None
            else torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        )
        xn = x + noise_draw * sigma
        grad_sum = torch.zeros_like(x)
        for a in alphas:
            interp = (baseline + a * (xn - baseline)).detach().requires_grad_(True)
            emb = model.forward_embedding(interp)
            if target == "class_logit":
                w = F.normalize(model.head.weight, dim=1)
                scalar = emb @ w[cls]
            else:
                scalar = emb.pow(2).sum()
            grad_sum = grad_sum + torch.autograd.grad(scalar, interp)[0]
        accum = accum + (grad_sum / float(len(alphas))) * (xn - baseline)
    return _finish((accum / float(n))[0], sigma=1.0)


def siamese_ig(model, record, steps: int, edge: int) -> np.ndarray:
    from bat_interpretability.siamese_saliency import SiameseSaliencyAdapter

    torch.manual_seed(0)  # reproducible random counterpart
    adapter = SiameseSaliencyAdapter(
        method="integrated_gradients", integration_steps=steps, input_size=edge
    )
    return np.asarray(adapter.explain(model, [record])[0].saliency, dtype=np.float32)


def resize_map(sal: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray((np.clip(sal, 0, 1) * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    return np.asarray(im).astype(np.float32) / 255.0


def overlay(orig_rgb: np.ndarray, sal: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat = (HOT(sal)[..., :3] * 255).astype(np.uint8)
    return (orig_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(
        np.uint8
    )


def on_black(sal: np.ndarray) -> np.ndarray:
    return (HOT(sal)[..., :3] * 255).astype(np.uint8)


EDGES_CMP = [112, 224, 320]  # ResNet50 /32 → 4×4, 7×7, 10×10 conv grids
GRID = {112: "4×4", 224: "7×7", 320: "10×10"}


def _res_run_dir(species: str, model_token: str, edge: int) -> Path | None:
    """Locate a resolution-study run dir (original bg, seed 42) by species/model/edge.

    Tolerant of both the legacy flat layout (outputs/res_runs/<leaf>/) and a day-nested
    layout (outputs/res_runs/<YYYY-MM-DD>/<leaf>/), and of both leaf schemes: the concise
    ``<sp>_<model>_original_e<edge>_s42`` and the experiment_name
    ``<sp>_video_original_<model>_..._e<edge>_s42``. Returns the most recent match."""
    base = Path("outputs/res_runs")
    pat = f"*{species}*{model_token}*_e{edge}_s42"
    matches = [p for p in (list(base.glob(pat)) + list(base.glob(f"*/{pat}"))) if p.is_dir()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def gradcam_ckpt(species: str, model: str, edge: int) -> Path | None:
    """Deterministic checkpoint dir from the resolution matrix (original bg, seed 42).
    The 112px originals are trained into the same namespace by a small extra pass so all
    three columns share one data lineage."""
    d = _res_run_dir(species, model, edge)
    return resolve_ckpt(d) if d else None


def render_gradcam_resolution(device: str, out: Path, target: str) -> None:
    """One fixed bat per species, GradCAM at 112 / 224 / 320 for ArcFace & AdaFace.

    The same 320px crop is re-loaded at each edge (identical source pixels), so
    the only variable is the conv-grid resolution — the point of the figure.
    Rows = {ArcFace, AdaFace} × {overlay, on-black}; cols = 112 | 224 | 320.
    """
    import matplotlib.pyplot as plt

    models = [("ArcFace", "arcface"), ("AdaFace", "adaface")]
    for sp, spec in BATS.items():
        fname = Path(spec["path"]).name
        src = Path(f"data/processed/{sp}/video/not_augmented/base_320/original_bg/{fname}")
        if not src.exists():  # fall back to the original committed crop
            src = Path(spec["path"])
        orig = np.asarray(Image.open(src).convert("RGB").resize((DISPLAY, DISPLAY)))
        rows: list[tuple[str, list[np.ndarray], list[np.ndarray]]] = []
        for mname, mkey in models:
            exp = f"{mkey}_{sp}_original_bg_video_tuned"
            ov_cells, ob_cells = [], []
            for edge in EDGES_CMP:
                ckpt = gradcam_ckpt(sp, mkey, edge)
                if ckpt is None:
                    print(f"  !! missing ckpt {sp}/{mkey}/e{edge}")
                    ov_cells.append(np.zeros_like(orig))
                    ob_cells.append(np.zeros_like(orig))
                    continue
                model = load_model(exp, ckpt, device, "embedding")
                x = default_image_loader(str(src), edge, "imagenet").unsqueeze(0).to(device)
                cam = embedding_gradcam(model, x, target)  # raw conv grid (edge/32)
                sal = resize_map(cam, DISPLAY)
                ov_cells.append(overlay(orig, sal))
                ob_cells.append(on_black(sal))
            rows.append((f"{mname}\noverlay", ov_cells, ob_cells))

        # 4 rows (2 models × overlay/on-black) × 3 cols (edges)
        fig, axes = plt.subplots(4, 3, figsize=(9, 12))
        row_labels = [
            f"{models[0][0]}\noverlay",
            f"{models[0][0]}\non black",
            f"{models[1][0]}\noverlay",
            f"{models[1][0]}\non black",
        ]
        cells = [rows[0][1], rows[0][2], rows[1][1], rows[1][2]]
        for r in range(4):
            for c, edge in enumerate(EDGES_CMP):
                axes[r, c].imshow(cells[r][c])
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(f"{edge}px  (Grad-CAM {GRID[edge]})", fontsize=11)
            axes[r, 0].set_ylabel(row_labels[r], fontsize=10, rotation=90, labelpad=10)
        fig.suptitle(
            f"{sp} — Grad-CAM vs input resolution (target={target})", fontsize=13
        )
        fig.tight_layout()
        dest = out / f"gradcam_resolution_{sp}_{target}.png"
        fig.savefig(dest, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {dest}")


def get_record(manifest_path: str, img_path: str):
    manifest = manifest_from_csv(manifest_path)
    want = Path(img_path)
    for r in manifest.records:
        if Path(r.path) == want or Path(r.path).name == want.name:
            return r
    raise SystemExit(f"record not found in {manifest_path}: {img_path}")


# --- gallery mode: many bats × methods × targets × sizes ---------------------
GALLERY_MODELS = [("ArcFace", "arcface"), ("AdaFace", "adaface")]
# (row label, method, target)
GALLERY_METHODS = [
    ("IG·class", "ig", "class_logit"),
    ("IG·emb", "ig", "emb_mag"),
    ("GradCAM·class", "gradcam", "class_logit"),
    ("GradCAM·emb", "gradcam", "emb_mag"),
]
SIAMESE_EDGE = 105  # 4-conv net is size-locked; runs only at its native small size


def siamese_ckpt(species: str) -> Path | None:
    d = _res_run_dir(species, "siamese", SIAMESE_EDGE)
    return resolve_ckpt(d) if d else None


def src_for_edge(species: str, filename: str, edge: int) -> Path:
    """Genuine source crop for a given edge: the 320 build for 320, else 224
    (the loader downsizes 224→112; 320 must come from the 320 build, not an
    upscale of 224)."""
    d = 320 if edge == 320 else 224
    return Path(f"data/processed/{species}/video/not_augmented/{d}/original_bg/{filename}")


def pick_gallery_images(species: str, n_bats: int, per_bat: int, seed: int):
    """Seeded pick of n_bats individuals × per_bat images from the 320 build."""
    import random

    root = Path(f"data/processed/{species}/video/not_augmented/base_320/original_bg")
    by_id: dict[str, list[Path]] = {}
    for f in sorted(root.glob("*.jpg")):
        parts = f.name.split("--")
        if len(parts) >= 3:
            by_id.setdefault(parts[1], []).append(f)
    rng = random.Random(f"{species}-{seed}")
    bats = sorted(by_id)
    chosen = sorted(rng.sample(bats, min(n_bats, len(bats))))
    picks: list[tuple[str, Path]] = []
    for b in chosen:
        imgs = sorted(by_id[b])
        for p in sorted(rng.sample(imgs, min(per_bat, len(imgs)))):
            picks.append((b, p))
    return chosen, picks


def render_gallery(device: str, out: Path, steps: int, n_bats: int, per_bat: int, seed: int) -> None:
    """For 10 individuals × 2 images per species: IG & GradCAM × {class_logit,
    emb_mag} × {112,224,320} for ArcFace/AdaFace + Siamese IG@105, overlay +
    on-black. One 9×6 figure per bat-image. Each of the 14 models is loaded once."""
    import matplotlib.pyplot as plt

    gdir = out.parent / "saliency_gallery"
    row_labels = [f"{mn}·{ml}" for mn, _ in GALLERY_MODELS for ml, _, _ in GALLERY_METHODS]
    row_labels.append("Siamese·IG")
    col_specs = [(e, r) for e in EDGES_CMP for r in ("ov", "ob")]  # 6 cols
    col_titles = [f"{e}px {'overlay' if r == 'ov' else 'on-black'}" for e, r in col_specs]

    for sp in ("mauritius", "rousettus"):
        chosen, picks = pick_gallery_images(sp, n_bats, per_bat, seed)
        print(f"[{sp}] {len(chosen)} bats: {', '.join(chosen)}")
        manifest = f"data/manifests/{sp}_original_aligned_224_manifest.csv"
        # results[pathname][row_label][edge] = (overlay, on_black)
        res: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {
            p.name: {rl: {} for rl in row_labels} for _, p in picks
        }
        orig320 = {
            p.name: np.asarray(Image.open(p).convert("RGB").resize((DISPLAY, DISPLAY)))
            for _, p in picks
        }
        # embedding models: load each (model,edge) once, sweep all images
        for mname, mkey in GALLERY_MODELS:
            exp = f"{mkey}_{sp}_original_bg_video_tuned"
            for edge in EDGES_CMP:
                ck = gradcam_ckpt(sp, mkey, edge)
                if ck is None:
                    print(f"  !! missing {sp}/{mkey}/e{edge}")
                    continue
                model = load_model(exp, ck, device, "embedding")
                for _, p in picks:
                    src = src_for_edge(sp, p.name, edge)
                    x = default_image_loader(str(src), edge, "imagenet").unsqueeze(0).to(device)
                    orig = orig320[p.name]
                    for ml, method, target in GALLERY_METHODS:
                        sal = (
                            embedding_ig(model, x, steps, target)
                            if method == "ig"
                            else embedding_gradcam(model, x, target)
                        )
                        sal = resize_map(sal, DISPLAY)
                        res[p.name][f"{mname}·{ml}"][edge] = (overlay(orig, sal), on_black(sal))
        # Siamese IG @105 (native), stored under the 112 column
        sck = siamese_ckpt(sp)
        if sck is not None:
            smodel = load_model(f"siamese_{sp}_original_bg_video", sck, device, "pair")
            for _, p in picks:
                sal = resize_map(siamese_ig(smodel, get_record(manifest, str(p)), steps, SIAMESE_EDGE), DISPLAY)
                res[p.name]["Siamese·IG"][112] = (overlay(orig320[p.name], sal), on_black(sal))
        else:
            print(f"  !! missing Siamese checkpoint for {sp}")

        # one figure per bat-image
        (gdir / sp).mkdir(parents=True, exist_ok=True)
        for bat, p in picks:
            cells = res[p.name]
            fig, axes = plt.subplots(len(row_labels), len(col_specs), figsize=(13, 20))
            for r, rl in enumerate(row_labels):
                for c, (edge, kind) in enumerate(col_specs):
                    ax = axes[r, c]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pair = cells[rl].get(edge)
                    if pair is None:
                        ax.axis("off")
                    else:
                        ax.imshow(pair[0] if kind == "ov" else pair[1])
                    if r == 0:
                        ax.set_title(col_titles[c], fontsize=9)
                axes[r, 0].set_ylabel(rl, fontsize=9, rotation=90, labelpad=8)
            fig.suptitle(f"{sp} · {bat} · {p.stem}", fontsize=13, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            dest = gdir / sp / f"{p.stem}.png"
            fig.savefig(dest, dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"  wrote {len(picks)} figures → {gdir / sp}/")


def _embed_saliency(model, x, method: str, steps: int) -> np.ndarray:
    if method == "gradcam":
        return embedding_gradcam(model, x, "emb_mag")
    if method == "smoothgrad":
        return embedding_smoothgrad_ig(model, x, steps, "class_logit")
    if method == "ig_class":
        return embedding_ig(model, x, steps, "class_logit")
    return embedding_ig(model, x, steps, "emb_mag")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["compare", "preview", "gradcam_res", "gallery", "projection"],
        default="compare",
    )
    ap.add_argument("--n-bats", type=int, default=10, help="Gallery: individuals per species.")
    ap.add_argument("--per-bat", type=int, default=2, help="Gallery: images per individual.")
    ap.add_argument("--pick-seed", type=int, default=0, help="Gallery: seed for bat/image pick.")
    ap.add_argument(
        "--target",
        choices=["emb_mag", "class_logit"],
        default="emb_mag",
        help="Grad-CAM target scalar for --mode gradcam_res.",
    )
    ap.add_argument(
        "--embed-method",
        choices=["gradcam", "smoothgrad", "ig_class", "ig_emb"],
        default="smoothgrad",
        help="Embedding saliency for --mode preview (single method).",
    )
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "gradcam_res":
        render_gradcam_resolution(args.device, out, args.target)
        return

    if args.mode == "gallery":
        render_gallery(args.device, out, args.steps, args.n_bats, args.per_bat, args.pick_seed)
        return

    if args.mode == "projection":
        render_projection(out)
        return

    import matplotlib.pyplot as plt

    # In compare mode, show every candidate embedding method next to the
    # reference Siamese IG so the user can pick one for the report.
    embed_methods = (
        [("SmoothGrad-IG", "smoothgrad"), ("GradCAM", "gradcam")]
        if args.mode == "compare"
        else [(args.embed_method, args.embed_method)]
    )

    for sp, spec in BATS.items():
        orig = np.asarray(Image.open(spec["path"]).convert("RGB").resize((DISPLAY, DISPLAY)))
        rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        for name, head_token, family, exp_tmpl, edge in SETTINGS:
            rd = find_run_dir(sp, head_token)
            if rd is None:
                print(f"  !! no run dir for {sp}/{name}")
                continue
            ckpt = resolve_ckpt(rd)
            print(f"  {sp}/{name}: {rd.name} ({ckpt.name})")
            model = load_model(exp_tmpl.format(sp=sp), ckpt, args.device, family)
            if family == "pair":
                sal = siamese_ig(
                    model, get_record(spec["manifest"], spec["path"]), args.steps, edge
                )
                sal = resize_map(sal, DISPLAY)
                rows.append((f"{name}\nIG · pair-sim", overlay(orig, sal), on_black(sal)))
            else:
                x = (
                    default_image_loader(spec["path"], edge, "imagenet")
                    .unsqueeze(0)
                    .to(args.device)
                )
                for label, m in embed_methods:
                    sal = resize_map(_embed_saliency(model, x, m, args.steps), DISPLAY)
                    rows.append((f"{name}\n{label}", overlay(orig, sal), on_black(sal)))

        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
        if len(rows) == 1:
            axes = axes[None, :]
        for r, (label, ov, ob) in enumerate(rows):
            for c, (img, title) in enumerate([(orig, "input"), (ov, "overlay"), (ob, "on black")]):
                axes[r, c].imshow(img)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(title, fontsize=11)
            axes[r, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=10)
        fig.suptitle(f"{sp} — saliency method comparison", fontsize=12)
        fig.tight_layout()
        dest = out / f"ig_compare_{sp}.png"
        fig.savefig(dest, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
