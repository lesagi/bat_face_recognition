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
        "path": "data/processed/mauritius/video/not_augmented/original_bg/"
        "m--20230902_052035--20230902_052035.1408.jpg",
    },
    "rousettus": {
        "manifest": "data/manifests/rousettus_original_bg_manifest.csv",
        "path": "data/processed/rousettus/video/not_augmented/original_bg/"
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


def find_run_dir(species: str, head_token: str) -> Path | None:
    runs = sorted(Path("outputs/runs").glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in runs:
        if list(d.glob(f"roc_curve__{species}_video_original_*{head_token}*.png")) and resolve_ckpt(
            d
        ):
            return d
    return None


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
    model, x: torch.Tensor, steps: int, target: str, n: int = 12, noise: float = 0.15
) -> np.ndarray:
    """SmoothGrad-IG (Smilkov 2017): average the 20-step IG attribution over n
    noisy copies of the input. Denoises deep-net gradients -> smooth maps."""
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
    for _ in range(n):
        xn = x + torch.randn_like(x) * sigma
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


def gradcam_ckpt(species: str, model: str, edge: int) -> Path | None:
    """Deterministic checkpoint dir from the resolution matrix (original bg,
    seed 42): outputs/res_runs/<sp>_<model>_original_e<edge>_s42/. The 112px
    originals are trained into the same namespace by a small extra pass so all
    three columns share one data lineage."""
    d = Path(f"outputs/res_runs/{species}_{model}_original_e{edge}_s42")
    return resolve_ckpt(d) if d.is_dir() else None


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
        src = Path(f"data/processed/{sp}/video/not_augmented/320/original_bg/{fname}")
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
    ap.add_argument("--mode", choices=["compare", "preview", "gradcam_res"], default="compare")
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
