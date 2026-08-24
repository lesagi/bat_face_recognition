"""Per-species saliency: group maps, anatomical ROI statistics, pointing game.

Reviewer comment 3 asks to kernel-smooth the saliency so each species' focus is
visible. The cosmetic half is already shipped (``SiameseSaliency`` applies a
sigma=1.0 Gaussian; Grad-CAM is a 10x10 map bilinearly upsampled to 224, so
further blurring adds nothing it does not already have). This script builds the
half that supports a claim:

1. **Group map** -- mean and per-pixel SD of saliency over every test image of a
   species, in the aligned 224x224 frame. The crops are eye-aligned and
   mask-centred, so pixel coordinates are anatomically comparable across images
   and averaging is meaningful. Smoothing happens *across images*, which is the
   only smoothing that adds information rather than hiding its absence.
2. **Anatomical ROIs** from ``face_pose.pt`` (eyes + nose), giving the fraction
   of saliency mass that lands on each region per image -- a number, not a
   picture.
3. **Species comparison** of that ROI mass, aggregated per identity, because
   images of one bat come from one video and are not independent replicates.
4. **Pointing game** -- does the saliency peak land inside the eye ROI?
5. **Randomised-weights control** (Adebayo et al. 2018) -- re-run everything with
   a randomly initialised model. If the ROI concentration survives that, the maps
   are reporting image structure rather than anything the model learned.

Pose runs directly on the aligned crops here, not via ``pose_on_mask_crop``: the
aligned 224px crop is already a tight, mask-centred square, which is the
distribution ``face_pose.pt`` was trained on.

Usage:
    uv run python scripts/species_saliency_maps.py --device cuda:0
    uv run python scripts/species_saliency_maps.py --model arcface --max-per-identity 20
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from gen_report_assets import (  # noqa: E402
    embedding_gradcam,
    embedding_smoothgrad_ig,
    load_model,
    resolve_ckpt,
)

from bat_core import ImageRecord  # noqa: E402
from bat_data.dataset import default_image_loader  # noqa: E402
from bat_stats import build_filename, compare_groups  # noqa: E402

SPECIES = ("mauritius", "rousettus")
EDGE = 224

DEFAULT_OUT = Path("outputs/saliency/roi_stats.json")
DEFAULT_FIG_DIR = Path("outputs/saliency/figures")

# Everything below used to hardcode `original`. It is now parameterised, because
# arm 3C showed a model trained on original images with the face DELETED still
# reaches ROC-AUC 0.831/0.780 — so an attention analysis on that background may
# not describe a face-driven model at all. See docs/saliency_preregistration.md.
#
# `slug` is the background token as it appears in a run-directory leaf
# (`{species}_{source}_{background}_{model}_{loss}_e{edge}_s{seed}`), slugified by
# bat_stats.naming. It is what pins the checkpoint: `_green_` does NOT match
# `_green-dilated_`, which is exactly the collision that made the published
# command unreproducible once Phase 3 added dilated runs.
#
# Must be the manifest the checkpoint was TRAINED on, or the model sees a
# different preprocessing than it learned.
BACKGROUND_SPECS: dict[str, dict[str, str]] = {
    "original": {
        "slug": "original",
        "experiment": "{model}_{sp}_original_bg_video_tuned",
        "manifest": "data/manifests/{species}_original_bg_manifest.csv",
        "aligned": "data/manifests/{species}_original_aligned_{edge}_manifest.csv",
    },
    "green": {
        "slug": "green",
        "experiment": "{model}_{sp}_green_bg_video_tuned",
        "manifest": "data/manifests/{species}_green_bg_manifest.csv",
        "aligned": "data/manifests/{species}_green_aligned_{edge}_manifest.csv",
    },
}

MODEL_KEYS = ("arcface", "adaface")

POSE_WEIGHTS = "models/preprocessing/face_pose.pt"

# ROI radii as a fraction of the crop edge. Eyes and nose are small features on a
# 224px crop; 0.16 gives a ~36px disc, comparable to the feature itself.
EYE_RADIUS_FRAC = 0.16
NOSE_RADIUS_FRAC = 0.16

# Smoothing of the *group* map, in pixels of the aligned frame. Reported in every
# caption: a heatmap without its kernel width is not reproducible.
GROUP_SIGMA_PX = 6.0

# Integration steps for the IG path. 20 is the value the report assets already
# use; SmoothGrad averages 12 noisy copies on top of that.
IG_STEPS = 20

SPECIES_COLOURS = {"mauritius": "#A2582C", "rousettus": "#2F6B7A"}


# ---------------------------------------------------------------------------
# ROI construction
# ---------------------------------------------------------------------------


def disc_mask(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    return ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) <= radius**2


def rois_from_keypoints(
    keypoints: np.ndarray, shape: tuple[int, int], min_conf: float = 0.25
) -> dict[str, np.ndarray] | None:
    """Eye / nose / periphery masks from pose keypoints.

    ``face_pose.pt`` emits (left eye, right eye, nose) as (x, y, conf) rows.
    Returns ``None`` when the keypoints are too weak to trust — better to drop an
    image than to place an ROI on a misdetection.
    """
    if keypoints is None or keypoints.shape[0] < 3:
        return None
    if float(np.min(keypoints[:3, 2])) < min_conf:
        return None

    eye_radius = EYE_RADIUS_FRAC * shape[1]
    nose_radius = NOSE_RADIUS_FRAC * shape[1]

    left = disc_mask(shape, (keypoints[0, 0], keypoints[0, 1]), eye_radius)
    right = disc_mask(shape, (keypoints[1, 0], keypoints[1, 1]), eye_radius)
    nose = disc_mask(shape, (keypoints[2, 0], keypoints[2, 1]), nose_radius)

    eyes = left | right
    # Nose overlapping an eye disc is assigned to the eyes, so the regions stay
    # disjoint and the mass fractions sum to <= 1.
    nose = nose & ~eyes
    periphery = ~(eyes | nose)
    return {"eyes": eyes, "nose": nose, "periphery": periphery}


def roi_mass(saliency: np.ndarray, rois: dict[str, np.ndarray]) -> dict[str, float] | None:
    """Fraction of total saliency mass inside each ROI, plus area-normalised density.

    Returns ``None`` for a **degenerate** map — one whose total is zero.
    ``embedding_gradcam`` emits an all-zero map whenever ReLU leaves no positive
    channel contribution, which happens often enough to matter. Scoring those as
    "density 0" would be wrong twice over: it is not a measurement, and averaging
    it in drags the mean toward zero (it is what made rousettus' densities sum to
    0.40 when they must sum to 1.0 by construction).
    """
    total = float(saliency.sum())
    if total <= 0:
        return None
    out: dict[str, float] = {}
    for name, mask in rois.items():
        mass = float(saliency[mask].sum()) / total
        area = float(mask.sum()) / float(mask.size)
        out[f"{name}_mass"] = mass
        # Density > 1 means the region attracts more saliency than its area would
        # give by chance — the interpretable version of the raw mass.
        out[f"{name}_density"] = mass / area if area > 0 else 0.0
    return out


def pointing_game(saliency: np.ndarray, rois: dict[str, np.ndarray]) -> str:
    """Which ROI contains the saliency peak."""
    peak = np.unravel_index(int(np.argmax(saliency)), saliency.shape)
    for name in ("eyes", "nose"):
        if rois[name][peak]:
            return name
    return "periphery"


# ---------------------------------------------------------------------------
# Per-species pass
# ---------------------------------------------------------------------------


def _build_untrained_like(
    experiment: str, ckpt: Path, device: str, control_seed: int = 0
) -> Any:
    """Same architecture as the checkpoint, at its default initialisation.

    The class count is read from the checkpoint's head so the architecture
    matches exactly, but no *trained* weights are loaded — the point of the
    control is that everything except the architecture is untrained.

    One thing to be precise about, because the write-up has been loose on it:
    ``build_model`` honours the model config, and every arcface config sets
    ``pretrained: imagenet``. So this control is an **ImageNet-pretrained
    backbone with a randomly initialised projection and head**, not a
    from-scratch random network. The trained-minus-untrained contrast therefore
    isolates *bat-specific* training, which is arguably the stronger Adebayo
    control -- but it is not what "untrained network" implies, and it must be
    reported as what it is.
    """
    import torch

    from bat_cli.runtime import build_model, compose_config

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    num_classes = int(state["model"]["head.weight"].shape[0])
    # Seeded: the control's projection and head are randomly initialised, so an
    # unseeded control makes the trained-minus-untrained contrast irreproducible
    # for the same reason the IG noise did.
    torch.manual_seed(control_seed)
    model = build_model(compose_config(experiment=experiment), num_classes=num_classes)
    return model.to(device).eval()


def _top_class_cosine(model: Any, batch: Any) -> float:
    """Cosine between the embedding and its nearest class centre.

    The Grad-CAM target scalar. Hypothesis for the all-zero maps: when this is
    small the gradients are flat, every channel weight comes out negative, and
    ReLU erases the whole map.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        emb = F.normalize(model.projection(model.backbone(batch)), dim=1)
        weight = F.normalize(model.head.weight, dim=1)
        return float((emb @ weight.t()).max().item())


def find_run_dir_for_edge(
    species: str, model_key: str, edge: int, background_slug: str
) -> Path | None:
    """The run for this species/model/edge/**background**, with a checkpoint.

    ``find_run_dir`` matches on the ROC-curve filename only, which does not encode
    the input size — so with both an e112 and an e320 run present it would return
    whichever is newer and silently mix resolutions. The run directory name
    carries the edge (``..._e320_s42``), so filter on that.

    It must also filter on **background**, and it must refuse to guess. The
    earlier version did neither: it took ``max(..., key=mtime)`` over everything
    matching species/model/edge. Once Phase 3 wrote ``green-dilated`` checkpoints
    on 2026-08-24, re-running the published original-background command silently
    selected a *dilated* checkpoint and loaded it into a *non-dilated* ResNet.
    Parameter shapes are identical between the two, so ``load_state_dict`` does
    not raise — every stride and receptive field would have been wrong with no
    error anywhere. Underscore-bracketing the slug is what separates them:
    ``_green_`` does not match ``_green-dilated_``.

    Ambiguity is now an error rather than a coin flip. Ten folds of one cell all
    match legitimately, so the caller is told to pin ``--checkpoint``.
    """
    base = Path("outputs/runs")
    candidates = [
        d
        for d in list(base.glob("*/")) + list(base.glob("*/*/"))
        if f"_e{edge}_" in d.name
        and f"_{background_slug}_" in d.name
        and d.name.startswith(f"{species}_")
        and model_key in d.name
        and resolve_ckpt(d) is not None
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        listing = "\n      ".join(sorted(str(c) for c in candidates))
        raise SystemExit(
            f"ambiguous run directory for {species}/{model_key}/e{edge}/"
            f"{background_slug} — {len(candidates)} candidates:\n      {listing}\n"
            "    Pass --checkpoint to pin one. Refusing to pick by mtime."
        )
    return candidates[0]


def sample_records(manifest: Path, max_per_identity: int, split: str | None) -> list[ImageRecord]:
    df = pd.read_csv(manifest)
    if split:
        df = df[df["split"] == split]
    frames = []
    for _, group in df.groupby("identity"):
        frames.append(group.head(max_per_identity))
    df = pd.concat(frames) if frames else df
    return [
        ImageRecord(
            path=Path(row.path),
            identity=str(row.identity),
            species=row.species,
            background=row.background,
            source=row.source,
            augmented=bool(row.augmented),
            split=row.split,
            quality=float(row.quality),
        )
        for row in df.itertuples(index=False)
    ]


def run_species(
    species: str,
    model_key: str,
    *,
    device: str,
    max_per_identity: int,
    split: str | None,
    randomise: bool,
    model_edge_override: int | None = None,
    method: str = "gradcam",
    background: str = "original",
    checkpoint: str | None = None,
    manifest_override: str | None = None,
) -> dict[str, Any] | None:
    spec = BACKGROUND_SPECS[background]
    experiment = spec["experiment"].format(model=model_key, sp=species)

    # Feed the model the edge length it was TRAINED at (112 for the tuned
    # original-bg experiments), not the analysis frame size. A ResNet50 accepts
    # 224 happily, but the conv grid and effective receptive field change, so the
    # Grad-CAM would describe a configuration that was never trained. The map is
    # upsampled to EDGE afterwards purely so group averaging and the pose-derived
    # ROIs share one coordinate frame.
    from bat_cli.runtime import _resolve_image_size, compose_config

    model_edge = int(
        model_edge_override
        if model_edge_override is not None
        else _resolve_image_size(compose_config(experiment=experiment))
    )

    if checkpoint is not None:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            print(f"  !! checkpoint not found: {ckpt}")
            return None
        run_dir = ckpt.parent
        # An explicitly pinned checkpoint still has to belong to the arm being
        # analysed. Shapes match across dilated/stock and across backgrounds, so
        # nothing downstream would catch a mismatch.
        if f"_{spec['slug']}_" not in run_dir.name:
            raise SystemExit(
                f"checkpoint {ckpt} lives in {run_dir.name}, which is not a "
                f"'{background}' run (expected the token _{spec['slug']}_). "
                "Refusing: parameter shapes match across arms, so this would "
                "load silently and every result would be wrong."
            )
        if f"_e{model_edge}_" not in run_dir.name:
            raise SystemExit(
                f"checkpoint {ckpt} was trained at a different input edge than "
                f"the requested {model_edge}px (dir: {run_dir.name})."
            )
    else:
        run_dir = find_run_dir_for_edge(species, model_key, model_edge, spec["slug"])
        if run_dir is None:
            print(f"  !! no {model_key}/{background} run with a checkpoint for {species}")
            return None
        ckpt = resolve_ckpt(run_dir)
        if ckpt is None:
            print(f"  !! no checkpoint in {run_dir}")
            return None

    print(f"  {species}/{model_key}/{background}: {run_dir.name} (model edge {model_edge}px)")

    if randomise:
        # Adebayo et al.: compare against an untrained network of the SAME
        # architecture. A map that survives this is not explaining anything the
        # model learned.
        #
        # It must be a *functioning* untrained network. An earlier version
        # overwrote weights in place and zeroed every 1-D parameter, which
        # includes each BatchNorm's scale gamma — that makes every activation
        # identically zero, so the network emits nothing and the control
        # "passes" vacuously: it demonstrates a broken model, not a
        # learning-dependent explanation. Building the model without loading the
        # checkpoint gives torchvision's own initialisation (gamma = 1), so the
        # network runs normally and only the training is missing.
        model = _build_untrained_like(experiment, ckpt, device)
    else:
        model = load_model(experiment, ckpt, device, "embedding")

    manifest = Path(
        manifest_override
        if manifest_override is not None
        else (
            spec["aligned"].format(species=species, edge=model_edge)
            if model_edge_override is not None
            else spec["manifest"].format(species=species)
        )
    )
    if not manifest.exists():
        print(f"  !! {manifest} not found")
        return None
    records = sample_records(manifest, max_per_identity, split)
    print(f"    {len(records)} images from {len({r.identity for r in records})} identities")

    from bat_preprocessing import YOLOPoseEstimator

    pose = YOLOPoseEstimator({"weights": POSE_WEIGHTS, "device": device})

    accumulator = np.zeros((EDGE, EDGE), dtype=np.float64)
    sq_accumulator = np.zeros((EDGE, EDGE), dtype=np.float64)
    face_accumulator = np.zeros((EDGE, EDGE, 3), dtype=np.float64)
    n_maps = 0
    rows: list[dict[str, Any]] = []
    n_no_pose = 0
    n_degenerate = 0
    diagnostics: list[dict[str, Any]] = []

    # One image at a time keeps peak memory flat; the batch win is not worth
    # holding hundreds of maps plus their autograd graphs.
    #
    # Uses ``embedding_gradcam(..., "class_logit")`` rather than
    # ``bat_interpretability.GradCAMAdapter``: the adapter's "class" target needs
    # an explicit ``target_class``, which does not exist in an open-set test
    # (held-out identities have no training class). class_logit instead targets
    # the *predicted* identity via argmax over the normalised head weights, which
    # is also the target that was found to localise the eyes sharply.
    for record in records:
        try:
            tensor = default_image_loader(str(record.path), model_edge, "imagenet").unsqueeze(0)
            batch = tensor.to(device)
            if method == "ig":
                # Integrated Gradients attributes at *input* resolution, so it is
                # not capped by the conv grid (edge/32 — only 4x4 at 112 px, which
                # cannot resolve an eye). SmoothGrad averaging tames the per-pixel
                # noise IG is otherwise prone to.
                # Per-image deterministic seed: reproducible, but not the same
                # noise for every image (which would correlate the maps).
                cam = embedding_smoothgrad_ig(
                    model,
                    batch,
                    IG_STEPS,
                    "class_logit",
                    seed=zlib.crc32(str(record.path).encode()) & 0x7FFFFFFF,
                )
            else:
                cam = embedding_gradcam(model, batch, "class_logit")
            top_cosine = _top_class_cosine(model, batch)
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill the pass
            print(f"    !! saliency failed for {record.path.name}: {exc}")
            continue
        saliency = np.asarray(cam, dtype=np.float64)
        if saliency.shape != (EDGE, EDGE):
            # Grad-CAM returns the raw conv grid (edge/32); upsample to the crop.
            saliency = cv2.resize(saliency, (EDGE, EDGE), interpolation=cv2.INTER_LINEAR)
        diagnostics.append(
            {
                "identity": record.identity,
                "top_class_cosine": top_cosine,
                "saliency_max": float(saliency.max()),
                "degenerate": bool(saliency.sum() <= 0),
            }
        )

        accumulator += saliency
        sq_accumulator += saliency**2
        n_maps += 1

        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image.shape[:2] != (EDGE, EDGE):
            image = cv2.resize(image, (EDGE, EDGE), interpolation=cv2.INTER_AREA)
        face_accumulator += image.astype(np.float64)

        prediction = pose.predict(image)
        rois = (
            rois_from_keypoints(prediction.keypoints, (EDGE, EDGE))
            if prediction is not None
            else None
        )
        if rois is None:
            n_no_pose += 1
            continue

        masses = roi_mass(saliency, rois)
        if masses is None:
            n_degenerate += 1
            continue

        rows.append(
            {
                "identity": record.identity,
                "path": str(record.path),
                **masses,
                "peak_roi": pointing_game(saliency, rois),
            }
        )

    if n_maps == 0:
        print(f"  !! no saliency maps produced for {species}")
        return None

    mean_map = accumulator / n_maps
    variance = np.maximum(0.0, sq_accumulator / n_maps - mean_map**2)
    smoothed = cv2.GaussianBlur(mean_map, (0, 0), GROUP_SIGMA_PX)

    print(
        f"    {n_maps} maps, {len(rows)} scored "
        f"({n_no_pose} dropped: no pose, {n_degenerate} dropped: all-zero Grad-CAM)"
    )
    return {
        "species": species,
        "model": model_key,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "randomised_weights": randomise,
        "n_maps": n_maps,
        "n_with_pose": len(rows),
        "n_pose_dropped": n_no_pose,
        "n_degenerate_maps": n_degenerate,
        "group_sigma_px": GROUP_SIGMA_PX,
        "model_edge": model_edge,
        "method": method,
        "diagnostics": diagnostics,
        "mean_map": smoothed,
        "sd_map": np.sqrt(variance),
        "mean_face": (face_accumulator / max(1, n_maps)).astype(np.uint8),
        "per_image": rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def identity_level(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    """Per-identity means — the honest unit, since one bat gives one video."""
    if not rows:
        return np.asarray([], dtype=float)
    df = pd.DataFrame(rows)
    if column not in df.columns:
        return np.asarray([], dtype=float)
    return df.groupby("identity")[column].mean().to_numpy(dtype=float)


def plot_group_maps(
    results: list[dict[str, Any]], fig_dir: Path, model_key: str, arm: str,
    background: str = "original",
) -> Path:
    fig, axes = plt.subplots(2, len(results), figsize=(4.6 * len(results), 8.6), squeeze=False)
    for column, entry in enumerate(results):
        face = cv2.cvtColor(entry["mean_face"], cv2.COLOR_BGR2RGB)

        ax = axes[0][column]
        ax.imshow(face)
        ax.imshow(entry["mean_map"], cmap="inferno", alpha=0.6)
        ax.set_title(
            f"{entry['species']} — mean saliency\n"
            f"n={entry['n_maps']} images, sigma={entry['group_sigma_px']:.0f}px",
            fontsize=10,
        )
        ax.axis("off")

        ax = axes[1][column]
        im = ax.imshow(entry["sd_map"], cmap="viridis")
        ax.set_title(f"{entry['species']} — per-pixel SD", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Group-level Grad-CAM over aligned crops — {model_key} [{arm}] "
        f"(class-logit target, mean face underlay)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "species": "both",
        "source": "video",
        # Real background, not a literal: build_filename keys on it, so a
        # hardcoded "original" makes a green run silently overwrite the original
        # run's figures.
        "background": background,
        "model": model_key,
        "loss": model_key,
    }
    path = fig_dir / build_filename(f"species_group_saliency_{arm}", cfg)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roi_stats(
    results: list[dict[str, Any]], fig_dir: Path, model_key: str, arm: str,
    background: str = "original",
) -> Path:
    regions = ("eyes", "nose", "periphery")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    width = 0.36
    for offset, entry in zip((-width / 2, width / 2), results):
        values = [float(np.mean(identity_level(entry["per_image"], f"{r}_density"))) for r in regions]
        ax.bar(
            np.arange(len(regions)) + offset,
            values,
            width=width,
            label=entry["species"],
            color=SPECIES_COLOURS.get(entry["species"], "#666"),
        )
    ax.axhline(1.0, color="#888", linestyle="--", linewidth=1, label="chance (area-matched)")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_ylabel("saliency density (mass / area)")
    ax.set_title("Where each species' model looks", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for offset, entry in zip((-width / 2, width / 2), results):
        df = pd.DataFrame(entry["per_image"])
        fractions = [
            float((df["peak_roi"] == r).mean()) if not df.empty else 0.0 for r in regions
        ]
        ax.bar(
            np.arange(len(regions)) + offset,
            fractions,
            width=width,
            label=entry["species"],
            color=SPECIES_COLOURS.get(entry["species"], "#666"),
        )
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_ylabel("fraction of images")
    ax.set_title("Pointing game: where the peak lands", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"ROI saliency statistics — {model_key} [{arm}]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "species": "both",
        "source": "video",
        # Real background, not a literal: build_filename keys on it, so a
        # hardcoded "original" makes a green run silently overwrite the original
        # run's figures.
        "background": background,
        "model": model_key,
        "loss": model_key,
    }
    path = fig_dir / build_filename(f"species_roi_saliency_{arm}", cfg)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODEL_KEYS, default="arcface")
    ap.add_argument(
        "--method",
        choices=("gradcam", "ig"),
        default="gradcam",
        help="gradcam is capped by the conv grid (edge/32); ig attributes at full "
        "input resolution and has no ReLU gate, so it also tests whether the "
        "degenerate-map problem is specific to Grad-CAM.",
    )
    ap.add_argument(
        "--model-edge",
        type=int,
        default=None,
        help="Override the model input edge (e.g. 320). Grad-CAM's conv grid is "
        "edge/32, so 112 gives a 4x4 map whose cells are larger than the eye ROI "
        "and 320 gives 10x10. ROI claims need 320.",
    )
    ap.add_argument(
        "--background",
        choices=tuple(BACKGROUND_SPECS),
        default="original",
        help="Which background arm to analyse. `green` is the arm arm 3C does not "
        "invalidate: on `original` a model trained with the face DELETED still "
        "reaches ROC-AUC 0.831/0.780, so attention measured there may describe a "
        "scene-scoring model rather than a face-driven one.",
    )
    ap.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="SPECIES=PATH",
        help="Pin an exact checkpoint, e.g. --checkpoint mauritius=outputs/.../best.pt. "
        "Repeatable, once per species. Required once more than one run matches a "
        "cell (ten folds all match legitimately), and validated against the "
        "background and edge tokens in the directory name.",
    )
    ap.add_argument(
        "--species",
        action="append",
        choices=SPECIES,
        default=None,
        help="Restrict to one species. Default: both. The cross-species "
        "comparison is only emitted when both are present.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Override the manifest. Must be the one the checkpoint was TRAINED "
        "on, or the model sees preprocessing it never learned.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-per-identity", type=int, default=15)
    ap.add_argument(
        "--split",
        default="",
        help="Restrict to one split (train/val/test). Default: all, so the group "
        "map is built from every identity rather than the 2-3 held-out ones.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument(
        "--randomise-control",
        action="store_true",
        help="Also run with randomly re-initialised weights (Adebayo sanity check).",
    )
    args = ap.parse_args()

    # --checkpoint is per species: one invocation analyses both, so a single bare
    # path would silently be applied to the wrong one.
    pinned: dict[str, str] = {}
    for item in args.checkpoint or []:
        if "=" not in item:
            ap.error(f"--checkpoint must be SPECIES=PATH, got {item!r}")
        sp, _, path = item.partition("=")
        if sp not in SPECIES:
            ap.error(f"--checkpoint species must be one of {SPECIES}, got {sp!r}")
        pinned[sp] = path

    print(f"=== per-species saliency ({args.model}, {args.background} background) ===")
    report: dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "group_sigma_px": GROUP_SIGMA_PX,
        "analysis_edge": EDGE,
        "eye_radius_frac": EYE_RADIUS_FRAC,
        "nose_radius_frac": NOSE_RADIUS_FRAC,
        "max_per_identity": args.max_per_identity,
        "split": args.split or "all",
        "background": args.background,
        "pinned_checkpoints": pinned or None,
        "arms": {},
    }

    for randomise in ([False, True] if args.randomise_control else [False]):
        arm = "randomised_weights" if randomise else "trained"
        print(f"\n--- arm: {arm}")
        results = []
        for species in (tuple(args.species) if args.species else SPECIES):
            entry = run_species(
                species,
                args.model,
                device=args.device,
                max_per_identity=args.max_per_identity,
                split=args.split or None,
                randomise=randomise,
                model_edge_override=args.model_edge,
                method=args.method,
                background=args.background,
                checkpoint=pinned.get(species),
                manifest_override=args.manifest,
            )
            if entry is not None:
                results.append(entry)

        if not results:
            print("  no results for this arm")
            continue

        arm_report: dict[str, Any] = {"species": {}}
        for entry in results:
            df = pd.DataFrame(entry["per_image"])
            summary: dict[str, Any] = {
                "n_maps": entry["n_maps"],
                "n_with_pose": entry["n_with_pose"],
                "n_pose_dropped": entry["n_pose_dropped"],
                "n_degenerate_maps": entry["n_degenerate_maps"],
                "model_edge": entry["model_edge"],
                "method": entry["method"],
                "diagnostics": entry["diagnostics"],
                "n_identities": int(df["identity"].nunique()) if not df.empty else 0,
                "checkpoint": entry["checkpoint"],
                "run_dir": entry["run_dir"],
                # Serialised so the trained/untrained contrast can be tested at
                # all. The two arms are built in disjoint loop iterations and
                # never coexist in memory, so pairing has to happen offline —
                # and on (identity, path), because pose runs separately per arm
                # and can drop different images.
                "per_image": entry["per_image"],
            }
            for region in ("eyes", "nose", "periphery"):
                per_identity = identity_level(entry["per_image"], f"{region}_mass")
                density = identity_level(entry["per_image"], f"{region}_density")
                # Grad-CAM maps are sparse and peaky, so per-region density is
                # strongly right-skewed: the mean is pulled up by a few images
                # whose hot spot lands in the region, while the median sits well
                # below it. Report both, or the two statistics disagree and the
                # table silently contradicts the chart.
                summary[region] = {
                    "mass_mean": float(per_identity.mean()) if per_identity.size else float("nan"),
                    "mass_median": (
                        float(np.median(per_identity)) if per_identity.size else float("nan")
                    ),
                    "density_mean": float(density.mean()) if density.size else float("nan"),
                    "density_median": (
                        float(np.median(density)) if density.size else float("nan")
                    ),
                    "area_fraction": float(
                        np.mean(
                            [
                                row[f"{region}_mass"] / row[f"{region}_density"]
                                for row in entry["per_image"]
                                if row.get(f"{region}_density", 0) > 0
                            ]
                        )
                        if entry["per_image"]
                        else float("nan")
                    ),
                }
            if not df.empty:
                # Image-level, as originally reported. Kept for continuity, but
                # it is a proportion over images clustered inside 16 / 12 bats,
                # so a naive binomial interval on it is anticonservative.
                summary["pointing_game"] = {
                    region: float((df["peak_roi"] == region).mean())
                    for region in ("eyes", "nose", "periphery")
                }
                # Identity-level, matching the unit the density statistics use.
                # Each bat contributes the fraction of ITS images peaking in the
                # region, then bats are averaged equally.
                per_bat = df.assign(
                    **{
                        f"_pk_{region}": (df["peak_roi"] == region).astype(float)
                        for region in ("eyes", "nose", "periphery")
                    }
                ).groupby("identity")
                summary["pointing_game_identity_level"] = {
                    region: float(per_bat[f"_pk_{region}"].mean().mean())
                    for region in ("eyes", "nose", "periphery")
                }
            arm_report["species"][entry["species"]] = summary

        # Species comparison of ROI density, identity as the unit.
        if len(results) == 2:
            comparisons = {}
            for region in ("eyes", "nose", "periphery"):
                a = identity_level(results[0]["per_image"], f"{region}_density")
                b = identity_level(results[1]["per_image"], f"{region}_density")
                if a.size >= 2 and b.size >= 2:
                    comparisons[region] = compare_groups(a, b).to_dict()
            arm_report["species_comparison_density"] = comparisons

            print(
                "\n    saliency density (mass/area); 1.0 = the region gets exactly "
                "its area share"
            )
            print(
                f"    {'region':<11} {'area':>6} | {'mau mean':>8} {'mau med':>8} | "
                f"{'rou mean':>8} {'rou med':>8} | {'delta':>6} {'p':>8}"
            )
            for region, comparison in comparisons.items():
                mau = arm_report["species"][results[0]["species"]][region]
                rou = arm_report["species"][results[1]["species"]][region]
                print(
                    f"    {region:<11} {mau['area_fraction']:6.3f} | "
                    f"{mau['density_mean']:8.3f} {mau['density_median']:8.3f} | "
                    f"{rou['density_mean']:8.3f} {rou['density_median']:8.3f} | "
                    f"{comparison['cliffs_delta']:+6.2f} {comparison['p_value']:8.4f}"
                )

            print(
                f"\n    {'species':<11} {'eyes':>7} {'nose':>7} {'periph':>7}  "
                "(pointing game; compare with the area column above)"
            )
            for entry in results:
                df = pd.DataFrame(entry["per_image"])
                if df.empty:
                    continue
                fractions = [
                    float((df["peak_roi"] == region).mean())
                    for region in ("eyes", "nose", "periphery")
                ]
                print(
                    f"    {entry['species']:<11} {fractions[0]:7.2%} "
                    f"{fractions[1]:7.2%} {fractions[2]:7.2%}"
                )

        if not randomise or len(results) == 2:
            arm_report["group_map_figure"] = str(
                plot_group_maps(results, args.fig_dir, args.model, arm, args.background)
            )
            arm_report["roi_figure"] = str(
                plot_roi_stats(results, args.fig_dir, args.model, arm, args.background)
            )

        report["arms"][arm] = arm_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
