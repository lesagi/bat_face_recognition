"""Measure how separable each species' individuals are, before any training.

The reviewer asked for an SNR-like measure. The image-quality battery
(``analyze_quality_parity.py``) answers the *acquisition* half of that question.
This script answers the half that actually matters for recognition:

    signal = variation between individuals
    noise  = variation between images of the same individual

Their ratio is the intrinsic difficulty of the dataset. It is measured in a
**frozen, task-agnostic embedding** — ImageNet-pretrained ResNet50, never a model
trained on these bats — so it says how distinguishable the individuals are *before*
our models get involved. A species whose individuals are less separable here is
harder for reasons that have nothing to do with our architecture choices.

Three metrics, all on L2-normalised embeddings (cosine geometry, matching how the
trained models are evaluated):

* ``d_prime``       -- separation between the within-identity and between-identity
                       cosine-similarity distributions.
* ``fisher_ratio``  -- between-identity scatter over within-identity scatter.
* ``silhouette``    -- mean silhouette of identity clusters under cosine distance.

Three sampling regimes, because the raw comparison is confounded:

* ``raw``                 -- every image, every identity. Confounded by identity
                             count (16 vs 12), images per identity (median 32 vs
                             91), and effective resolution (1.97x).
* ``matched_design``      -- equal identity count and equal images per identity,
                             removing the sample-size confound.
* ``matched_resolution``  -- additionally restricted to the native-resolution band
                             both species share, removing the resolution confound.
                             **This is the comparison to quote.**

Usage:
    uv run python scripts/analyze_separability.py --device cpu
    uv run python scripts/analyze_separability.py --background green --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from bat_stats import build_filename  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

SPECIES_A = "mauritius"
SPECIES_B = "rousettus"
BACKGROUNDS = ("green", "original", "random")

MANIFEST_TEMPLATE = "data/manifests/{species}_{background}_aligned_{edge}_manifest.csv"
DEFAULT_NATIVE_BOXES = Path("outputs/quality/native_boxes.parquet")
DEFAULT_OUT = Path("outputs/quality/separability.json")
DEFAULT_FIG_DIR = Path("outputs/quality/figures")

# ImageNet normalisation — the backbone is pretrained, so inputs must match the
# statistics it was trained on (same reasoning as the fix in bat_data).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REGIMES = ("raw", "matched_design", "matched_resolution")

# Inside the shared resolution band each species has far fewer images, so the
# matched design is re-solved there; refuse designs thinner than this.
MIN_IMAGES_PER_IDENTITY_IN_BAND = 8


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def load_backbone(device: str) -> torch.nn.Module:
    """Frozen ImageNet ResNet50 with the classifier removed."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def embed_paths(
    paths: list[str],
    model: torch.nn.Module,
    *,
    device: str,
    edge: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Return L2-normalised embeddings, one row per path."""
    import cv2

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = []
            for path in chunk:
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise SystemExit(f"unreadable image: {path}")
                if bgr.shape[0] != edge or bgr.shape[1] != edge:
                    bgr = cv2.resize(bgr, (edge, edge), interpolation=cv2.INTER_AREA)
                images.append(bgr[:, :, ::-1].copy())
            batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
            batch = (batch.to(device) - mean) / std
            features = model(batch)
            features = torch.nn.functional.normalize(features, dim=1)
            out.append(features.cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Separability metrics
# ---------------------------------------------------------------------------


def similarity_pools(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split all pairwise cosine similarities into within- and between-identity."""
    sim = embeddings @ embeddings.T
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    values = sim[iu]
    return values[same], values[~same]


def d_prime(within: np.ndarray, between: np.ndarray) -> float:
    """Signal-detection d': separation of the two similarity distributions."""
    if within.size < 2 or between.size < 2:
        return float("nan")
    pooled_sd = np.sqrt((within.var(ddof=1) + between.var(ddof=1)) / 2.0)
    if pooled_sd <= 1e-12:
        return float("nan")
    return float((within.mean() - between.mean()) / pooled_sd)


def fisher_ratio(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Between-identity scatter divided by within-identity scatter."""
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    global_centroid = embeddings.mean(axis=0)
    centroids = np.stack([embeddings[labels == u].mean(axis=0) for u in unique])
    between = float(np.mean(np.sum((centroids - global_centroid) ** 2, axis=1)))
    within_terms = [
        float(np.mean(np.sum((embeddings[labels == u] - centroids[i]) ** 2, axis=1)))
        for i, u in enumerate(unique)
    ]
    within = float(np.mean(within_terms))
    if within <= 1e-12:
        return float("nan")
    return between / within


def separability(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    within, between = similarity_pools(embeddings, labels)
    unique = np.unique(labels)
    result = {
        "d_prime": d_prime(within, between),
        "fisher_ratio": fisher_ratio(embeddings, labels),
        "mean_within_similarity": float(within.mean()) if within.size else float("nan"),
        "mean_between_similarity": float(between.mean()) if between.size else float("nan"),
        "n_images": int(embeddings.shape[0]),
        "n_identities": int(unique.size),
    }
    if unique.size >= 2 and embeddings.shape[0] > unique.size:
        result["silhouette"] = float(
            silhouette_score(embeddings, labels, metric="cosine")
        )
    else:
        result["silhouette"] = float("nan")
    return result


def jackknife_over_identities(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    delete: int = 2,
    max_replicates: int = 400,
    seed: int = 42,
) -> tuple[float, float]:
    """Delete-*d* jackknife interval for a separability metric, over identities.

    Identities are the independent units — images of one bat come from one video,
    so resampling images would fabricate precision.

    A *jackknife* rather than a bootstrap, deliberately. Resampling identities
    with replacement makes a twice-drawn identity appear as two clusters whose
    centroids coincide, which corrupts every clustering metric: silhouette and
    Fisher ratio both collapse, and the resulting interval does not even contain
    the point estimate. Deleting identities instead keeps every cluster distinct.

    The interval describes the metric at ``n - delete`` identities, so it is
    comparable across species only when the identity count matches — which is
    what the ``matched_*`` regimes are for.
    """
    unique = np.unique(labels)
    if unique.size - delete < 2:
        return float("nan"), float("nan")

    from itertools import combinations

    all_subsets = list(combinations(range(unique.size), delete))
    if len(all_subsets) > max_replicates:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(all_subsets), size=max_replicates, replace=False)
        subsets = [all_subsets[i] for i in picks]
    else:
        subsets = all_subsets

    replicates: list[float] = []
    for dropped in subsets:
        keep = np.setdiff1d(unique, unique[list(dropped)])
        sel = np.isin(labels, keep)
        value = separability(embeddings[sel], labels[sel])[metric]
        if np.isfinite(value):
            replicates.append(value)

    if not replicates:
        return float("nan"), float("nan")
    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return float(lo), float(hi)


def matched_images_per_identity(frames: dict[str, pd.DataFrame], n_identities: int) -> int:
    """Largest images-per-identity that *both* species can supply for *n_identities*.

    Taking the median count per species (the obvious choice) is infeasible: only
    8 of 16 mauritius identities reach rousettus' median of 32. The binding
    constraint is the ``n_identities``-th largest identity count in each species.
    """
    limits = []
    for frame in frames.values():
        counts = sorted(frame.groupby("identity").size().to_numpy(), reverse=True)
        if len(counts) < n_identities:
            raise ValueError(f"only {len(counts)} identities available; need {n_identities}")
        limits.append(int(counts[n_identities - 1]))
    return max(2, min(limits))


# ---------------------------------------------------------------------------
# Sampling regimes
# ---------------------------------------------------------------------------


def subsample(
    df: pd.DataFrame,
    *,
    n_identities: int,
    n_per_identity: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministically take *n_identities* identities and *n_per_identity* images each."""
    rng = np.random.default_rng(seed)
    counts = df.groupby("identity").size()
    eligible = sorted(counts[counts >= n_per_identity].index)
    if len(eligible) < n_identities:
        raise ValueError(
            f"only {len(eligible)} identities have >= {n_per_identity} images; "
            f"need {n_identities}"
        )
    chosen = rng.choice(np.asarray(eligible, dtype=object), size=n_identities, replace=False)
    frames = []
    for identity in chosen:
        rows = df[df["identity"] == identity]
        take = rng.choice(rows.index.to_numpy(), size=n_per_identity, replace=False)
        frames.append(df.loc[take])
    return pd.concat(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--background", choices=(*BACKGROUNDS, "all"), default="green")
    ap.add_argument("--edge", type=int, default=224)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--n-boot",
        type=int,
        default=400,
        dest="n_boot",
        help="Cap on delete-2 jackknife replicates per interval.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--native-boxes", type=Path, default=DEFAULT_NATIVE_BOXES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = ap.parse_args()

    bg_list = BACKGROUNDS if args.background == "all" else (args.background,)

    native = None
    if args.native_boxes.exists():
        native = pd.read_parquet(args.native_boxes)[["basename", "species", "native_side_px"]]
    else:
        print(f"! {args.native_boxes} not found — matched_resolution regime will be skipped")

    print("=== intrinsic separability (frozen ImageNet ResNet50) ===")
    print(f"device={args.device} edge={args.edge} jackknife<={args.n_boot} replicates\n")
    model = load_backbone(args.device)

    report: dict[str, Any] = {
        "backbone": "torchvision resnet50 IMAGENET1K_V2 (frozen, fc removed)",
        "embedding": "2048-d global average pool, L2-normalised",
        "interval_method": "delete-2 jackknife over identities (95% percentile)",
        "max_jackknife_replicates": args.n_boot,
        "backgrounds": {},
    }

    for background in bg_list:
        print(f"--- background={background}")
        per_species: dict[str, pd.DataFrame] = {}
        for species in (SPECIES_A, SPECIES_B):
            manifest = Path(
                MANIFEST_TEMPLATE.format(species=species, background=background, edge=args.edge)
            )
            if not manifest.exists():
                print(f"  SKIP {species}: {manifest} not found")
                continue
            frame = pd.read_csv(manifest)
            frame["basename"] = frame["path"].map(lambda p: Path(p).name)
            frame["species"] = species
            if native is not None:
                frame = frame.merge(native, on=["basename", "species"], how="left")
            per_species[species] = frame

        if len(per_species) < 2:
            continue

        # Embed once per species; every regime is a row subset of the same matrix.
        embeddings: dict[str, np.ndarray] = {}
        for species, frame in per_species.items():
            print(f"  embedding {species}: {len(frame)} images ...", flush=True)
            embeddings[species] = embed_paths(
                frame["path"].tolist(), model, device=args.device, edge=args.edge
            )

        # Matched design: equal identities and equal images per identity.
        n_ids = min(f["identity"].nunique() for f in per_species.values())
        n_per = matched_images_per_identity(per_species, n_ids)

        # Matched resolution: the native-side band both species share.
        band: tuple[float, float] | None = None
        band_ids = band_per = 0
        if native is not None and all("native_side_px" in f.columns for f in per_species.values()):
            lo = max(float(f["native_side_px"].min()) for f in per_species.values())
            hi = min(float(f["native_side_px"].max()) for f in per_species.values())
            band = (lo, hi)

            # Inside the band both species have fewer usable images, so the
            # matched design must be re-solved: pick the (identities, images)
            # pair that maximises total images while both species can supply it.
            in_band = {
                sp: f[f["native_side_px"].between(lo, hi)] for sp, f in per_species.items()
            }
            # Maximise the *identity* count first, then take whatever
            # images-per-identity both species can supply at that count.
            # Identities are the inference unit — the jackknife resamples them,
            # and d'/Fisher/silhouette all depend on how many clusters there
            # are — so 10 identities x 8 images beats 6 x 25 even though it is
            # fewer images overall.
            available = min(f["identity"].nunique() for f in in_band.values())
            band_ids = band_per = 0
            for ids in range(available, 1, -1):
                try:
                    per = matched_images_per_identity(in_band, ids)
                except ValueError:
                    continue
                if per < MIN_IMAGES_PER_IDENTITY_IN_BAND:
                    continue
                band_ids, band_per = ids, per
                break
            if band_ids == 0:
                band = None

        bg_report: dict[str, Any] = {
            "matched_design": {"n_identities": n_ids, "n_per_identity": n_per},
            "resolution_band_px": list(band) if band else None,
            "matched_resolution": {"n_identities": band_ids, "n_per_identity": band_per},
            "regimes": {},
        }

        for regime in REGIMES:
            regime_out: dict[str, Any] = {}
            skipped = False
            for species, frame in per_species.items():
                emb_all = embeddings[species]
                positions = pd.Series(np.arange(len(frame)), index=frame.index)

                if regime == "raw":
                    subset = frame
                elif regime == "matched_design":
                    subset = subsample(
                        frame, n_identities=n_ids, n_per_identity=n_per, seed=args.seed
                    )
                else:
                    if band is None:
                        skipped = True
                        break
                    in_band = frame[frame["native_side_px"].between(*band)]
                    if in_band.empty:
                        skipped = True
                        break
                    subset = subsample(
                        in_band,
                        n_identities=band_ids,
                        n_per_identity=band_per,
                        seed=args.seed,
                    )

                rows = positions.loc[subset.index].to_numpy()
                emb = emb_all[rows]
                labels = subset["identity"].to_numpy()
                stats_out = separability(emb, labels)
                for metric in ("d_prime", "fisher_ratio", "silhouette"):
                    lo_ci, hi_ci = jackknife_over_identities(
                        emb, labels, metric=metric, max_replicates=args.n_boot, seed=args.seed
                    )
                    stats_out[f"{metric}_ci_low"] = lo_ci
                    stats_out[f"{metric}_ci_high"] = hi_ci
                regime_out[species] = stats_out

            if skipped or len(regime_out) < 2:
                print(f"  {regime:<20} SKIPPED (insufficient data)")
                continue

            bg_report["regimes"][regime] = regime_out
            a, b = regime_out[SPECIES_A], regime_out[SPECIES_B]
            print(
                f"  {regime:<20} "
                f"ids={a['n_identities']}/{b['n_identities']} "
                f"imgs={a['n_images']}/{b['n_images']}"
            )
            for metric in ("d_prime", "fisher_ratio", "silhouette"):
                print(
                    f"    {metric:<14} "
                    f"mau={a[metric]:7.3f} [{a[f'{metric}_ci_low']:6.3f},{a[f'{metric}_ci_high']:6.3f}]   "
                    f"rou={b[metric]:7.3f} [{b[f'{metric}_ci_low']:6.3f},{b[f'{metric}_ci_high']:6.3f}]"
                )

        report["backgrounds"][background] = bg_report
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    # Figure: d' with identity-bootstrap CIs, one group of bars per regime.
    for background, bg_report in report["backgrounds"].items():
        regimes = [r for r in REGIMES if r in bg_report["regimes"]]
        if not regimes:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ("d_prime", "fisher_ratio", "silhouette")):
            width = 0.36
            for offset, species, colour in (
                (-width / 2, SPECIES_A, "#A2582C"),
                (width / 2, SPECIES_B, "#2F6B7A"),
            ):
                xs = np.arange(len(regimes)) + offset
                vals = [bg_report["regimes"][r][species][metric] for r in regimes]
                lows = [bg_report["regimes"][r][species][f"{metric}_ci_low"] for r in regimes]
                highs = [bg_report["regimes"][r][species][f"{metric}_ci_high"] for r in regimes]
                errs = np.vstack(
                    [
                        np.maximum(0.0, np.asarray(vals) - np.asarray(lows)),
                        np.maximum(0.0, np.asarray(highs) - np.asarray(vals)),
                    ]
                )
                ax.bar(xs, vals, width=width, color=colour, label=species, yerr=errs, capsize=3)
            ax.set_xticks(np.arange(len(regimes)))
            ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
            ax.set_title(metric, fontsize=10)
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"Intrinsic identity separability, frozen ImageNet ResNet50 "
            f"— background={background} (delete-2 jackknife over identities)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "species": "both",
            "source": "video",
            "background": background,
            "model": "resnet50",
            "loss": "none",
        }
        path = args.fig_dir / build_filename("separability", cfg)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"figure: {path}")

    print("\nQuote the matched_resolution regime; see docs/quality_parity.md for why.")


if __name__ == "__main__":
    main()
