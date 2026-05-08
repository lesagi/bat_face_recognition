"""Embedding-projection adapter — t-SNE and UMAP scatter plots.

Given a batch of embeddings (or a list of :class:`bat_core.ImageRecord` from
which embeddings will be computed via the model), produce a 2-D projection
and save a PNG plot per technique. Returns one :class:`SaliencyImage` whose
``saliency`` field contains the (N, 2) projection coordinates (numpy float32)
and whose ``image_path`` points at the saved PNG.

This adapter is independent of model family — it works on any embedding —
and is always returned by the dispatcher when 2-D projections are requested.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from collections.abc import Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from bat_core.interfaces import FaceModel
    from bat_core.types import Embedding, ImageRecord, SaliencyImage

ProjectionMethod = Literal["tsne", "umap", "both"]


@dataclass
class EmbeddingProjectionAdapter:
    """Project embeddings to 2-D and save a scatter plot.

    Parameters
    ----------
    method:
        ``"tsne"``, ``"umap"``, or ``"both"`` (default — produces one PNG per
        technique side-by-side).
    output_dir:
        Where to save the PNG. If ``None``, a system temp directory is used.
    perplexity:
        t-SNE perplexity (default ``min(30, n-1)`` clamped at runtime).
    n_neighbors:
        UMAP ``n_neighbors`` (default ``15``).
    random_state:
        Reproducibility seed (default ``42``).
    figsize:
        Matplotlib figure size in inches (default ``(10, 8)``).
    input_size:
        Image edge length used when computing embeddings from ``ImageRecord``
        samples (default ``224``).
    """

    method: ProjectionMethod = "both"
    output_dir: Path | None = None
    perplexity: float = 30.0
    n_neighbors: int = 15
    random_state: int = 42
    figsize: tuple[float, float] = (10.0, 8.0)
    input_size: int = 224
    _produced: list[Path] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # InterpretabilityAdapter protocol                                   #
    # ------------------------------------------------------------------ #
    def explain(
        self,
        model: FaceModel,
        samples: list[ImageRecord],
    ) -> list[SaliencyImage]:
        """Compute embeddings for ``samples`` and project them to 2-D."""

        embeddings, identities = _compute_embeddings(
            model, samples, size=self.input_size
        )
        out_dir = self._resolve_output_dir()

        results: list[SaliencyImage] = []
        if self.method in ("tsne", "both"):
            coords_tsne = self._project_tsne(embeddings)
            png = self._save_scatter(coords_tsne, identities, out_dir, "tsne")
            results.append(self._wrap(samples, coords_tsne, png, "tsne"))
        if self.method in ("umap", "both"):
            coords_umap = self._project_umap(embeddings)
            png = self._save_scatter(coords_umap, identities, out_dir, "umap")
            results.append(self._wrap(samples, coords_umap, png, "umap"))
        return results

    # ------------------------------------------------------------------ #
    # Direct entry point — no model, just embeddings                     #
    # ------------------------------------------------------------------ #
    def project_embeddings(
        self, embedding: Embedding
    ) -> list[SaliencyImage]:
        """Project pre-computed :class:`bat_core.Embedding` to 2-D."""
        from bat_core.types import SaliencyImage

        tensor = _embedding_array(embedding)
        identities = list(embedding.identities)
        out_dir = self._resolve_output_dir()

        results: list[SaliencyImage] = []
        if self.method in ("tsne", "both"):
            coords = self._project_tsne(tensor)
            png = self._save_scatter(coords, identities, out_dir, "tsne")
            results.append(
                SaliencyImage(
                    identity="<projection>",
                    image_path=png,
                    saliency=coords.astype(np.float32),
                    method="tsne",
                )
            )
        if self.method in ("umap", "both"):
            coords = self._project_umap(tensor)
            png = self._save_scatter(coords, identities, out_dir, "umap")
            results.append(
                SaliencyImage(
                    identity="<projection>",
                    image_path=png,
                    saliency=coords.astype(np.float32),
                    method="umap",
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #
    def _project_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE

        n = embeddings.shape[0]
        # t-SNE requires perplexity < n_samples.
        perplexity = float(min(self.perplexity, max(2, n - 1)))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.random_state,
            init="random",
            learning_rate="auto",
        )
        return tsne.fit_transform(embeddings).astype(np.float32)

    def _project_umap(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            import umap  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "umap-learn is required for UMAP projection; install it via "
                "`pip install umap-learn`."
            ) from exc

        n = embeddings.shape[0]
        n_neighbors = int(min(self.n_neighbors, max(2, n - 1)))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings).astype(np.float32)

    def _save_scatter(
        self,
        coords: np.ndarray,
        identities: Sequence[str],
        out_dir: Path,
        tag: str,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")  # noqa: E402  - headless safety
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"embedding_projection_{tag}.png"

        fig, ax = plt.subplots(figsize=self.figsize)
        unique = sorted(set(identities))
        cmap = plt.get_cmap("tab20", max(1, len(unique)))
        for idx, ident in enumerate(unique):
            mask = np.array([i == ident for i in identities])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                color=cmap(idx),
                label=str(ident),
                alpha=0.8,
                edgecolor="none",
            )
        ax.set_title(f"{tag.upper()} embedding projection (n={len(identities)})")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        if len(unique) <= 20:
            ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self._produced.append(png_path)
        return png_path

    def _wrap(
        self,
        samples: Sequence[ImageRecord],
        coords: np.ndarray,
        png: Path,
        tag: str,
    ) -> SaliencyImage:
        from bat_core.types import SaliencyImage

        identity = (
            samples[0].identity if samples else "<projection>"
        )
        return SaliencyImage(
            identity=identity,
            image_path=png,
            saliency=coords.astype(np.float32),
            method=tag,
        )

    def _resolve_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return Path(tempfile.mkdtemp(prefix="bat_projection_"))


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _compute_embeddings(
    model: FaceModel,
    samples: list[ImageRecord],
    size: int,
) -> tuple[np.ndarray, list[str]]:
    """Run ``forward_embedding`` over each image; return (N, D), [identity]."""
    import torch
    from PIL import Image

    if not samples:
        raise ValueError("samples list is empty; nothing to project")

    if not isinstance(model, torch.nn.Module):
        raise TypeError("EmbeddingProjectionAdapter requires a torch.nn.Module model")

    model.eval()
    device = next(model.parameters()).device

    tensors: list[torch.Tensor] = []
    identities: list[str] = []
    for record in samples:
        img = (
            Image.open(str(record.path))
            .convert("RGB")
            .resize((size, size), Image.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        identities.append(record.identity)

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        emb = model.forward_embedding(batch)
    return emb.detach().cpu().numpy().astype(np.float32), identities


def _embedding_array(embedding: Embedding) -> np.ndarray:
    """Return ``embedding.tensor`` as a 2-D numpy array."""
    import torch

    tensor = embedding.tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
    return np.asarray(tensor, dtype=np.float32)
