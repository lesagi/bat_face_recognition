"""Gradient-based saliency for pair-family (Siamese) models.

PyTorch port of ``app/visualization/saliency.py``. The TF version used
``tf.GradientTape``; the PyTorch equivalent is ``torch.autograd.grad`` (or
``loss.backward()`` on a leaf tensor with ``requires_grad=True``).

Supported methods:

- ``vanilla``                — single backward pass on the pair similarity.
- ``guided``                 — guided backprop (positive gradients only).
- ``integrated_gradients``   — Sundararajan et al. 2017.
- ``smoothgrad``             — Smilkov et al. 2017.

Output:

* one :class:`bat_core.SaliencyImage` per input image record.
* the ``saliency`` field is a 2-D ``numpy.ndarray`` of the same H, W as the
  input (after preprocessing) — channels are aggregated by L2 magnitude.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import torch
    from bat_core.interfaces import FaceModel
    from bat_core.types import ImageRecord, SaliencyImage

SaliencyMethod = Literal[
    "vanilla", "guided", "integrated_gradients", "smoothgrad"
]


@dataclass
class SiameseSaliencyAdapter:
    """Gradient saliency adapter for pair-family models.

    Parameters
    ----------
    method:
        One of ``"vanilla"``, ``"guided"``, ``"integrated_gradients"``,
        ``"smoothgrad"``. Default ``"vanilla"``.
    input_size:
        Edge length used when resizing input images. The TF baseline used
        the model's input shape; here we pass it explicitly. Default ``105``
        to match the ported Siamese network (see
        :data:`bat_models.siamese.SIAMESE_INPUT_EDGE_LENGTH`).
    integration_steps:
        Number of integration steps for ``integrated_gradients``
        (default ``20``).
    smoothing_samples:
        Number of noisy samples for ``smoothgrad`` (default ``10``).
    smoothing_noise:
        Std-dev of Gaussian noise added in ``smoothgrad`` (default ``0.1``).
    aggregation:
        How to reduce gradient channels into a 2-D map. ``"magnitude"`` (L2
        norm — default), ``"max"`` (TF baseline), ``"mean"``, or ``"sum"``.
    apply_smoothing:
        Apply a small Gaussian blur to the final saliency map for cleaner
        visualisation. Default ``True``.
    """

    method: SaliencyMethod = "vanilla"
    input_size: int = 105
    integration_steps: int = 20
    smoothing_samples: int = 10
    smoothing_noise: float = 0.1
    aggregation: Literal["magnitude", "max", "mean", "sum"] = "magnitude"
    apply_smoothing: bool = True

    # ------------------------------------------------------------------ #
    # InterpretabilityAdapter protocol                                   #
    # ------------------------------------------------------------------ #
    def explain(
        self,
        model: FaceModel,
        samples: list[ImageRecord],
    ) -> list[SaliencyImage]:
        """Compute saliency maps for each sample image."""
        from bat_core.types import SaliencyImage

        device = _infer_device(model)
        # Cast to nn.Module for eval()/no-grad semantics.
        torch_model = _as_module(model)
        torch_model.eval()

        results: list[SaliencyImage] = []
        for record in samples:
            image_tensor = _load_and_preprocess(
                record.path, size=self.input_size, device=device
            )
            counterpart = _random_counterpart(image_tensor)
            saliency_2d = self._compute_saliency(
                model, image_tensor, counterpart
            )
            saliency_np = saliency_2d.detach().cpu().numpy().astype(np.float32)
            saliency_np = _normalize(saliency_np)
            if self.apply_smoothing:
                saliency_np = _gaussian_blur_2d(saliency_np, sigma=1.0)
            results.append(
                SaliencyImage(
                    identity=record.identity,
                    image_path=Path(record.path),
                    saliency=saliency_np,
                    method=self.method,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Internal: gradient computation                                     #
    # ------------------------------------------------------------------ #
    def _compute_saliency(
        self,
        model: FaceModel,
        image: torch.Tensor,
        counterpart: torch.Tensor,
    ) -> torch.Tensor:
        if self.method == "vanilla":
            grads = _vanilla_grads(model, image, counterpart)
        elif self.method == "guided":
            grads = _vanilla_grads(model, image, counterpart)
            import torch

            grads = torch.where(grads > 0, grads, torch.zeros_like(grads))
        elif self.method == "integrated_gradients":
            grads = _integrated_gradients(
                model, image, counterpart, steps=self.integration_steps
            )
        elif self.method == "smoothgrad":
            grads = _smooth_grads(
                model,
                image,
                counterpart,
                n_samples=self.smoothing_samples,
                noise_level=self.smoothing_noise,
            )
        else:
            raise ValueError(f"Unknown saliency method: {self.method!r}")

        return _aggregate_channels(grads, mode=self.aggregation)


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _vanilla_grads(
    model: FaceModel,
    image: torch.Tensor,
    counterpart: torch.Tensor,
) -> torch.Tensor:
    """Single-pass gradient of the pair similarity w.r.t. the input image."""
    import torch

    image = image.clone().detach().requires_grad_(True)
    output = _pair_similarity(model, image, counterpart)
    grads = torch.autograd.grad(
        outputs=output.sum(),
        inputs=image,
        retain_graph=False,
        create_graph=False,
    )[0]
    return grads


def _integrated_gradients(
    model: FaceModel,
    image: torch.Tensor,
    counterpart: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Sundararajan et al. 2017."""
    import torch

    baseline = torch.zeros_like(image)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=image.device)
    grad_sum = torch.zeros_like(image)
    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        output = _pair_similarity(model, interpolated, counterpart)
        grad = torch.autograd.grad(output.sum(), interpolated)[0]
        grad_sum = grad_sum + grad
    avg_grads = grad_sum / float(len(alphas))
    return avg_grads * (image - baseline)


def _smooth_grads(
    model: FaceModel,
    image: torch.Tensor,
    counterpart: torch.Tensor,
    n_samples: int,
    noise_level: float,
) -> torch.Tensor:
    """Smilkov et al. 2017."""
    import torch

    grad_sum = torch.zeros_like(image)
    for _ in range(n_samples):
        noise = torch.randn_like(image) * noise_level
        noisy = (image + noise).clamp(0.0, 1.0)
        noisy = noisy.detach().requires_grad_(True)
        output = _pair_similarity(model, noisy, counterpart)
        grad = torch.autograd.grad(output.sum(), noisy)[0]
        grad_sum = grad_sum + grad
    return grad_sum / float(n_samples)


def _pair_similarity(
    model: FaceModel,
    image: torch.Tensor,
    counterpart: torch.Tensor,
) -> torch.Tensor:
    """Run the pair-family model on (image, counterpart).

    Tries the standard ``forward_train(x_pair)`` API first (stacking on a
    new ``pair`` dim), falls back to passing a 2-tuple if the model expects
    that. Always returns a similarity scalar tensor of shape ``(B,)`` or
    ``(B, 1)``.
    """
    import torch

    # Stacked layout: (B, 2, C, H, W).
    pair = torch.stack([image, counterpart], dim=1)
    try:
        out = model.forward_train(pair, None)  # type: ignore[arg-type]
    except (TypeError, ValueError, RuntimeError):
        # Fall back to tuple layout.
        out = model.forward_train((image, counterpart), None)  # type: ignore[arg-type]
    return out


def _aggregate_channels(
    grads: torch.Tensor, mode: str
) -> torch.Tensor:
    """Reduce a (B, C, H, W) gradient tensor to a (H, W) saliency map.

    Assumes ``B == 1`` (per-image saliency); the leading batch dim is dropped.
    """
    import torch

    if grads.ndim != 4:
        raise ValueError(
            f"expected gradient tensor of shape (B, C, H, W); got {tuple(grads.shape)}"
        )
    g = grads[0]  # (C, H, W)
    if mode == "magnitude":
        s = torch.sqrt(torch.sum(g.pow(2), dim=0))
    elif mode == "max":
        s = torch.max(torch.abs(g), dim=0).values
    elif mode == "mean":
        s = torch.mean(torch.abs(g), dim=0)
    elif mode == "sum":
        s = torch.sum(torch.abs(g), dim=0)
    else:
        raise ValueError(f"Unknown aggregation: {mode!r}")
    return s


def _load_and_preprocess(
    path: Path | str, size: int, device: torch.device
) -> torch.Tensor:
    """Load image from disk and return a (1, 3, size, size) tensor in [0, 1]."""
    import torch
    from PIL import Image

    img = Image.open(str(path)).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _random_counterpart(image: torch.Tensor) -> torch.Tensor:
    """Generate a pseudo-random counterpart image with similar statistics."""
    import torch

    return torch.rand_like(image)


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Light-weight Gaussian blur (avoids scipy hard dep)."""
    try:
        from scipy import ndimage  # type: ignore[import-untyped]

        return ndimage.gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except ImportError:
        # Box-blur fallback (3x3) — good enough for visual smoothing.
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        h, w = arr.shape
        padded = np.pad(arr, 1, mode="edge")
        out = np.zeros_like(arr)
        for i in range(h):
            for j in range(w):
                out[i, j] = float(
                    np.sum(padded[i : i + 3, j : j + 3] * kernel)
                )
        return out


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Map a 2-D array to ``[0, 1]``."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _infer_device(model: FaceModel) -> torch.device:
    """Return the device that the model's parameters live on."""
    import torch

    try:
        return next(_as_module(model).parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _as_module(model: FaceModel):
    """Best-effort ``nn.Module`` view of a FaceModel."""
    import torch

    if isinstance(model, torch.nn.Module):
        return model
    raise TypeError("SiameseSaliencyAdapter requires a torch.nn.Module FaceModel")


# Silence unused-import lint when running with no torch installed (for
# documentation/IDE completeness).
_ = io
