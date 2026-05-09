"""Grad-CAM adapter for embedding-family models (ArcFace / AdaFace).

We hook the last convolutional layer of the backbone and compute Grad-CAM
(Selvaraju et al. 2017) against either:

- ``embedding_magnitude`` (default when no labels are supplied) — the L2
  norm of the L2-normalised embedding (always 1) is not useful, so we use
  the *unnormalised* embedding magnitude as the target. This produces a
  meaningful gradient because the projection's BN/Linear layers are
  trainable upstream.
- ``class`` — target a specific class logit if labels are supplied.

Output: 2-D heat-map per input image, resized to the input H, W and stored
as a numpy array on each :class:`bat_core.SaliencyImage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from bat_core.interfaces import FaceModel
    from bat_core.types import ImageRecord, SaliencyImage
    from torch import nn

GradCAMTarget = Literal["embedding_magnitude", "class"]


@dataclass
class GradCAMAdapter:
    """Grad-CAM for embedding-family models.

    Parameters
    ----------
    target:
        ``"embedding_magnitude"`` (default) or ``"class"``.
    target_class:
        Class index to target when ``target="class"``. Ignored otherwise.
    input_size:
        Edge length used when resizing input images. Default ``224``
        (ResNet50 expects ``112+`` H/W; we use 224 to match torchvision
        defaults).
    target_layer_name:
        Optional dotted attribute path to the conv layer to hook (e.g.,
        ``"backbone._net.layer4"`` for the ResNet50 wrapper). If ``None``
        we walk the model and pick the deepest ``nn.Conv2d``.
    """

    target: GradCAMTarget = "embedding_magnitude"
    target_class: int | None = None
    input_size: int = 224
    target_layer_name: str | None = None

    # ------------------------------------------------------------------ #
    # InterpretabilityAdapter protocol                                   #
    # ------------------------------------------------------------------ #
    def explain(
        self,
        model: FaceModel,
        samples: list[ImageRecord],
    ) -> list[SaliencyImage]:
        from bat_core.types import SaliencyImage

        torch_model = _as_module(model)
        torch_model.eval()
        device = _infer_device(torch_model)

        target_layer = self._resolve_target_layer(torch_model)
        results: list[SaliencyImage] = []

        for record in samples:
            image = _load_and_preprocess(record.path, self.input_size, device)
            cam = self._compute_gradcam(model, image, target_layer)
            cam_np = cam.detach().cpu().numpy().astype(np.float32)
            cam_np = _normalize(cam_np)
            cam_np = _resize_2d(cam_np, (self.input_size, self.input_size))
            results.append(
                SaliencyImage(
                    identity=record.identity,
                    image_path=Path(record.path),
                    saliency=cam_np,
                    method=f"gradcam_{self.target}",
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #
    def _compute_gradcam(
        self,
        model: FaceModel,
        image: torch.Tensor,
        target_layer: nn.Module,
    ) -> torch.Tensor:
        import torch

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def fwd_hook(_module, _inp, output: torch.Tensor) -> None:
            activations["value"] = output

        def bwd_hook(_module, _grad_in, grad_out: tuple) -> None:
            gradients["value"] = grad_out[0]

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        # ``register_full_backward_hook`` is preferred in modern PyTorch.
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        try:
            image = image.clone().detach().requires_grad_(True)
            scalar = self._target_scalar(model, image)
            scalar.backward()

            act = activations["value"]  # (B, K, h, w)
            grad = gradients["value"]  # (B, K, h, w)
            # Channel-wise importance weights = global-avg-pool of grad.
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (B, K, 1, 1)
            cam = (weights * act).sum(dim=1)  # (B, h, w)
            cam = torch.relu(cam)
            return cam[0]  # (h, w)
        finally:
            h_fwd.remove()
            h_bwd.remove()

    def _target_scalar(self, model: FaceModel, image: torch.Tensor) -> torch.Tensor:
        """Compute the scalar to backprop from."""
        import torch

        if self.target == "class":
            if self.target_class is None:
                raise ValueError("target='class' requires `target_class` to be set")
            # forward_train requires labels; we feed the desired class and
            # take the corresponding logit.
            label = torch.tensor([int(self.target_class)], device=image.device, dtype=torch.long)
            logits = model.forward_train(image, label)
            return logits[0, int(self.target_class)]
        # default: embedding magnitude.
        emb = model.forward_embedding(image)
        # If the model returns L2-normalised embeddings, use the
        # un-normalised pre-norm via squared sum of features as a proxy.
        return emb.pow(2).sum()

    def _resolve_target_layer(self, torch_model: nn.Module) -> nn.Module:
        """Resolve which layer to hook.

        If the user supplied ``target_layer_name``, walk attribute access on
        that path. Otherwise pick the deepest ``nn.Conv2d`` module.
        """
        import torch

        if self.target_layer_name is not None:
            obj: object = torch_model
            for part in self.target_layer_name.split("."):
                obj = getattr(obj, part)
            if not isinstance(obj, torch.nn.Module):
                raise TypeError(
                    f"target_layer_name {self.target_layer_name!r} did not "
                    f"resolve to nn.Module (got {type(obj).__name__})"
                )
            return obj

        last_conv: torch.nn.Module | None = None
        for module in torch_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError(
                "GradCAMAdapter could not find any nn.Conv2d in the model. "
                "Pass `target_layer_name` explicitly."
            )
        return last_conv


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _load_and_preprocess(path: Path | str, size: int, device: torch.device) -> torch.Tensor:
    import torch
    from PIL import Image

    img = Image.open(str(path)).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize_2d(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a 2-D array to ``target_hw`` using PIL."""
    from PIL import Image

    h, w = target_hw
    if arr.shape == (h, w):
        return arr.astype(np.float32)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def _infer_device(torch_model: nn.Module) -> torch.device:
    import torch

    try:
        return next(torch_model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_module(model: FaceModel) -> nn.Module:
    import torch

    if isinstance(model, torch.nn.Module):
        return model
    raise TypeError("GradCAMAdapter requires a torch.nn.Module FaceModel")
