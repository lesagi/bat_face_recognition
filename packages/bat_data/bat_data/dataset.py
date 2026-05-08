"""PyTorch :class:`~torch.utils.data.Dataset` over a Manifest split.

Returns ``(image_tensor, identity_int, record)`` per index. Identity
integers are stable across the dataset's lifetime — derived from the
**alphabetically sorted** list of unique identities present in the
filtered manifest. The original :class:`bat_core.ImageRecord` is
returned alongside so trainers / miners can still reach the metadata
without a second lookup.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from bat_core import ImageRecord, Manifest

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.utils.data import Dataset as _DatasetBase
else:
    try:
        from torch.utils.data import Dataset as _DatasetBase
    except ImportError:  # pragma: no cover - bat_data declares torch as a dep
        _DatasetBase = object  # type: ignore[assignment,misc]

__all__ = ["BatDataset", "default_image_loader"]


def default_image_loader(path: Path, image_size: int = 224) -> torch.Tensor:
    """Load an image as a ``float32`` ``(3, H, W)`` tensor in ``[0, 1]``.

    Falls back gracefully if torchvision is missing — uses OpenCV +
    NumPy directly. Both branches resize to ``image_size``.
    """
    import numpy as np  # local: keep top-level import light
    import torch

    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
    except ImportError:  # pragma: no cover - cv2 is a dep
        from PIL import Image

        with Image.open(path) as pil:
            pil = pil.convert("RGB").resize((image_size, image_size))
            img = np.asarray(pil)

    tensor = torch.from_numpy(img.astype("float32") / 255.0)
    # (H, W, C) -> (C, H, W)
    return tensor.permute(2, 0, 1).contiguous()


class BatDataset(_DatasetBase):
    """Dataset over a Manifest filtered by split.

    Parameters
    ----------
    manifest:
        Source manifest (already split).
    split:
        ``"train" | "val" | "test"`` — selects which records to expose.
    loader:
        Callable ``(Path) -> Tensor`` returning a ``(C, H, W)`` float
        tensor. Defaults to :func:`default_image_loader` (OpenCV-based).
    image_size:
        Spatial size passed to the default loader. Ignored if a custom
        ``loader`` is provided.
    transform:
        Optional ``Tensor -> Tensor`` (e.g. albumentations / torchvision
        v2) applied after loading.
    """

    def __init__(
        self,
        manifest: Manifest,
        split: str = "train",
        *,
        loader: Callable[[Path], torch.Tensor] | None = None,
        image_size: int = 224,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train'/'val'/'test'; got {split!r}")
        self._records: tuple[ImageRecord, ...] = tuple(
            r for r in manifest.records if r.split == split
        )
        if not self._records:
            raise ValueError(f"no records found for split {split!r}")

        self._split = split
        self._image_size = image_size
        self._transform = transform
        if loader is None:
            self._loader = lambda p: default_image_loader(p, image_size)
        else:
            self._loader = loader

        # Stable identity integer mapping (sorted for reproducibility).
        unique_ids = sorted({r.identity for r in self._records})
        self._identity_to_int: dict[str, int] = {ident: idx for idx, ident in enumerate(unique_ids)}
        self._int_to_identity: tuple[str, ...] = tuple(unique_ids)

    # -- introspection -------------------------------------------------

    @property
    def split(self) -> str:
        return self._split

    @property
    def num_classes(self) -> int:
        return len(self._int_to_identity)

    @property
    def records(self) -> Sequence[ImageRecord]:
        return self._records

    @property
    def identity_to_int(self) -> dict[str, int]:
        return dict(self._identity_to_int)

    def identity_for(self, idx: int) -> str:
        return self._int_to_identity[idx]

    # -- Dataset protocol ---------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, ImageRecord]:
        record = self._records[index]
        tensor = self._loader(Path(record.path))
        if self._transform is not None:
            tensor = self._transform(tensor)
        identity_int = self._identity_to_int[record.identity]
        return tensor, identity_int, record
