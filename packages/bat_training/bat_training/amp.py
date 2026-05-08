"""Mixed-precision context.

Most of the heavy lifting (precision flag, scaler) is delegated to
``accelerate.Accelerator``: see :class:`PairTrainer` and
:class:`EmbeddingTrainer`, which construct ``Accelerator(mixed_precision=...)``.

This module provides a tiny, *Accelerate-free* fallback context for unit
tests and for users running raw PyTorch without Accelerate. Constructing
:class:`AMPContext` is cheap and works on CPU (where it is a no-op).

The default precision is ``bf16`` because Ampere+ GPUs handle it without
loss-scaling and bf16 is robust to gradient overflow. Users who want
``fp16`` automatic loss scaling should pass ``precision="fp16"``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Literal

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import torch

Precision = Literal["no", "fp16", "bf16"]


class AMPContext:
    """Standalone mixed-precision context.

    Args:
        precision: ``"no"`` (default ``"bf16"`` if a CUDA device is
            available), ``"fp16"``, or ``"bf16"``. ``"no"`` makes every
            method a no-op (useful for tests on CPU).
        device_type: ``"cuda"`` (default), ``"cpu"``, or ``"mps"``.
        enabled: If ``False``, all methods are no-ops regardless of the
            precision argument.
    """

    def __init__(
        self,
        precision: Precision = "bf16",
        device_type: str = "cuda",
        enabled: bool = True,
    ) -> None:
        self.precision = precision
        self.device_type = device_type
        self.enabled = bool(enabled and precision != "no")

        # GradScaler is only needed for fp16; bf16 has the dynamic range.
        self._use_scaler = self.enabled and precision == "fp16"
        self._scaler: object | None = None
        if self._use_scaler:
            try:
                import torch

                # torch>=2.4 supports torch.amp.GradScaler with a device_type;
                # fall back to torch.cuda.amp.GradScaler on older versions.
                if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                    self._scaler = torch.amp.GradScaler(device_type)
                else:  # pragma: no cover - older torch path
                    self._scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - cpu / no-cuda fallback
                self._use_scaler = False
                self._scaler = None

    @property
    def scaler(self) -> object | None:
        return self._scaler

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Enter a :func:`torch.amp.autocast` context.

        On ``precision="no"`` (or ``enabled=False``) this is a no-op
        ``with``-block.
        """
        if not self.enabled:
            yield
            return

        import torch

        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        with torch.amp.autocast(device_type=self.device_type, dtype=dtype):
            yield

    def backward(self, loss: "torch.Tensor") -> None:
        """Run backward; uses :class:`torch.cuda.amp.GradScaler` for fp16."""
        if self._use_scaler and self._scaler is not None:
            self._scaler.scale(loss).backward()  # type: ignore[attr-defined]
            return
        loss.backward()

    def step(self, optimizer: object) -> None:
        """Step the optimizer (unscales fp16 gradients first)."""
        if self._use_scaler and self._scaler is not None:
            self._scaler.step(optimizer)  # type: ignore[attr-defined]
            self._scaler.update()  # type: ignore[attr-defined]
            return
        optimizer.step()  # type: ignore[attr-defined]


__all__ = ["AMPContext", "Precision"]
