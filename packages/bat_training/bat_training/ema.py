"""Exponential Moving Average of model parameters.

Used by :class:`EmbeddingTrainer` (default ``decay=0.999``) and *disabled*
by default for :class:`PairTrainer` (the TF Siamese baseline did not use
an EMA, so leaving it off preserves parity).

The implementation tracks a shadow copy of every floating-point
parameter. Buffers (e.g. BatchNorm running stats) are kept in sync via a
plain copy because they are not gradient-updated. Calling
:meth:`apply_to` swaps the EMA weights into the model in-place;
:meth:`restore` puts the original weights back. The standard usage in a
trainer is::

    ema = ExponentialMovingAverage(model, decay=0.999)
    ...
    optimizer.step()
    ema.update(model)
    ...
    # at eval time:
    ema.apply_to(model)
    eval_metrics = run_eval(model)
    ema.restore(model)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import torch
    from torch import nn


class ExponentialMovingAverage:
    """Track an EMA over a model's parameters.

    Args:
        model: The model whose parameters to mirror.
        decay: EMA decay; ``shadow = decay * shadow + (1 - decay) * param``.
            ``decay = 0.999`` matches the common face-recognition default.
        device: Optional device for the shadow copies. Defaults to the
            parameters' device.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1]; got {decay}")
        self.decay = float(decay)
        self._device = device

        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not param.dtype.is_floating_point:
                continue
            tensor = param.detach().clone()
            if device is not None:
                tensor = tensor.to(device)
            self._shadow[name] = tensor

    @property
    def shadow(self) -> dict[str, torch.Tensor]:
        return self._shadow

    def update(self, model: nn.Module) -> None:
        """Update the shadow copy from ``model``'s current parameters."""
        for name, param in model.named_parameters():
            if name not in self._shadow:
                continue
            shadow = self._shadow[name]
            new = param.detach()
            if shadow.device != new.device:
                new = new.to(shadow.device)
            shadow.mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        """Replace ``model``'s parameters with their EMA values.

        Stashes the previous values internally so :meth:`restore` can put
        them back. Calling :meth:`apply_to` twice without a
        :meth:`restore` in between will overwrite the backup.
        """
        self._backup = {}
        for name, param in model.named_parameters():
            if name not in self._shadow:
                continue
            self._backup[name] = param.detach().clone()
            shadow = self._shadow[name]
            with _no_grad():
                param.data.copy_(shadow.to(param.device))

    def restore(self, model: nn.Module) -> None:
        """Restore the parameters saved by the last :meth:`apply_to`."""
        if not self._backup:
            return
        for name, param in model.named_parameters():
            if name not in self._backup:
                continue
            with _no_grad():
                param.data.copy_(self._backup[name].to(param.device))
        self._backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialise the shadow tensors for checkpointing."""
        return {name: t.detach().clone() for name, t in self._shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore from a previously-saved state dict."""
        self._shadow = {name: t.detach().clone() for name, t in state.items()}
        self._backup = {}


def _no_grad():  # pragma: no cover -- trivial wrapper
    import torch

    return torch.no_grad()


__all__ = ["ExponentialMovingAverage"]
