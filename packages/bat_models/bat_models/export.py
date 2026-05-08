"""ONNX export utility on top of ``torch.onnx``.

Prefers ``torch.onnx.dynamo_export`` (PyTorch 2.x, traces via TorchDynamo) and
falls back to the legacy tracer-based ``torch.onnx.export`` if the dynamo
exporter is unavailable on the installed Torch build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: Sequence[int],
    *,
    opset_version: int = 17,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> Path:
    """Export ``model`` to ONNX at ``output_path``.

    Parameters
    ----------
    model:
        A PyTorch ``nn.Module``. The caller is responsible for switching it
        into ``eval`` mode if required; we do so defensively here as well.
    output_path:
        Destination path (``.onnx``).
    input_shape:
        Shape of a single sample input (including batch dim), e.g.
        ``(1, 3, 112, 112)``.
    opset_version:
        ONNX opset to target when falling back to the legacy exporter.
    input_names / output_names:
        Optional ONNX graph IO names.

    Returns
    -------
    Path
        Resolved path of the written ``.onnx`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    dummy = torch.zeros(*input_shape, dtype=torch.float32)

    dynamo_export = getattr(torch.onnx, "dynamo_export", None)
    if dynamo_export is not None:
        try:
            program = dynamo_export(model, dummy)
            program.save(str(output_path))
            return output_path
        except Exception:
            # Fall through to legacy exporter.
            pass

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=list(input_names) if input_names else ["input"],
        output_names=list(output_names) if output_names else ["output"],
        dynamic_axes={
            (input_names[0] if input_names else "input"): {0: "batch"},
            (output_names[0] if output_names else "output"): {0: "batch"},
        },
    )
    return output_path
