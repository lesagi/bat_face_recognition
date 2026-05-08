"""ONNX export round-trip on a tiny model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bat_models.export import export_onnx  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 8)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_export_onnx_writes_file_and_inference_matches() -> None:
    onnx = pytest.importorskip("onnx")

    model = _Tiny().eval()
    input_shape = (1, 3, 16, 16)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tiny.onnx"
        written = export_onnx(model, out, input_shape=input_shape)
        assert written.exists()
        assert written.stat().st_size > 0

        graph = onnx.load(str(written))
        onnx.checker.check_model(graph)

        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(written), providers=["CPUExecutionProvider"])
        dummy = torch.zeros(*input_shape).numpy()
        outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
        assert outputs[0].shape == (1, 8)
