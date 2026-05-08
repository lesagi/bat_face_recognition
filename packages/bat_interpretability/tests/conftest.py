"""Test configuration for ``bat_interpretability``.

Each test module top-loads ``pytest.importorskip("torch")`` (or ``umap``,
``sklearn``) so that test collection survives on machines that don't have
the heavy deps installed. We do *not* skip globally because some tests
(e.g. the dispatcher unit test) work without torch.
"""
