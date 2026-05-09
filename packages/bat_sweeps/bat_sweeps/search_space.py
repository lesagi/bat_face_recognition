"""Hydra ``sweep:`` search-space spec → Optuna sampler.

The Hydra config carries a flat dict shaped like::

    search_space:
      loss.margin: { type: float, low: 0.3, high: 0.6 }
      loss.scale:  { type: float, low: 32.0, high: 96.0 }
      trainer.lr:  { type: loguniform, low: 1.0e-3, high: 3.0e-1 }
      model.head.size: { type: int, low: 256, high: 1024, step: 128 }
      trainer.optimizer: { type: categorical, choices: [adam, sgd] }

:func:`parse_search_space` returns a callable that consumes an
:class:`optuna.trial.Trial` and produces a flat overrides dict whose
**keys preserve the dotted path** (``"loss.margin": 0.42``). The merge
into a ``base_cfg`` is done elsewhere (see :mod:`bat_sweeps.objective`).

Supported ``type`` values:

* ``float``       — ``trial.suggest_float(low, high, step=...)``
* ``int``         — ``trial.suggest_int(low, high, step=...)``
* ``loguniform``  — ``trial.suggest_float(low, high, log=True)``
* ``categorical`` — ``trial.suggest_categorical(choices)``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from optuna.trial import Trial


SearchSpaceSpec = dict[str, dict[str, Any]]
SamplerFn = Callable[["Trial"], dict[str, Any]]


_SUPPORTED_TYPES = frozenset({"float", "int", "loguniform", "categorical"})


def _validate_param_spec(name: str, spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"search_space[{name!r}] must be a dict; got {type(spec).__name__}")
    if "type" not in spec:
        raise ValueError(f"search_space[{name!r}] missing required 'type' field")
    t = spec["type"]
    if t not in _SUPPORTED_TYPES:
        raise ValueError(
            f"search_space[{name!r}].type={t!r} unsupported; "
            f"choose one of {sorted(_SUPPORTED_TYPES)}"
        )
    if t in {"float", "int", "loguniform"}:
        for k in ("low", "high"):
            if k not in spec:
                raise ValueError(f"search_space[{name!r}] missing required {k!r} for type={t!r}")
    if t == "categorical":
        if "choices" not in spec:
            raise ValueError(f"search_space[{name!r}] missing required 'choices' for categorical")
        if not isinstance(spec["choices"], list | tuple) or len(spec["choices"]) == 0:
            raise ValueError(f"search_space[{name!r}].choices must be a non-empty list")


def _suggest_for(name: str, spec: dict[str, Any], trial: Trial) -> Any:
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=spec.get("step"),
        )
    if t == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step)
    if t == "loguniform":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=True,
        )
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    # _validate_param_spec already rejects this branch; included for safety.
    raise ValueError(f"unsupported search-space type {t!r}")  # pragma: no cover


def parse_search_space(spec: SearchSpaceSpec) -> SamplerFn:
    """Compile a search-space spec into a sampler callable.

    Args:
        spec: Flat dotted-key dict; values are per-param specs (see
            module docstring for the schema).

    Returns:
        A function ``sampler(trial)`` that returns a flat overrides dict
        with the same keys as ``spec``.

    Raises:
        ValueError: if a param spec is malformed.
        TypeError: if ``spec`` is not a dict-of-dicts.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"search_space spec must be a dict; got {type(spec).__name__}")
    for name, param in spec.items():
        _validate_param_spec(name, param)

    # Snapshot to avoid mutation surprises.
    frozen: SearchSpaceSpec = {k: dict(v) for k, v in spec.items()}

    def _sampler(trial: Trial) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for name, param in frozen.items():
            overrides[name] = _suggest_for(name, param, trial)
        return overrides

    return _sampler


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied ``base`` with dotted-key ``overrides`` applied.

    ``overrides`` keys follow the Hydra dotted convention; intermediate
    dicts are created on demand. Pure dict-tree merge -- no Hydra
    dependency.
    """
    import copy

    merged = copy.deepcopy(base)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        cursor: Any = merged
        for part in parts[:-1]:
            if not isinstance(cursor, dict):
                raise TypeError(
                    f"cannot apply override {dotted!r}: ancestor at {part!r} is "
                    f"a {type(cursor).__name__}, expected dict"
                )
            cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise TypeError(
                f"cannot apply override {dotted!r}: final container is "
                f"a {type(cursor).__name__}, expected dict"
            )
        cursor[parts[-1]] = value
    return merged


__all__ = [
    "SamplerFn",
    "SearchSpaceSpec",
    "apply_overrides",
    "parse_search_space",
]
