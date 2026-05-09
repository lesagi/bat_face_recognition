"""Optuna :class:`Study` factory.

The default storage is a SQLite file (``optuna_studies.db``) sitting
next to the ``mlruns/`` tracking directory so the two artifact stores
live in the same project root. Passing ``storage=None`` falls back to
Optuna's in-memory backend (handy for tests).

The default pruner is the median pruner, matching the Phase 2 plan
(`bat_sweeps` row): trials whose intermediate values fall below the
historical median get pruned mid-fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import optuna
    from optuna.pruners import BasePruner


Direction = Literal["maximize", "minimize"]
PrunerName = Literal["median", "none"]

DEFAULT_STORAGE = "sqlite:///optuna_studies.db"


def _build_pruner(name: PrunerName | str) -> BasePruner:
    """Resolve a pruner name to an Optuna :class:`BasePruner` instance.

    ``"median"`` (the plan's choice) maps to :class:`MedianPruner` with
    Optuna's defaults. ``"none"`` maps to :class:`NopPruner`.
    """
    import optuna

    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"unknown pruner {name!r}; choose 'median' or 'none'")


def make_study(
    name: str,
    storage: str | None = DEFAULT_STORAGE,
    direction: Direction = "maximize",
    pruner: PrunerName | str = "median",
    *,
    load_if_exists: bool = True,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> optuna.Study:
    """Create or load an Optuna study with the requested configuration.

    Args:
        name: Study name (used as the storage key).
        storage: SQLAlchemy URL or ``None`` for in-memory storage.
            Defaults to :data:`DEFAULT_STORAGE`.
        direction: ``"maximize"`` (default; matches ``val/roc_auc``) or
            ``"minimize"`` (e.g. for ``val/loss``).
        pruner: One of ``"median"`` (Phase 2 default) or ``"none"``.
        load_if_exists: If ``True`` (default), reuse a study with the
            same name if present in storage; otherwise raise.
        sampler: Optional explicit Optuna sampler. ``None`` falls back to
            the default :class:`TPESampler`.

    Returns:
        An :class:`optuna.Study` ready to receive ``study.optimize(...)``.
    """
    import optuna

    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction=direction,
        pruner=_build_pruner(pruner),
        sampler=sampler,
        load_if_exists=load_if_exists,
    )


__all__ = ["DEFAULT_STORAGE", "Direction", "PrunerName", "make_study"]
