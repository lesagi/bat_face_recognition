"""bat_sweeps -- Optuna hyperparameter search on the val split.

The sweep runs :class:`bat_training.EmbeddingTrainer` (or
:class:`bat_training.PairTrainer` if the cfg's loss is pair-family) and
optimises a ``val/<target_metric>``. The test split is sacred -- the
objective never calls ``trainer.test()``.

Public API:

* :func:`parse_search_space` / :data:`SearchSpaceSpec` / :func:`apply_overrides`
* :func:`make_study` (default median pruner, sqlite storage)
* :func:`build_objective`, :class:`TrialComponents`
* :class:`OptunaPruningCallback`, :class:`OptunaPruningTracker`
* :func:`run_sweep`
"""

from __future__ import annotations

from bat_sweeps.objective import BuildComponentsFn, ObjectiveFn, TrialComponents, build_objective
from bat_sweeps.pruning_callback import OptunaPruningCallback, OptunaPruningTracker
from bat_sweeps.runner import ChampionLoggerFn, run_sweep
from bat_sweeps.search_space import SamplerFn, SearchSpaceSpec, apply_overrides, parse_search_space
from bat_sweeps.study import DEFAULT_STORAGE, Direction, PrunerName, make_study

__all__ = [
    "DEFAULT_STORAGE",
    "BuildComponentsFn",
    "ChampionLoggerFn",
    "Direction",
    "ObjectiveFn",
    "OptunaPruningCallback",
    "OptunaPruningTracker",
    "PrunerName",
    "SamplerFn",
    "SearchSpaceSpec",
    "TrialComponents",
    "apply_overrides",
    "build_objective",
    "make_study",
    "parse_search_space",
    "run_sweep",
]
