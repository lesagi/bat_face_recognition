"""HP audit allowlist for MLflow parameter logging.

Implements the curated allowlist from the refactor plan's "HP audit" section.
Only the keys in :data:`KEEP` (or those matching wildcard prefixes from
:data:`KEEP_PREFIXES`) are forwarded to MLflow's ``log_param`` API; everything
else (constants, derivable values, family-irrelevant settings) is dropped.

Banned categories (see plan):
    - ``mlflow_tracking_uri``, ``mlflow_experiment_name`` (implicit in run location).
    - Per-class weight raw arrays (replaced by aggregated stats).
    - Sample image dimension list (replaced by manifest profile artifact).
    - ``interpolation``, ``scale_factor``, ``normalize`` (preprocessing constants).
    - ``pair_mode``, ``anchor_negative_balance``, ``anchor_target_ratio``,
      ``negative_pair_combination`` for embedding-family runs (filtered upstream
      by the trainer; we still include them in :data:`DROP_PREFIXES` to be
      explicit).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Allowed exact-match keys.
# ---------------------------------------------------------------------------
KEEP: frozenset[str] = frozenset(
    {
        # Model identity.
        "model_family",
        "model_arch",
        "backbone_dilated",
        "embedding_dim",
        # Loss configuration.
        "loss_type",
        # Optimization.
        "optimizer",
        "lr",
        "weight_decay",
        "ema_decay",
        "gradient_accumulation_steps",
        # Training schedule.
        "batch_size",
        "epochs",
        "early_stop_patience",
        "early_stop_monitor",
        # Class balancing.
        "class_balancing_scheme",
        "ens_beta",
        "global_distribution_strategy",
        # Split configuration.
        "split_mode",
        "split_seed",
        "val_fraction",
        "test_fraction",
        # Experiment metadata.
        "species",
        "background",
        "data_source",
        "augmentation_preset",
        "manifest_hash",
        # Final evaluation thresholds.
        "final_threshold",
        "final_youden_j",
    }
)

# ---------------------------------------------------------------------------
# Allowed key prefixes. ``filter_params`` retains any key starting with one of
# these, so structured groups like ``loss_params.margin`` are passed through
# without enumerating every leaf.
# ---------------------------------------------------------------------------
KEEP_PREFIXES: frozenset[str] = frozenset(
    {
        # Loss-family-relevant sub-params: margin, scale, alpha, gamma, ...
        "loss_params.",
        # Schedule sub-params: warmup_epochs, T_max, min_lr, ...
        "lr_schedule_params.",
        # Best-of-training summary: best_f1_value, best_f1_epoch, ...
        "best_",
        # Test metrics promoted to params for sortable run tables. The trainer
        # emits these with a ``test/`` prefix; the audit allows them through.
        "test/",
        # Runtime re-split provenance: seed, size mode, realised identity and
        # image counts per split, identity-assignment digest, source manifest
        # hash. These are what makes a k-fold run reproducible from MLflow
        # alone, so they must not be dropped.
        "split.",
    }
)

# ---------------------------------------------------------------------------
# Explicitly banned prefixes / exact keys. Anything matching here is dropped
# even if it would otherwise pass via ``KEEP_PREFIXES``. This is the audit's
# safety net for the cases the plan calls out by name.
# ---------------------------------------------------------------------------
DROP_PREFIXES: frozenset[str] = frozenset(
    {
        # MLflow internals: tracking server URI, experiment name --- both are
        # implicit in run location and shouldn't pollute params.
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        # Raw per-class weight arrays. We expect aggregated stats logged
        # separately under ``class_weights_stats.{min,max,mean,std}``.
        "class_weights_raw",
        "class_weights.",
        "per_class_weights",
        # Sample image dimension list (replaced by manifest profile artifact).
        "sample_image_dimensions",
        "image_dimensions",
        # Preprocessing-version constants.
        "interpolation",
        "scale_factor",
        "normalize",
        # Pair-only knobs that don't apply to embedding-family runs.
        "pair_mode",
        "anchor_negative_balance",
        "anchor_target_ratio",
        "negative_pair_combination",
    }
)


def _is_dropped(key: str) -> bool:
    """Return True if *key* matches one of the banned exact keys / prefixes."""
    for banned in DROP_PREFIXES:
        if banned.endswith("."):
            if key.startswith(banned):
                return True
        else:
            if key == banned or key.startswith(banned + "."):
                return True
    return False


def _is_kept(key: str) -> bool:
    """Return True if *key* is allowed by KEEP / KEEP_PREFIXES."""
    if key in KEEP:
        return True
    return any(key.startswith(p) for p in KEEP_PREFIXES)


def filter_params(params: dict[str, Any]) -> dict[str, Any]:
    """Filter *params* down to the curated MLflow-loggable subset.

    A key is kept when it appears in :data:`KEEP` or starts with a prefix from
    :data:`KEEP_PREFIXES`, **and** does not match :data:`DROP_PREFIXES`.

    Args:
        params: Flat dict of HP key -> value. Nested dicts are not supported;
            callers should flatten with ``hydra``-style dotted keys first.

    Returns:
        A new dict containing only the allowed entries, in insertion order.
    """
    return {k: v for k, v in params.items() if _is_kept(k) and not _is_dropped(k)}


__all__ = ["KEEP", "KEEP_PREFIXES", "DROP_PREFIXES", "filter_params"]
