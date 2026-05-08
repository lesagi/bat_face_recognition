"""Tests for the HP audit allowlist."""

from __future__ import annotations

from bat_tracking.hp_audit import DROP_PREFIXES, KEEP, filter_params


def test_keep_contains_plan_required_keys() -> None:
    """Sanity-check the allowlist matches the plan's "Retained" section."""
    required = {
        "model_family",
        "model_arch",
        "embedding_dim",
        "loss_type",
        "optimizer",
        "lr",
        "weight_decay",
        "ema_decay",
        "gradient_accumulation_steps",
        "batch_size",
        "epochs",
        "early_stop_patience",
        "early_stop_monitor",
        "class_balancing_scheme",
        "ens_beta",
        "global_distribution_strategy",
        "split_mode",
        "split_seed",
        "val_fraction",
        "test_fraction",
        "species",
        "background",
        "data_source",
        "augmentation_preset",
        "manifest_hash",
        "final_threshold",
        "final_youden_j",
    }
    assert required <= KEEP, f"missing from KEEP: {required - KEEP}"


def test_drop_prefixes_contains_plan_banned_keys() -> None:
    """The plan-banned constants must be in DROP_PREFIXES."""
    banned = {
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        "interpolation",
        "scale_factor",
        "normalize",
        "pair_mode",
        "anchor_negative_balance",
        "anchor_target_ratio",
        "negative_pair_combination",
    }
    assert banned <= DROP_PREFIXES, f"missing from DROP_PREFIXES: {banned - DROP_PREFIXES}"


def test_filter_keeps_allowed_keys() -> None:
    params = {
        "model_family": "embedding",
        "lr": 1e-3,
        "batch_size": 32,
        "manifest_hash": "abc",
    }
    out = filter_params(params)
    assert out == params


def test_filter_drops_disallowed_keys() -> None:
    params = {
        "lr": 1e-3,  # kept
        "mlflow_tracking_uri": "http://x",  # dropped
        "interpolation": "bilinear",  # dropped
        "scale_factor": 1.0,  # dropped
        "normalize": True,  # dropped
        "pair_mode": "anchor",  # dropped
        "anchor_negative_balance": 0.5,  # dropped
    }
    out = filter_params(params)
    assert out == {"lr": 1e-3}


def test_filter_drops_per_class_weight_arrays() -> None:
    params = {
        "lr": 1e-3,
        "class_weights_raw": [0.1, 0.2, 0.3],
        "per_class_weights": [0.1, 0.2],
        "class_weights.W": 0.7,
    }
    out = filter_params(params)
    assert out == {"lr": 1e-3}


def test_filter_drops_sample_image_dimensions() -> None:
    params = {
        "lr": 1e-3,
        "sample_image_dimensions": [[224, 224], [224, 224]],
        "image_dimensions": [224, 224],
    }
    out = filter_params(params)
    assert out == {"lr": 1e-3}


def test_filter_keeps_loss_params_subkeys() -> None:
    """`loss_params.margin`, `loss_params.scale`, etc. flow through."""
    params = {
        "loss_params.margin": 0.5,
        "loss_params.scale": 64,
        "loss_params.alpha": 0.25,
        "loss_params.gamma": 2.0,
        "lr_schedule_params.warmup_epochs": 5,
    }
    out = filter_params(params)
    assert out == params


def test_filter_keeps_best_and_test_metrics() -> None:
    params = {
        "best_f1_value": 0.92,
        "best_f1_epoch": 17,
        "test/roc_auc": 0.95,
        "test/tar_at_far_1e3": 0.81,
    }
    out = filter_params(params)
    assert out == params


def test_filter_unknown_key_dropped() -> None:
    """Allowlist is closed by default --- unknown keys are dropped."""
    out = filter_params({"some_random_undocumented_key": 42, "lr": 1e-3})
    assert out == {"lr": 1e-3}


def test_filter_preserves_insertion_order() -> None:
    params = {"manifest_hash": "h", "lr": 1e-3, "epochs": 10, "batch_size": 8}
    out = filter_params(params)
    assert list(out.keys()) == list(params.keys())
