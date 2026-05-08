"""Experiment-aware filename / title helpers.

The legacy TF code emitted plots and PDFs with generic, non-discriminative names
(e.g. ``_saliency_integrated_gradients.pdf``), which made it impossible to tell
which model / loss / dataset combination produced a given artifact.  The Phase-1
plan requires every plot title, filename, and axis label to embed the
experiment signature ``{species}_{source}_{background}_{model}_{loss}`` derived
from the Hydra cfg.

This module is the single source of truth for that fix.  ``visualizer.py`` (and
any future plot emitter) **must** route every filename through
:func:`build_filename` and every plot title through
:func:`build_title_suffix` so no artifact escapes with a generic name.
"""

from __future__ import annotations

from typing import Any, Mapping

# Order matters: this is the human- and machine-readable signature.
EXPERIMENT_NAME_FIELDS: tuple[str, ...] = (
    "species",
    "source",
    "background",
    "model",
    "loss",
)


def _coerce_value(value: Any) -> str:
    """Reduce a cfg value (str, int, bool, None, nested dict, ...) to a slug token."""

    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        token = str(value).strip()
        return token or "unknown"
    # Hydra often nests ``model: {name: arcface, ...}`` -- prefer ``name`` if present.
    if isinstance(value, Mapping):
        for nested_key in ("name", "_target_", "type", "id"):
            if nested_key in value:
                return _coerce_value(value[nested_key])
        # Fallback: serialise mapping deterministically.
        return "_".join(f"{k}-{_coerce_value(v)}" for k, v in sorted(value.items()))
    # Last resort: best-effort attribute lookup, then repr.
    for attr in ("name", "_target_", "type", "id"):
        if hasattr(value, attr):
            return _coerce_value(getattr(value, attr))
    return str(value).strip() or "unknown"


def _slugify_field(raw: str) -> str:
    """Slugify a single experiment-name field (species/source/etc).

    Lower-cases, keeps alphanumerics, collapses everything else to a dash.
    Used only for the 5 components that appear in the experiment signature.
    """

    cleaned = []
    for ch in raw.strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in ("-", "_", ".") or ch.isspace():
            cleaned.append("-")
        # drop everything else (slashes, colons, etc.)
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"


def _slugify_stem(raw: str) -> str:
    """Slugify a filename *stem* (e.g. ``null_dist_f1``).

    Same as :func:`_slugify_field` except underscores are preserved -- we want
    ``null_dist_f1`` to stay readable and the user explicitly spells stems
    with underscores.
    """

    cleaned = []
    for ch in raw.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("-")
    slug = "".join(cleaned).strip("-_")
    return slug.lower() or "unknown"


def _get(cfg: Any, key: str) -> Any:
    """Look up ``key`` in a Hydra cfg-ish object (Mapping, dataclass, namespace)."""

    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
        # Hydra runs are often nested under ``experiment`` or ``cfg``.
        for parent_key in ("experiment", "cfg", "config"):
            parent = cfg.get(parent_key)
            if isinstance(parent, Mapping) and key in parent:
                return parent[key]
        return None
    # Attribute access (dataclasses, OmegaConf DictConfig in attribute mode, SimpleNamespace).
    if hasattr(cfg, key):
        return getattr(cfg, key)
    for parent_key in ("experiment", "cfg", "config"):
        if hasattr(cfg, parent_key):
            parent = getattr(cfg, parent_key)
            if hasattr(parent, key):
                return getattr(parent, key)
    return None


def extract_components(cfg: Any) -> dict[str, str]:
    """Return slugified ``species/source/background/model/loss`` from a Hydra cfg.

    Any missing field becomes ``"unknown"`` rather than raising -- the downstream
    naming code prefers a degraded name to a crashed plot generator.  Tests
    enforce that ``unknown`` does not appear in production cfgs.
    """

    out: dict[str, str] = {}
    for field in EXPERIMENT_NAME_FIELDS:
        raw = _get(cfg, field)
        if raw is None:
            # Common nested locations: ``data.species``, ``model.name``, ``loss.name``.
            if field == "species":
                raw = _get(_get(cfg, "data"), "species")
            elif field == "source":
                raw = _get(_get(cfg, "data"), "source")
            elif field == "background":
                raw = _get(_get(cfg, "data"), "background")
            elif field == "model":
                raw = _get(_get(cfg, "model"), "name")
            elif field == "loss":
                raw = _get(_get(cfg, "loss"), "name")
        out[field] = _slugify_field(_coerce_value(raw))
    return out


def experiment_name(cfg: Any) -> str:
    """Return ``'rousettus_video_random_arcface_arcface'`` (or similar) from cfg.

    Concretely, joins ``species_source_background_model_loss`` from the Hydra
    cfg with underscores after slugifying each field.  Used as the primary
    discriminator in every output filename and plot title.
    """

    components = extract_components(cfg)
    return "_".join(components[field] for field in EXPERIMENT_NAME_FIELDS)


def build_filename(stem: str, cfg: Any, suffix: str = ".png") -> str:
    """Compose ``{stem}__{experiment_name}{suffix}``.

    Examples
    --------
    >>> build_filename("null_dist_f1", cfg)
    'null_dist_f1__rousettus_video_random_arcface_arcface.png'
    """

    if not stem:
        raise ValueError("stem must be non-empty")
    name = experiment_name(cfg)
    if not suffix.startswith("."):
        suffix = "." + suffix
    return f"{_slugify_stem(stem)}__{name}{suffix}"


def build_title_suffix(cfg: Any) -> str:
    """Return the trailing ``[species/source/background | model/loss]`` for plot titles.

    A trailing bracketed suffix is unobtrusive but makes every PDF page
    self-identifying when leafed through a stack of unrelated reports.
    """

    c = extract_components(cfg)
    return (
        f"[{c['species']}/{c['source']}/{c['background']} | "
        f"{c['model']}/{c['loss']}]"
    )


def build_axis_label(base: str, cfg: Any) -> str:
    """Decorate an axis label with the experiment short-name in parentheses.

    Most axis labels (e.g. ``"F1"``) are too short to be self-identifying.  We
    append a parenthetical ``({experiment_name})`` so log-style figure dumps
    can be cross-referenced by reading the axis alone.
    """

    return f"{base} ({experiment_name(cfg)})"


__all__ = [
    "EXPERIMENT_NAME_FIELDS",
    "build_axis_label",
    "build_filename",
    "build_title_suffix",
    "experiment_name",
    "extract_components",
]
