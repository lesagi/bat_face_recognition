"""Title page: experiment name, run id, manifest hash, full Hydra cfg snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from bat_reporting.sections.common import SectionContext, add_heading, add_paragraph, cfg_to_mapping
from bat_stats.naming import experiment_name

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    """Render the title page.

    Layout: experiment name (large) -> run id, manifest hash, timestamp ->
    full cfg snapshot in monospaced YAML -> page break.
    """

    from reportlab.platypus import PageBreak, Preformatted, Spacer

    name = experiment_name(data.cfg)
    add_heading(story, "Bat face recognition: post-training report", ctx, level=1)
    add_heading(story, f"Experiment: {name}", ctx, level=2)
    story.append(Spacer(1, 6))

    add_paragraph(story, f"<b>run_id:</b> {data.run_id}", ctx)
    if data.manifest_hash:
        add_paragraph(story, f"<b>manifest_hash:</b> {data.manifest_hash}", ctx)
    if data.timestamp:
        add_paragraph(story, f"<b>timestamp:</b> {data.timestamp}", ctx)
    story.append(Spacer(1, 12))

    add_heading(story, "Hydra cfg snapshot", ctx, level=2)
    cfg_yaml = _safe_yaml_dump(cfg_to_mapping(data.cfg))
    code_style = ctx.styles["Code"] if "Code" in ctx.styles.byName else ctx.styles["BodyText"]
    story.append(Preformatted(cfg_yaml, code_style))
    story.append(PageBreak())


def _safe_yaml_dump(payload: Any) -> str:
    """YAML-dump a (possibly nested) cfg, falling back to ``str()`` for exotica."""

    try:
        return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    except (TypeError, ValueError, yaml.YAMLError):
        return str(payload)


__all__ = ["render_section"]
