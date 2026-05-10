# CODEX.md

General handoff for Codex agents working in this repository.

## PR closing requests

When asked to close PRs for this project, first verify the intent against the current plan instead of blindly closing open PRs:

- Run `gh pr list --state open` and inspect each open PR with `gh pr view` plus `gh pr diff`.
- Compare the PR purpose with `docs/PHASE_4_CHECKLIST.md`, `docs/REFACTOR_STATUS.md`, and the current `main` branch.
- If a PR is clean, plan-aligned, and still useful, merge it to close it rather than closing it unmerged.
- Close unmerged only when the PR is clearly obsolete, duplicate, unsafe, or contradicted by the current plan.
- After any merge or close, verify `gh pr list --state open` again and report the final PR state.
