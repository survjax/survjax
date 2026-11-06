# ALIGN.0 — Restore moved-worktree launchers

Completed 2026-11-06. Environment-only repair; independent review is not
applicable to this local environment repair and planning/evidence record.
No code, tests, docs pages, dependencies or protected paths changed.

Read first: PLAN.md's ALIGN.0 completion record, then this verification record.
Starting commit: 94a4474; previous docs step/worktree starting point: 3cb4cd7.
The prior DOCS.6 review remains in git history at 94a4474:REVIEW.md.

## Invariants

- Recreate .venv at the current worktree path from the unchanged uv.lock.
- Preserve the original environment as a reversible backup until verified.
- Preserve every installed package version and the editable source location.
- Do not infer numerical correctness from environment/launcher checks.

Original pytest/mypy shebangs named the nonexistent old worktree
dev-spectests-rmst. Moved .venv to
/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-align0-u4rcxltr/.venv,
then ran `uv sync --locked --group test`: installed 34 packages successfully.
Backup retained; no deletion was needed.

## Checks and outputs

- Compared `uv pip freeze --python <backup>/bin/python` with `uv pip freeze`:
  exact equality. Python 3.12.9, JAX 0.11.2, NumPy 2.5.3, pytest 9.1.1.
- Asserted pytest, mypy and pre-commit shebangs all equal
  `#!/Users/wamster3/dev/survjax/rmsttest/.venv/bin/python`: passed.
- `uv run pytest --version`: pytest 9.1.1.
- `uv run mypy src`: Success: no issues found in 22 source files.
- `uv run ruff format --check .`: 173 files already formatted.
- `uv run ruff check .`: All checks passed!
- `JAX_ENABLE_X64=1 uv run --group test pytest -n 6 tests/unit/test_import.py -q`:
  3 passed in 1.93s.
- `JAX_ENABLE_X64=0 uv run --group test pytest -n 6 tests/unit/test_import.py -q`:
  3 passed in 1.93s.
- `git diff --exit-code -- pyproject.toml uv.lock`: exit 0, unchanged.
- Imported survjax from this worktree's src/survjax/__init__.py.
- `git diff --check`: passed before commit.

The import smoke tests exercise the local package after environment creation
(spec 10.1/10.3); no new tests were added for this reversible repair.
The numerical fast/full gates and docs builds were not rerun. Known A11
constructor and acceptance discrepancies from inspection remain unchanged.
No uncertainty remains about the repaired launcher paths; this does not settle
D-042/D-043/D-044. Stop here: ALIGN.1 and subsequent substeps remain pending.
