# TOOLING.MAKE-PREK — Makefile, extracted CI code, prek hook runner

Step starting commit: 4d97c943a5570164adf0d3ef83d6748a39e15c54.
Previous step: RELEASE.A3 (last commit on `main` at branch time); original
branch base: same commit. Branch: `tooling-makefile-prek`. No dependencies.

Source: [issue #2](https://github.com/vanamsterdam/survjax/issues/2), items
1, 2 (runner only) and the CI part of the suggested PR order, as accepted in
the [maintainer reply](https://github.com/vanamsterdam/survjax/issues/2#issuecomment-6767165987):
"the Makefile, extracting the inline CI code, and switching to prek while
keeping mypy for now". Out of scope: `ty`, jaxtyping/beartype, list-input
typing, extra ruff rules, notebook formatting, `no-commit-to-branch`, hook
version bumps.

Feature selection: unchanged (`scripts/spec-features.txt`). No library code,
public API, numerical behaviour or tests change.

## Changes

1. `pyproject.toml`, `uv.lock`: dev group swaps `pre-commit` for `prek`
   (same `.pre-commit-config.yaml`, read unchanged by prek). The local
   `mypy-core` hook stays.
2. `Makefile` (new): `help` (default), `setup`, `lint`, `format`,
   `typecheck`, `hooks`, `install-hooks`, `test`, `test-fast`, `check`,
   `check-fast`, `docs`. Each target is one existing command or script;
   nothing moves out of `scripts/check.sh`.
3. `.github/workflows/ci.yml`: inline logic moves to scripts a contributor can
   read and run by name:
   - `scripts/ci_plan.sh`: weekly-skip decision of the `plan` job;
   - `scripts/ci_branch_refs.sh`: local `main` plus PR branch name for the
     protected-path guard (also reused by `release.yml`, which had a copy of
     the `main` half);
   - `scripts/ci_environment.py`: records wheel path and CPU environment.
   The `core.autocrlf` step stays inline: it must run before checkout.
4. CI `hooks` job: `uv sync --locked --group dev`, then
   `uv run prek run --all-files`; the check matrix `needs` it. Uses the
   locked prek rather than `j178/prek-action` so the hook runner version
   lives in `uv.lock` only and the `language: system` mypy hook finds its
   environment.
5. `.pre-commit-config.yaml`: `end-of-file-fixer` also excludes
   `tests/spec/` (read-only) and `plans/` (legacy history). On `main`,
   `prek run --all-files` failed only that hook, on two spec tests, eight
   legacy plans and `CONTRIBUTING_DOCS.md` (extra trailing blank lines);
   pre-commit had only run on staged files. `CONTRIBUTING_DOCS.md` is fixed.
   Chosen by the contributor (juanitorduz) over a changed-files-only job.
6. `CONTRIBUTING.md`: `make help`, `make install-hooks`, `make hooks`.
   No CHANGELOG entry: it records package changes; nothing shipped changes.

## Decisions

- `scripts/check.sh` does not call `make`: it is protected, and the weekly
  Windows cell runs it on `windows-2022`, whose image has no GNU make on
  `PATH` (runner-images Windows2022 README lists MSYS2 "not added to PATH").
  For the same reason the CI matrix keeps calling scripts, not `make`.
  `make lint`/`make typecheck` therefore repeat the three static-check
  commands of `check.sh` step 2.
- `make check` runs `scripts/check.sh` with no `--through`: the gate's
  default is all baseline tests plus completed features; `--through` is only
  for historical diagnostics.

## Not verifiable here

- Spec 10.1 (pre-commit hooks, dev tools) could not be re-read: the `spec`
  symlink target `../design/spec` is absent in this checkout. The maintainer
  approved prek in the issue; whether spec 10.1 or `CLAUDE.md` ("Dev deps:
  `ruff`, `mypy`, `pre-commit`") needs a wording update is left to the
  maintainer (protected paths; a `meta-*` branch would be needed).
- The independent-review skill needs the maintainer's reviewer workspace
  (`/Users/wamster3/dev/survjax/review`, `roles/`), absent here.
- GitHub Actions behaviour (hooks job, `GITHUB_OUTPUT`/`GITHUB_ENV` writes)
  is checked locally with simulated environment files only.

## Docs to-do

None: `docs/` does not mention pre-commit or the CI layout.
