# TOOLING.MAKE-PREK — handoff

Independent review: not run (0/3). The skill needs the maintainer's reviewer
workspace (`/Users/wamster3/dev/survjax/review`, `roles/reviewer-settings.json`),
which does not exist in this contributor checkout.

Plan: [PLAN.md](PLAN.md). Source: issue #2 and the maintainer's reply.

## Read first, in order

1. `.github/workflows/ci.yml`: `plan` now checks out and calls
   `scripts/ci_plan.sh`; new `hooks` job; `check` needs `[plan, hooks]` and
   calls `scripts/ci_branch_refs.sh` and `scripts/ci_environment.py`.
2. The three `scripts/ci_*` files: verbatim moves of the former inline blocks,
   except `ci_environment.py` also prints the wheel path and writes
   `$GITHUB_ENV` only when it is set (so it runs locally).
3. `.github/workflows/release.yml`: the `git branch main origin/main` copy is
   replaced by `scripts/ci_branch_refs.sh` (no-op branch switch on `release`).
4. `Makefile`, `.pre-commit-config.yaml`, `pyproject.toml`/`uv.lock`,
   `CONTRIBUTING.md`, `CONTRIBUTING_DOCS.md` (one trailing blank line).

## Invariants

- `scripts/check.sh` is unchanged and remains the only gate; nothing in CI or
  in `check.sh` depends on `make` (no GNU make on the `windows-2022` image).
- The hooks job and `make hooks` run the prek locked in `uv.lock`; the mypy
  hook is still the local `uv run --no-sync mypy src`.
- `uv.lock` diff only removes pre-commit and its exclusive dependencies
  (cfgv, distlib, filelock, identify, nodeenv, python-discovery, virtualenv)
  and adds prek 0.5.10; no other version moves.
- `ci_branch_refs.sh` keeps the guard semantics: local `main` from
  `origin/main`; on PRs the merge commit is named after `head_ref`, or
  `ci-pr-main` for a source branch named `main`.

## Evidence

No library behaviour changes, so no new tests. Checks run:

- `uv run prek run --all-files --show-diff-on-failure`: all 11 hooks passed
  (before the exclude: end-of-file-fixer failed on 11 files, see PLAN).
- `make` (help), `make -n` of every wrapper target, `make lint typecheck`:
  ruff format/check clean, mypy "no issues found in 25 source files".
- `scripts/ci_plan.sh` with simulated `$GITHUB_OUTPUT`: `push` gives
  `run=true`; `schedule` with a stub `gh` returning the same SHA gives
  `run=false` plus the summary line; a different SHA gives `run=true`.
- `scripts/ci_branch_refs.sh` in throwaway clones without a local `main`:
  `pull_request` from `tooling-x` → HEAD `tooling-x`, `main` at origin/main;
  from `main` → `ci-pr-main`; `release` → `main` created, HEAD detached.
- `scripts/ci_environment.py` in a temp dir with one wheel in `dist/`:
  prints environment, CPU device check passes, writes `SURVJAX_WHEEL=...`.
- `actionlint` on `ci.yml` and `release.yml`: clean.
- `scripts/check.sh --fast` (macOS arm64, working tree of `56b5b35`):
  "check: all gates passed (baseline=all, features=observation.censoring,
  observation.truncation,observation.traced_metadata,family.lognormal,
  family.loglogistic) (fast: slow tests skipped)"; last spec-test process
  4595 passed, 95 skipped. The full gate (`scripts/check.sh`) was not run.

Not verified: the workflows on GitHub Actions (hooks job, Windows weekly cell).

## Unsure / for the maintainer

- `CLAUDE.md` still lists `pre-commit` under dev deps; it is protected and
  needs a `meta-*` branch. Spec 10.1 could not be checked (spec symlink target
  absent here).
- Static checks are defined twice (`Makefile` and `check.sh` step 2) by
  design, so the gate stays make-free on Windows.
