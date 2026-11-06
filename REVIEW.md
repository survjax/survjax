Independent review: round 2/3

# RELEASE.2 — Prepare package release metadata

## Read first

1. PLAN.md RELEASE.2: metadata scope, proposed version and remaining steps.
2. pyproject.toml project fields: Markdown README, alpha/tested-Python
   classifiers and repository/documentation/issues URLs.
3. README.md diff: only the two guide link destinations change, making the
   description portable to PyPI. No prose or scope claims changed.
4. tests/install/test_wheel.py: new built-wheel metadata test and existing
   out-of-tree runtime isolation/version/license checks.

Spec revision 41: 10.1–10.3, 12.2 and 13. No protected path, dependency,
API, numerical, docs page/navigation or specification changes.

## Invariants and decisions

- README is the package description; the backend infers text/markdown from
  its .md extension. Built metadata must carry that exact text.
- Project URLs use the supplied GitHub repository and existing intended docs
  address. No ownership, public-access or hosted-build claim follows from URLs.
- Alpha classifier matches the README's existing maturity statement; only
  tested Python 3.12 is classified. No untested Python/platform/GPU matrix.
- Runtime remains jax>=0.11.2 and numpy>=2.5.3; Apache-2.0 is unchanged.
  Lockfile unchanged; no new build hooks or rendering dependencies.
- Development version remains 0.0.1.dev0. Proposed release version 0.0.1a1,
  Git tag v0.0.1a1, RTD tag version v0.0.1a1. Before release artifacts/tagging,
  update project version, refresh lock and update installed-version assertion
  together. No tags or version-specific URLs created by preparation.
- Historical RELEASE.1 planning remains in PLAN.md, with hosted/Linux checks
  still open. Prior review remains in git history and its independent file.

## Sources checked (2026-11-06)

- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/:
  README description, classifiers and project URL tables.
- https://pypi.org/classifiers/: Development Status :: 3 - Alpha and
  Programming Language :: Python :: 3.12 are recognized classifiers.
  Keep existing SPDX license expression; no deprecated license classifier.

## Verification

- uv lock --check: exit 0, 166 packages resolved; uv.lock unchanged.
- uv build --out-dir /tmp/survjax-release2-dist: exit 0; source archive and
  wheel built successfully, wheel built from source archive by uv.
  Both filenames use 0.0.1.dev0.
- Standard-library archive/email metadata audit: exit 0; README included
  verbatim in sdist, both descriptions equal README bytes; Name/Version,
  Requires-Python/Requires-Dist, URLs, classifiers, license and content type
  match between wheel and source archive. Runtime requirements reported
  ['jax>=0.11.2', 'numpy>=2.5.3'].
- JAX_ENABLE_X64=1 uv run --group test pytest -n 6 -q tests/install/test_wheel.py:
  4 passed in 42.62s.
- JAX_ENABLE_X64=0 uv run --group test pytest -n 6 -q tests/install/test_wheel.py:
  4 passed in 42.03s.
  These include installation into fresh out-of-tree environments, core imports
  without demo dependencies, parameter derivatives and external-family typing.
  New artifact test covers exact description/content type, release URLs,
  alpha/tested-Python classifiers and absolute Markdown link destinations.
- First scripts/check.sh --fast --through A10 attempt stopped at Ruff I001
  (new re import out of order). uv run ruff check --fix corrected only that
  import; subsequent full fast-gate result recorded below.

- scripts/check.sh --fast --through A10 rerun: exit 0, all gates passed.
  Ruff: 173 files already formatted, all checks passed. mypy: 22 files clean.
  Implementer float64 1060 passed / 8 skipped; float32 1065 / 3.
  Source doctests: 4 passed in each mode. Spec through A10: float64
  2149 passed / 27 skipped; float32 2156 / 20. Slow tests deselected by fast
  gate; touched install tests were separately run in full as required.
- git diff --check: exit 0. No lockfile or runtime dependency changes.

## Limitations and remaining substeps

No PyPI/TestPyPI upload, public clean-install, PyPI-rendered README inspection,
account changes, tag, push, merge or release. No Twine dependency added just for
rendering checks. README's intended RTD links are not yet verified publicly.
No Sphinx build in this metadata step: source pages, cells and build config
unchanged. RELEASE.1's macOS docs evidence and open Linux/hosting checks remain.
No A11/A12 acceptance claim; their independent integration has separate ownership.

Remaining RELEASE.3: narrowly scoped explicit-release Trusted Publishing
workflow and local artifact validation. RELEASE.4: operator docs covering
publisher/account fields and consistent version/tag/docs/public checks.
Final release version change must accompany the operator's release commit;
0.0.1a1 remains the proposal, not an existing release.

## Commit trace

Previous completed step / clean starting SHA:
ac50796fc82ff087df3c8973724992a4ec2e2eb5 (RELEASE.1 review fix).
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
This commit completes RELEASE.2; independent review follows, then stop.

## Round 1 response

F-1 addressed in PLAN.md: assign changelog, citation and contribution guidance
to separate reviewed RELEASE.5a–5c docs steps. Treat them as prerequisites for
any public alpha upload under spec 10.1/13; infrastructure preparation alone
is not completion of the release gate. Do not implement those topics here or
close unfinished work belonging to other agents.

N-1 (optional) retained: the inline Markdown destination check covers today's
README syntax, not every Markdown/HTML link form. Extending it would add a
partial link parser unrelated to this correction. If README syntax changes,
revisit the guard; no claim of a general README link validator.
N-2: add an explicit RELEASE.4 handoff item for moving /en/latest/ links and
consistent version-specific links if selected before release artifacts build.

Only PLAN.md and this review response changed after round 1. Existing passing
artifact/install/fast-gate results remain applicable; no code, metadata, tests
or lockfile changes. git diff --check passed; no redundant test rerun for this
planning-only follow-up. Independent review round 2 follows the fix-up commit.

Reviewer housekeeping limitation: round 1 reports a prunable mutation-worktree
Git entry after its directory was removed. Left to the reviewer/maintainer;
this agent did not edit reviewer-owned state or prune other worktrees.
