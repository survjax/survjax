Independent review: round 1/3

# RELEASE.5a — Initial changelog

Independent reviewer: Astra, low reasoning, as authorized for this workstream.
No Claude invocation or protected skill edits.

## Read first

1. PLAN.md RELEASE.5a: unreleased notes for one bounded docs topic.
2. CHANGELOG.md: single prose source for the draft initial-alpha entry.
3. docs/changelog.md: standalone orphan include, no navigation/config changes.
4. README.md, src/survjax/__init__.py, docs/proportional-hazards.md,
   docs/rmst.md and the two use-case pages (unchanged): source for capabilities
   and limitations; docs/releasing.md: pending launch procedure.
5. Current spec revision 46, sections 10.5, 12.2 and 13. No numerical changes.

## Invariants and claims

- No released version, date or published tag is invented. Heading is
  Unreleased; 0.0.1a1 remains a proposal, 0.0.1.dev0 a preparation baseline.
- Notes describe merged/current public exports and executable docs, not code
  on other worktrees. No simulation implementation or docs cleanup imported.
- Distinguish the data representation from supported exact/right likelihoods
  without truncation. Unsupported/invalid rows give NaN; diagnostics matter.
- PH extreme-scaling derivative limits and Weibull/PH-Weibull RMST higher-shape
  and zero-cutoff limitations stay explicit. No generic RMST or later-slice
  distribution/summary/metric/sampling capability is advertised as available.
- Runtime remains JAX/NumPy only. Recorded CPU baseline is Python 3.12,
  JAX 0.11.2, NumPy 2.5.3; no broader tested matrix or GPU certification added.
- Bayesian trial and MNIST tutorials are present in this checkout. Notes claim
  runnable notebooks/examples, not a new fit or independently reproduced full
  neural training/posterior run during this prose step.
- RTD/PyPI infrastructure is locally prepared. Hosting, publication, public
  access, separate demo/docs/acceptance completion and citation/contribution
  guidance are still prerequisites. Through A10 is not A11/A12 evidence.
- Single CHANGELOG.md prose source included at changelog.html; links use moving
  latest docs explicitly. No toctree or other-agent page changes.
- No API, implementation, dependencies, configuration, lockfile, version,
  protected path, spec or remote/account/publication changes.

## Verification

- scripts/check.sh --fast --through A10: exit 0, all gates passed.
  Ruff: 178 files already formatted / all checks passed. mypy: 22 files clean.
  Implementer float64: 1066 passed / 8 skipped; float32: 1071 / 3.
  Source doctests: 4 passed each. Spec float64: 2149 passed / 27 skipped;
  float32: 2156 / 20. Fast gate deselects slow cases; no new tests for prose,
  no redundant full numerical/install-suite run.
- scripts/build_docs.sh directly, with approved escalation for notebook-kernel
  process inspection (known earlier sandbox restriction): both dtype processes,
  forced execution retained, no execution/configuration weakening. Final build
  and rendered include/link evidence recorded below before commit/review.
- First direct docs build: exit 0, both modes succeeded. Clarified the
  risk-score sentence afterwards; reran scripts/build_docs.sh directly on the
  final wording: exit 0, both modes report build succeeded. Existing notebook
  execution remains force/on-error-fatal with both use-case pages executed.
- uv run --locked --group docs python /tmp/survjax-changelog-audit.py:
  both rendered pages' complete normalized prose matches Markdown rendered
  from root CHANGELOG.md; heading/bullet structure, wrapper source-copy bytes
  and search entry pass. No browser visual-layout claim; Furo unchanged.
- uv run --locked --group docs python /tmp/survjax-release-audit.py:
  each site has 25 HTML pages, 1,623 local links/fragments/assets, 14 downloads,
  zero failures; existing downloadable notebooks byte-match sources, use-case
  images/navigation present. Shared sequential execution tree has 18 notebooks
  / 153 code cells; logs confirm fresh execution separately in both modes.
- git diff --check: passed.

## Limits and remaining work

No hosted Linux/RTD run, account/public access, PyPI rendering/upload, OIDC or
public clean-install check. Existing docs builds use warm MNIST cache and
recorded Bayesian draws with their live calculations; no new full demo run.
No overall release-readiness or A11/A12 acceptance claim. Publication remains
separate and gated by the operator handoff and other owners' completion work.

Next docs steps are RELEASE.5b citation and RELEASE.5c contribution guidance.
The changelog is a draft: reconcile it with the final merged release, complete
remaining prerequisites, then record the selected version/date. Standalone
changelog page has no navigation entry; that decision remains with docs owner.

## Commit trace

Previous completed step / clean start:
b0090b6b49c3d59dca6636a4f8d3893699352f96 (RELEASE.4).
Original worktree base: 3cb4cd7fbba3313f00dfabed0028ee6f404384df.
Commit the bounded step, independent Astra review, then stop.
