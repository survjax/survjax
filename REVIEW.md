# A7 merge reconciliation and remaining A8 plan — 2026-10-23

Requested: review PLAN.md/REVIEW.md merge conflicts and plan the rest of A8.
A8 parent: `5d46899`; incoming main: `3389ee0`.

## Read in this order

1. PLAN.md, active remaining-A8 section: ordered steps and approval stops.
2. tests/spec/fixtures/README.md: independent schema and protected handoff.
3. Historical sections of PLAN.md: prior A8 and incoming main history retained.
   Earlier review records remain at `5d46899:REVIEW.md` and
   `3389ee0:REVIEW.md`; this file replaces their competing active status.

## Resolution and invariants

- Only PLAN.md and REVIEW.md conflicted. A8 is active; A7 is merged history.
- All cleanly merged files remain exactly as Git merged them, including the
  incoming docs, A7 tests/tolerances, independent A9/A10 tests and sync script.
- No library implementation, R generator, fixture payload, spec source,
  independent test or protected script is edited during this resolution.
- Existing synthetic R references are retained. The next code step exports
  the independent schema from evaluated R objects, without patient data or
  Python-generated reference values.
- A verified candidate stays under tests/fixtures until the maintainer's
  authoritative tester-repo handoff and spec-test sync. No protected-path bypass.
- A8.3a only is proposed for the next implementation approval. A8.4 is split
  into loader/integrity and exponential comparisons to keep review diffs small.

## Spec-to-test plan and remaining work

- Sections 10.4/11.1: R export shape/case/precision checks and deterministic
  regeneration; byte identity between generated and synced spec fixture.
- Sections 3.3/10.4/12.1: shared loader, manifest and corruption tests.
- Sections 4.6–4.7/5/6/12.1: per-row exponential and Weibull comparisons in
  both dtypes, two likelihood entry points, diagnostics and transformations.
- Sections 10.5/12.2: separate R-evidence docs step after reviewed code merges.
  Through-A8 evidence waits for durable fixture delivery; later Slice A gates
  are not claimed by this plan.

## Checks actually run

- Focused baseline: `uv run pytest -q` with JAX_ENABLE_X64=1 and then 0,
  SPEC_THROUGH=A7 and UV_CACHE_DIR=/private/tmp/survjax-uv-cache, selecting:
  tests/unit/test_likelihood_boundaries.py,
  tests/unit/test_likelihood_debug_nans.py,
  tests/unit/test_likelihood_tails.py,
  tests/spec/test_a7_likelihood_mpmath.py,
  tests/spec/test_a7_likelihood_tails.py,
  tests/spec/test_a7_likelihood_boundaries.py,
  tests/spec/test_a7_debug_nans.py.
  Float64: 144 passed in 13.19s; float32: 144 passed in 14.58s.
  Logs: /private/tmp/survjax-a8-plan-a7-x64.log and
  /private/tmp/survjax-a8-plan-a7-x32.log. This is targeted baseline evidence,
  not a full through-A7 gate.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run ruff format --check .`:
  exit 0, 73 files already formatted.
- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache uv run ruff check .`:
  exit 0, All checks passed!
- `git diff --check -- PLAN.md REVIEW.md` and `git diff --cached --check`:
  exit 0, no output. No conflict markers remain in either resolved file.
- Scripted comparisons confirm both prior plan histories are preserved;
  incoming docs, tests, tolerances and sync script match MERGE_HEAD; library,
  generators, fixtures and spec marker match the A8 parent.

No full gate, R regeneration, docs build or new A8 implementation in this
planning turn. The plan requires a maintainer-coordinated fixture handoff;
no numerical or API choice needed maintainer arbitration to resolve conflicts.
