Independent review: round 1/3

# A10.4 — Synthetic R concordance references

Authorized by “plan and implement A10.4”. Branch `A10.4`, base `e8dbbed`.
Exactly 300 changed implementation/test/fixture lines, including 105 CSV
lines, plus PLAN.md and this review. No production or public API changes.

## Reading order

1. scripts/fixtures/generate_concordance.R: pinned versions, synthetic
   cohorts, exact R call, reference counts and base-R CSV/JSON export.
2. tests/fixtures/concordance/README.md and concordance_synthetic.json:
   regeneration, case coverage, reference provenance and protected handoff.
3. tests/conftest.py's new concordance branch/helper: existing shared
   manifest verification precedes parsing; CSV/JSON observations and scores
   must agree in case/row order, with matching provenance/CSV MD5.
4. tests/reference/test_concordance_r.py: four cases crossing two cohorts
   and prepared/raw inputs, each evaluated eagerly, jitted and vmapped.
5. PLAN.md's A10.4 scope/results. Earlier review is in base commit history.

## Invariants and spec/test mapping

- Spec 9.1–9.2/10.4: only synthetic positive exact/right observations, no
  truncation, weights or strata. Seed 20261023, explicit RNG kind. R 4.4.2,
  survival 3.7-0 (R reports 3.7.0), flexsurv 2.3.2 checked before output.
  Export uses base R; no dependency changes. The official manual and pinned
  installed help establish reverse/timewt/timefix and count-field meanings.
- Spec 9.2/12.1: all expectations are R fit$concordance/fit$count. Generator
  uses reverse=TRUE, timewt="n", timefix=FALSE. Concordant/discordant map
  directly, tied.x is tied_score, and their sum is comparable; tied.y and
  tied.xy are excluded. A separate Python pair enumeration of the spec
  rules matched all six R results; no convention difference was found.
- Heavy-ties cohort: 100 rows, 36 events, 64 censored; fixed first six rows
  guarantee event/event, event/censor and censor/censor ties. Two seeded
  integer score vectors and one constant vector exercise score ties. R
  counts (C,D,T,total): (698,937,156,1791), (810,848,133,1791),
  (0,0,1791,1791). The untied four-row cohort has 75% censoring and
  perfect/reversed/constant scores: counts (2,0,0,2), (0,2,0,2), (0,0,2,2).
- Spec 6.1/9.1/12.1: reference tests check three score draws together,
  prepared/raw forms and eager/jit/vmap. Values use unchanged centralized
  dtype tolerances; eight hi/lo words and three masks compare exactly.
  Output dtype follows explicitly typed observation data in each process.
  Integer/quarter times and integer scores preserve every order/tie in both
  modes. Existing suites cover invalid/unsupported/undefined cohorts and
  nonzero high words; this fixture intentionally has finite defined ratios.
- Spec 10.4: CSV and JSON generated together, JSON records seed/RNG/versions,
  platform, generator path/hash, CSV hash and call. SHA-256 covers both
  payloads and is checked by the existing shared loader before JSON parsing.
  Missing data fail rather than skip; normal tests require no R/network.

## Checks actually run

R version/package query: R 4.4.2, survival 3.7.0, flexsurv 2.3.2; inspected
installed concordance help with Rd2txt. Also consulted the official manual:
https://stat.ethz.ch/R-manual/R-devel/library/survival/html/concordance.html

From the repository root:
- `Rscript --vanilla scripts/fixtures/generate_concordance.R --check-only`:
  exit 0, “Checked 104 synthetic rows, two cohorts, three score draws each”.
- Ran the generator with default output, then independently to
  `/private/tmp/survjax-a104-run1` and `/private/tmp/survjax-a104-run2`.
  `diff -r` of both runs and `cmp` of each payload against the committed
  directory: exit 0, no output; CSV/JSON byte-identical on recorded platform.
- `shasum -a 256 -c SHA256SUMS` in tests/fixtures/concordance: both OK.
- Standalone Python JSON audit checked current generator MD5 and enumerated
  every unordered pair from exported time/event/score arrays, using spec 9.2
  rules without survjax or independent test helpers. All six R count triples
  match exactly, defining fractions agree within 1e-15; all three time-tie
  kinds present. This audit checks conventions; it generates no fixture values.
- One invocation mistakenly ran the root-relative generator from the fixture
  directory: exit 2, script not found. Repeated from the repository root,
  exit 0; no output files were changed by the failed invocation.

Focused command, separately with JAX_ENABLE_X64=1 and =0:
`uv run --group test pytest -q tests/reference/test_concordance_r.py tests/unit/test_concordance_counts.py tests/unit/test_concordance_results.py tests/unit/test_concordance_entry_points.py tests/unit/test_concordance_transformations.py tests/property/test_concordance.py`
77 passed in 27.85s (float64), 77 passed in 27.86s (float32), no skips.

`uv run ruff format --check .`: 104 files already formatted.
`uv run ruff check .`: All checks passed!
`uv run mypy src`: Success: no issues found in 20 source files.
`scripts/check_wheel.sh --through A9 > /private/tmp/survjax-a104-wheel-gate.log 2>&1`:
exit 0, ending “check: all gates passed through A9”. Fresh wheel retained at
`/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.dczbdn/survjax-0.0.1.dev0-py3-none-any.whl`,
SHA-256 `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9`.
The installed-wheel smoke test is included in own suites; all gate stages
ran unchanged, with the normal existing skips and no added exclusions.

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Own tests, including wheel | 1048 passed, 7 skipped, 63.67s | 1054 passed, 1 skipped, 60.47s |
| Source doctests | 4 passed, 0.86s | 4 passed, 0.86s |
| Independent through A9 | 2211 passed, 28 skipped, 62.81s | 2202 passed, 37 skipped, 54.43s |

No implementation/test changes after these runs. Final Ruff and whitespace
checks pass. All protected paths, spec marker, production, dependency and
docs files remain unchanged. Initial commit and independent review follow.
Sandboxed git add/commit each failed creating the external worktree index
lock; staging/commit use the approved escalated Git commands. No source or
test change resulted, and staged whitespace/scope checks pass.

Independent A10 command, separately with JAX_ENABLE_X64=1 and =0 and
SPEC_THROUGH=A10: `uv run --group test pytest -q` followed by all six
tests/spec/test_a10_concordance_{api,diagnostics,fixtures,properties,shapes,values}.py
paths (explicitly expanded). Exit 1 in each mode: 8 failed, 113 passed in
57.50s (float64); 8 failed, 112 passed, 1 skipped in 59.90s (float32).
All eight failures are the missing protected concordance_synthetic.json,
at tests/spec/conftest.py:101. No assertion was reached in those tests;
no new skip or exclusion. Logs: /private/tmp/survjax-a104-spec-x{1,0}.log.

Python 3.12.9, JAX 0.11.2, NumPy 2.5.3; no version changes or matrix claim.

## Limits and handoff

The independent loader reads tests/spec/fixtures/concordance_synthetic.json,
which is absent. Its matching generated export is in our fixture directory;
copying into the protected directory belongs to the maintainer/spec-test
workflow. No protected paths, selection rules, tests or assertions changed.
Do not interpret the through-A9 regression or our R references as a full
through-A10 pass. Concordance docs/playground/held-out example remain later
steps. No docs build or performance evidence claimed for this reference step.
No spec ambiguity or new deviation. Stop after independent review.
