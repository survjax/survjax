# A9.4/main merge resolution

Maintainer-initiated merge on a9b: HEAD 51319c8, incoming main 7eb2610.
Only PLAN.md and REVIEW.md conflicted. No code, docs, tests or dependencies
required manual resolution. Leave the merge commit to the maintainer.

## Reading order and invariants

1. PLAN.md's A9 merge-resolution section and A9.4 plan/evidence. It contains
   only the current A9 sub-slice, as required by main's planning format.
2. plans/A10.md: all incoming main A10 sections, retained verbatim including
   A10.d2/A10.d1/A10.4; its old archive remains an unchanged suffix.
   plans/README.md indexes that complete archive and preserves incoming
   release-agent handoff links.
3. The two parent review records below, retained verbatim. A9.4's separate
   independent report at ../review/slices/A9.4.md returned merge in round 1;
   optional nits N-1/N-2 remain. Main's A10.d2 record describes its own
   reviewer-owned validation and round-2 request; no new verdict is inferred.
4. Original non-conflict index objects remain unchanged except the intentional
   A10 archive/index corrections. All executable inputs, incoming protected
   .workmux.yaml and benchmark measurements remain untouched.

## Resolution checks and format correction

The first resolution preserved incoming A10 sections in PLAN.md, violating
its current-sub-slice format. The maintainer caught this before the merge
commit; move those sections verbatim into the existing plans/A10.md archive
and update its index. This is a correction to the merge resolution, not a
new implementation or a new sub-slice. The merge remains uncommitted.

Verify PLAN.md has no A10 step sections, plans/A10.md exactly equals main's
complete A10 body, and both parent review records remain verbatim. Verify
all 208 original stage-0 entries outside the two archive/index corrections
remain unchanged, no conflict markers/unmerged entries remain, resolutions
are staged and whitespace is clean. The benchmark script and measurements
must still match HEAD byte-for-byte.

No Ruff, mypy, pytest, benchmark or docs build rerun: changes only affect
planning/review organization. Evidence below belongs to parent steps,
not a fresh combined-tree run. No new independent-review round or commit.

---

# Parent A9.4 review record (verbatim)

Independent review: round 1/3

# A9.4 — Reproducible CPU baseline

Work on maintainer-created `a9b` from `1c0fd06`; plan archival commit
`0a20008` preserves the previous A10 sections verbatim. The maintainer
requested a roadmap audit, remaining-work plan and implementation. This
step measures existing Slice A code; no library/API/dependency changes.
Earlier A9 evidence remains in plans/A9.md and historical reviews in Git.

## Reading order

1. PLAN.md: roadmap audit, remaining A9 sequence, current step and limits.
2. scripts/benchmark_slice_a.py: independent analytical formulas, timed
   workload, synchronized sampling and trace-reuse assertions, report metadata.
3. benchmarks/slice_a_cpu.json: float64 and float32 metadata, then 24 rows
   each (2 families × 3 sizes × 2 parameter layouts × 2 construction paths).
4. Existing tests/unit/test_likelihood_composition.py and
   test_likelihood_custom_derivatives.py: regression evidence only, unchanged.

## Invariants and requirement mapping

- Spec 12.3: separate processes select float32/float64 through the existing
  JAX environment setting. The script never toggles it. CPU is selected
  explicitly and every initial result leaf is checked for CPU residence.
- Spec 7.2/12.3: both paths time the summed physical-parameter likelihood
  and reverse-mode gradient. Data and parameter diagnostics are auxiliary
  outputs of value_and_grad and are returned in full; their work survives
  compilation. The prepared path consumes all six data leaves, including
  stored issue bits. The raw path constructs ordinary SurvivalData inside
  jit from traced arrays. Recorded static flags are respectively
  (true,true,false,false,false,false) and six trues.
- Spec 7.3/12.1/12.3: each row's initial value and full gradient are checked
  against independently differentiated exact/right NumPy formulas evaluated
  in float64 from the actual input values. Exponential uses
  e*log(r)-r*t, derivative e/r-t. Weibull uses
  e*(log(k)-log(b)+(k-1)*log(t/b))-(t/b)**k; the code spells out its two
  derivatives. Shared gradients sum the per-row derivatives. The only
  tolerances are existing ATOL/RTOL loaded from tests/impl_tolerances.py.
  Data/parameter diagnostic `ok` outputs must both be true.
- Spec 7.2/12.3: after timing, each compiled function is called on two
  same-shaped slices of one 2N parent with changed parameters, times and
  event values; each result is analytically checked and total trace count
  must remain one. Parent slices preserve metadata. Raw event arrays are
  explicitly converted to the original boolean dtype so normalized weak
  bool fields do not change the jit signature. Existing A9.1 tests retain
  independent coverage of transformed diagnostics and second derivatives.
- Spec 12.3: every timer uses perf_counter and blocks on all returned leaves.
  Host generation, placement and concrete data construction are separate
  fields; first call includes tracing/compilation/execution. Three warmups
  precede ten synchronized samples; min/median/max and all samples are kept.
  Preparation and post-timing correctness/reuse checks are outside warmed
  evaluation. Record commands, seed, shapes, versions, OS/hardware/device,
  source revision and library Git status. Logical leaf byte sums are
  labelled as such; peak memory explicitly unmeasured.

No new timing-sensitive regular tests. The executable harness checks the
full 48-case workload numerically, and unchanged composition tests provide
regression coverage. No new tolerance, unchecked path or numerical repair.
No spec ambiguity/deviation encountered. Code is 230 lines; the JSON is
measurement evidence rather than additional implementation.

## Measurements actually run

Sequential commands; this agent started regression tests only after measurement:

- `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run python scripts/benchmark_slice_a.py --output benchmarks/slice_a_cpu.json`
- `JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu uv run python scripts/benchmark_slice_a.py --output benchmarks/slice_a_cpu.json`

Both exited 0, **24 checked rows each**. Each row checks the initial and two
changed workloads: **144 analytical value/gradient comparisons** in total,
plus 48 single-trace assertions. The machine is Apple M4 Max, arm64, 16
logical CPUs, macOS 26.6.2, actual device cpu:0. Python 3.12.9, JAX/jaxlib
0.11.2, NumPy 2.5.3, survjax 0.0.1.dev0. The measured library revision is
0a200085b0ab7a61f154da3ebc6cdcceaaf893d9, with no dirty source files.
`uv run python -c 'import survjax; print(survjax.__file__)'` confirms
`/Users/wamster3/dev/survjax/dev-a9b/src/survjax/__init__.py`.
Benchmark/report files were uncommitted during measurement.

| Mode | Warm medians, N=100000 (all 8 cases) | First call range (all 24 cases) |
| --- | --- | --- |
| float64 | 0.305–1.117 ms | 85.6–144.3 ms |
| float32 | 0.231–0.523 ms | 71.8–132.9 ms |

Median times grow much less than 100-fold over the 100-fold increase in N,
consistent with dispatch overhead and CPU parallelism; these three sizes
do not establish asymptotic complexity. Row-parameter gradients and diagnostic outputs grow linearly in
logical bytes. Maximum logical input/output bytes are 5,300,000/3,100,010 in
float64 and 2,900,000/2,300,006 in float32. These are not peak-memory claims.
At N=10000, traced/prepared median ratios range 1.14–1.60 in float64 and
1.23–1.58 in float32. At N=100000 they range 0.89–1.18 and 0.89–1.07.
Concrete construction ranges from 1.56 to 634.99 ms across both modes,
with the first row at each size including eager compilation/cache setup;
initial placement ranges from 0.11 to 1.30 ms. These costs are separate
from warmed evaluation and are sensitive to process cache state.
The ordering reverses for some cases: retain raw distributions and make no
universal overhead or speedup claim. This is the first recorded baseline,
so there is no earlier baseline against which to assert a regression.

An independent JSON audit checked both modes, exactly the 24 expected unique
case keys per mode, ten finite positive samples per row, min/max summaries,
positive byte counts, CPU metadata, clean library source status and all
correctness markers: **48-row JSON audit passed**. A second independent
standard-library audit checked each median, parameter/time shapes and static
flags: **48-row independent median/shape/metadata audit passed**.

Initial attempts and corrections: sandbox hardware identification (`sysctl`)
failed with Operation not permitted; the measurements above were authorized
outside the sandbox. The first executable attempt caught an extra trace when
a normalized weak boolean event array replaced a strong raw input. Explicit
boolean dtype conversion fixes the harness signature; no library change.
An earlier successful float64 matrix was rerun after provenance metadata
was added; only the final run is retained. No tolerances were changed.

## Verification

`JAX_ENABLE_X64=1 uv run --group test pytest -q tests/unit/test_likelihood_composition.py tests/unit/test_likelihood_custom_derivatives.py`
and the same command with `JAX_ENABLE_X64=0`: **15 passed in 28.03s** in each
mode, no skips. These focused regressions ran alongside the full gate, after
all benchmark measurement had finished.

`scripts/check_wheel.sh --through A9`: exited **0**, ending
**check: all gates passed through A9**. Fresh wheel:
`/var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.bigUQH/survjax-0.0.1.dev0-py3-none-any.whl`,
SHA-256 `8ca5093b02ae2ee23e85bc8feea86374f6ab2114906b5f3896a8b78c80af37f9`.
The unchanged runner supplies the same wheel to both dtype suites.

| Gate stage | float64 | float32 |
| --- | --- | --- |
| Own tests, including external wheel | 1044 passed, 7 skipped, 80.00s | 1050 passed, 1 skipped, 120.83s |
| Source doctests | 4 passed, 1.38s | 4 passed, 1.25s |
| Independent through A9 | 2211 passed, 28 skipped, 176.18s | 2202 passed, 37 skipped, 172.91s |

Guard: **ok (a9b)**. Ruff: **104 files already formatted**, **All checks
passed!** Mypy: **Success: no issues found in 20 source files**. No new skips,
selection exceptions or protected-file changes. Detailed gate/focused output:
`/private/tmp/survjax-a94-gate.log`, `/private/tmp/survjax-a94-focused64.log`,
`/private/tmp/survjax-a94-focused32.log`. Reports are the durable evidence;
temporary logs/artifacts need not survive.

`git diff --check` passes; an independent content comparison confirms the
archive commit preserved A10's plan sections verbatim. The only changed
implementation is the 230-line benchmark. No library source changed before
or after measurement. R, docs build/rendering, sdist and accelerator checks
were not run in this code step. Independent review follows the commit.

## Limits and next work

One machine/backend, one seed, positive interior parameters, valid exact/right
observations only; no accelerator, invalid-input performance or optimizer
claims. Sequential fixed ordering, short ten-sample distributions, no CPU
isolation or thermal control; machine load and caches can affect timings.
Concrete construction is single-shot and includes available eager compilation
caches, so it is not an isolated steady-state preprocessing benchmark.
CPU placement on Apple unified memory does not establish physical bus-transfer
costs. Diagnostics are deliberately returned in full; timings are specific to
that work and are not a minimal loss-only benchmark. Peak memory is unmeasured.

The workspace roadmap is stale; subrepo instructions reserve its update to
the workspace-root maintainer. Handoff should note A9.1/A9.2 and combined-tree
A9 gates already passed, plus the present CPU-baseline review status.
A9.d1 executable docs/rendered review, replacement tiny-example evidence and
the final Slice A gate map remain pending. This is not the final A9 review or
a Slice A completion claim. A10 fixture/reference/docs work remains separate.
Stop after this step's independent review; do not begin the next step.

---

# Incoming A10.d2 review record (verbatim)

Independent review: round 2/3

# A10.d2 — Concordance in the editable playground

Authorized after the maintainer reviewed A10.d1. Branch A10.4, clean base
fdce5db. Scope: docs/playground.ipynb only, plus PLAN.md and REVIEW.md.
No source, tests, dependency, protected paths or spec changes. The separate
8757c0f planning packet is not part of this step and remains unchanged.

## Reading order

1. docs/playground.ipynb: corrected introduction, then the new concordance
   cells before Things to try; existing data/distribution cells are preserved.
2. PLAN.md A10.d2: requirements, single-runner validation and remaining steps.
3. docs/concordance.md: accepted background for the same hand-enumerated
   cohort and the semantics; unchanged by this step.

## Validation ownership — reviewer is the sole runner

The maintainer explicitly requested “only you or the reviewer run them not
both as it would be exactly the same no?” The previous reviewer repeated the
implementer's docs build because its role requires one. For this step the
implementer deliberately runs NO docs build, Ruff, mypy, pytest or full gate.
The reviewer owns the fresh scripts/build_docs.sh run in both modes, Ruff
and notebook validation. Checks are pending at this commit, assigned to
review rather than omitted or claimed passed. Please record exact outcomes,
including the hand-derived C/count assertions and unexecuted source state.
The implementer reviews source and staged changes only. The final A10 audit
will inspect the complete rendered site; no browser QA claim for this step.

## Invariants and requirement mapping

Spec 6.1/9.1–9.2: finite caller-supplied higher-risk scores; exact/right only,
no truncation; N is last and must match; leading score draws evaluated
separately. Default cohort is the guide's time [1,2,2,3], events [T,T,F,T],
scores [4,3,1,4]. Hand pairs: (1,2),(1,3),(2,3) concordant; (1,4) tied;
(2,4) discordant; (3,4) excluded. Counts (3,1,1,5), C=3.5/5=7/10.
Reversal gives (1,3,1,5), C=3/10; constant scores give C=1/2. Assertions
check these expected fractions and transformation agreement on the defaults;
users editing defaults are told to recalculate their expectations.
Counts are reconstructed as Python ints from hi/lo on the host. No gradient
is taken through C. The existing mixed cohort is valid but unsupported;
all-censored data are evaluated with no comparable pairs. NaNs retain their
correct distinct masks and failed counts are zero, without dropping rows.
Spec 10.5/12.1: synthetic, public-API, network/R-free examples execute during
the two-mode build; notebook code cells are committed with empty outputs
and null execution counts. Remove the existing saved import error and stale
Python 3.14.5 metadata; the locked environment determines the interpreter.
Spec 12.2: editable playground coverage, not the held-out example/final audit.

## Checks and limits at commit

Implemented six new cells (three prose/code pairs) before Things to try.
Notebook diff: 138 additions, 23 deletions, 161 changed lines. Existing data
and distribution cell sources/IDs are preserved. Intro and exploration
prompts changed as planned; all code outputs/counts are cleared, including
the old ModuleNotFoundError output, and stale Python version metadata removed.
The new value assertions use exactly the correctly rounded hand-derived
7/10, 3/10 and 1/2 in the result dtype; they apply only to this default cohort,
not as a general numerical tolerance guarantee.

Source/diff inspection completed. No automated checks run by implementer.
Automated checks are pending the reviewer as sole runner. No claim of a
passing build or lint before review. Do not rerun the full through-A10 gate:
source/tests have not changed since the passing 7c4f57b wheel gate.
No browser inspection performed; held-out synthetic example and final
evidence/rendered-site audit remain. Commit and round 1 follow.

## Round 1 result and B-1 follow-up

Round 1 on 93dbca8 returned fix-first: B-1 is the sole blocker, a missing
Ruff line wrap in the new mapped_c call (91 characters). No should-fixes,
nits or spec questions. The reviewer confirmed the hand counts, C fractions,
mask prose, cleared outputs/counts and preserved existing cells/IDs.

Reviewer-owned checks (not repeated by implementer):
- scripts/build_docs.sh: sandboxed attempt exited 2 (uv cache permissions);
  permitted rerun exited 0, both builds succeeded with no Sphinx warnings.
  Playground execution: 8.73s float32 / 18.59s float64. Existing IPKernelApp
  TCP-transport warnings only. The same assertions passed in both modes.
- Separate float32 nbconvert execution: C=0.7, counts (3,1,1,5), all three
  draw evaluators [0.7,0.3,0.5], unsupported NaN/True and zero-pair NaN/True.
- Ruff lint passed; format check failed for B-1. Plain check.sh --through A10
  exited 1 at formatting, so no later gate stages ran. No full gate claim.
- Float32 notebook mutants: tie credit 0 caught by C assertion; failed
  cohorts incorrectly flagged no-pair caught by mask assertion. Dropping only
  delayed-entry detection survived as expected because this mixed cohort
  also has interval/left censoring; no new rule coverage claimed.

B-1 fix: manually wrap that one call exactly as the report's formatter
direction specifies. Two added notebook source lines; no numerical, prose,
metadata, output or assertion change. Final notebook delta from fdce5db:
140 additions / 23 deletions (163 changed lines). The implementer still ran
no build, lint, pytest or other automated validation. Commit this formatting
fix-up and request round 2, with the reviewer still the sole check runner.

Reviewer cleanup note: /tmp/survjax-mut-A10.d2 was reverted and reported
clean but removal was refused twice. It is reviewer-owned; the implementer
does not manipulate it under the repository's independent-review boundary.
