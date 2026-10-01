# Step DOC-4: readable reprs for SurvivalData and DataIssues (2026-10-01)

Requested by the maintainer after DOC-3 ("make a nice repr for SurvivalData
and DataIssues as well"). The spec does not cover repr. This replaces DOC-3's
"no summary repr" choice and adds no public API. One step, one commit on
`docs-startup`. Spec sections touched: 3.2 (private fields stay private), 6.2
(issue names), 6.3 (containers remain pytrees, unchanged).

## Behaviour

Both reprs read concrete values on the host. If any leaf is a JAX tracer, they
fall back to a short form with no value conversion (CLAUDE.md: never convert
tracers to NumPy), e.g. `SurvivalData(shape=(6,), dtype=float32, traced)`.

`SurvivalData`, eager: one line per observation in R-like notation, plus the
eligibility window when it is not the default `(0, inf]`. Invalid rows are
printed as stored, with a footer pointing at `.issues`:

```text
SurvivalData: 6 observations, float64
  0  4
  1  12+
  2  7
  3  9+
  4  (6, 9]   window (2, inf]
  5  (0, 3]   window (2, inf]
```

- Notation: exact `t`, right censored `t+` (as in R's print), every other
  censored row as its stored set `(lower, upper]`, so left censoring prints
  `(0, 3]` (maintainer's choice). Numbers use `:g`.
- More than 20 rows: first 10, `...`, last 5. N-d data: header shows the
  shape and rows show their index tuple.
- Footer only when non-empty: `issues: 4 invalid rows, 1 uninformative
  (see .issues)`.

`DataIssues`, eager: the overall verdict, then only the rules that fire,
with the row count and first index (the same wording as
`raise_if_invalid`):

```text
DataIssues: 4 of 5 rows invalid
  invalid_time            1 row, first index (0,)
  missing_time2           1 row, first index (1,)
  invalid_interval        1 row, first index (2,)
  censored_before_entry   1 row, first index (3,)
  uninformative           1 row, first index (4,)  (not an error)
```

A valid result prints `DataIssues: all 6 rows valid`.

## Files

Approved with changes: each repr lives next to its class, not in a separate
module, accepting files somewhat over 300 lines.

- `data.py`: `@dataclass(..., repr=False)` plus `__repr__` and two small
  formatting functions. The DOC-3 `field(repr=False)` declarations become
  redundant and are removed.
- `diagnostics.py`: the same for `DataIssues`. `raise_if_invalid` and the
  repr share one per-rule count helper; the exception message is unchanged.
- `tests/unit/test_data_repr.py`: replace the DOC-3 tests with hand-written
  expected strings for: each observation kind; truncation window; invalid
  footer; long-data elision; 2-d index tuples; traced fallback for both
  classes (inside `jax.jit`); valid and invalid `DataIssues`. Keep the
  six-leaves check.
- `docs/survival-data.md`: no code changes needed (cells already end with
  `data`); reread the rendered page and update prose that describes the
  printed repr. `REVIEW.md` as usual.

Checks: `scripts/check.sh --through A2`, `scripts/build_docs.sh`, `git diff
--check`. Expected diff about 250 lines.

## Decisions

- Left censoring prints as the stored set `(0, 3]`.
- No precision information beyond the dtype in the header.

# Step DOC-3: tutorial-style SurvivalData docs and a clean repr (2026-10-01)

Approved by the maintainer after reviewing DOC-2 (`726e561`): the guide reads
like a unit test, not a tutorial. One step, one commit, on a
maintainer-created branch (the worktree is a detached HEAD at `docs-startup`).
Spec sections: 3.1-3.3, 6.2, 10.5.

## Repr

- `src/chronox_jax/data.py`: declare `_issue_bits` and `_may_contain` with
  `dataclasses.field(repr=False)`, so the repr shows only the five public
  arrays (spec 3.2 keeps the others private). No summary repr. `DataIssues`
  is unchanged. Pytree flattening and `__init__` are unchanged.
- `tests/unit/test_data_repr.py`: the eager repr names the five public
  fields and omits the private ones; `repr` of data built inside `jax.jit`
  does not raise and omits them; `jax.tree_util.tree_leaves` still gives six
  leaves.

## Docs: rewrite `docs/survival-data.md` as a tutorial

Keep the facts of DOC-2; change the form.

- Open with a short note: what works now (construction, diagnostics) and what
  doesn't yet (likelihoods). No closing scope section.
- Use one running synthetic story: a small cohort (about six patients,
  follow-up in months) built up section by section. Each section answers a
  reader task: "time and status, as in R"; "the event happened between two
  visits"; "known only to have occurred before the first visit"; "patients
  entered late (truncation)"; "finding bad rows"; "inside `jax.jit`".
- Show results instead of asserting them: end cells with the displayed value
  (`data`, `data.upper`, `data.issues.invalid_rows`). Use asserts only where a
  silent check helps the reader, if at all. The build failing on execution
  errors is the executable check (spec 10.5).
- Show real errors: call `raise_if_invalid()` and the misuse examples in cells
  tagged `raises-exception`, so the rendered page shows the message. No
  `try/except/else: raise AssertionError` scaffolding.
- Explain why before how: a sentence or two on what each censoring kind means
  for the patient before the call that encodes it. Introduce the normalized
  fields when the first output shows them, not as an opening list of rules.
- Present the R comparison as a table (`Surv(...)` call next to the Chronox
  call, plus a notes column). Re-check each claim against the official `Surv`
  help page. In particular, the "(start, end]" wording there may refer to
  counting-process data, not interval censoring; drop the claim unless the
  page supports it.
- Dtype behavior: one short section with a single example, not a test of both
  modes.

`docs/playground.ipynb`: remove the empty trailing cell and reset the
kernelspec and `language_info` metadata to the DOC-1 minimal form. Keep the
maintainer's `data` cell, which now shows the clean repr. Have it follow the
same cohort as the guide, without asserts. `docs/index.md`: update the guide's
one-line description if needed.

## Checks and review

`scripts/build_docs.sh` (both dtype modes), `scripts/check.sh --through A2`,
`git diff --check`. Read the rendered x64 guide end to end and report in
`REVIEW.md` whether every output shown matches its surrounding prose. The
guide rewrite replaces most of a 176-line page, so the diff may exceed about
300 lines. If it does, say so in `REVIEW.md` rather than splitting the step.

# Documentation startup on `docs-startup` (2026-09-30)

The maintainer requested two separate tasks following specification section
10.5. Each is one reviewable step and one commit. Stop after the first step's
`REVIEW.md` and await instruction before starting the second. This plan is for
approval before either implementation step; the earlier plans below are history.

## Step DOC-1: local documentation process

Spec sections: 10.2, 10.3, 10.5, and the documentation item in 12.2.

Files to create or change:

- `pyproject.toml`, `uv.lock`: add a locked optional documentation group with
  Sphinx, MyST-NB, JupyterLab and a Python kernel. Keep core dependencies
  unchanged.
- `docs/conf.py`, `docs/index.md`: minimal Sphinx site for Markdown and
  notebooks, navigation, and a clear statement of the currently implemented
  subset. Include the existing `docs/simulation-guide.md` only if it can be
  visibly labelled as illustrative and unverified without build warnings.
- `docs/playground.ipynb`: a small synthetic, editable notebook using the
  current public `SurvivalData` API; no saved outputs.
- `scripts/build_docs.sh`: one documented, non-interactive command that builds
  HTML from a clean checkout, executes runnable content afresh in separate
  float32 and float64 processes, and fails on warnings or execution errors.
- `.gitignore`: exclude generated site, notebook execution artifacts and
  caches. `REVIEW.md`: report exact commands, outcomes and rendered-page
  inspection. Update this plan if implementation reveals a necessary split.

Public API: none. Tests: the build command itself is the end-to-end check;
verify it from a clean working tree and inspect the rendered index and notebook.
Run the existing A2 gate and `git diff --check` if dependency or tooling edits
affect those checks. No network, R, real data or demo dependencies at doc build
time. Do not publish or host the site.

## Step DOC-2: A1–A2 documentation

Spec sections: 3.1–3.3, 6.2, 10.2, 10.5, and 12.2. Cover implemented behavior
only; label A3 indexing/static-metadata guarantees and later likelihood modes
as planned. Revise this step plan, if needed, before writing its code and seek
approval under the working agreement.

The maintainer's follow-up authorizes DOC-2 and also asks for the commands to
open the rendered HTML. Add those commands to `docs/index.md`. A locally edited
playground currently contains exploratory cells and saved outputs: preserve its
cell code, clear saved outputs for the committed source, and extend it with the
new guide's working API. This is still the one documentation step below.

Files to create or change:

- `docs/survival-data.md`: executable synthetic examples for event/status,
  explicit right/left/interval flags, mixed data, `time2`, bounds, and
  truncation. Show normalized fields and the exactness rule for bounds.
- In the same page, explain issue masks, `invalid_rows`, `ok`, valid but
  uninformative rows, structural `TypeError`/dtype `ValueError`, and explicit
  host `InvalidDataError`. Demonstrate ordinary construction and diagnostic
  access inside `jax.jit`, with the host checkpoint after compiled execution.
- In the same page, compare the flag and bounds conventions to R's
  `survival::Surv` forms, including event coding, interval endpoints and
  counting-process `time2`. Check comparison wording against official R
  documentation. Claim no numerical agreement without R-generated fixtures.
- `docs/index.md` and `docs/playground.ipynb`: link the guide and make the
  notebook a short interactive path through the same working API.
- `REVIEW.md`: reading order, invariants, exact example/build outcomes in both
  dtypes, source of R-convention claims, known gaps and rendered-page review.

Public API: none. Tests: executable examples through `scripts/build_docs.sh`
in both dtype modes, plus the existing A2 gate and `git diff --check`. Examples
must check their stated outputs rather than merely run. No new unit tests that
repeat documentation examples. Keep this step near 300 changed lines; split it
before implementation if the reviewed plan requires more.

# D-006 diagnostic compatibility decision (2026-09-30)

The maintainer approved retiring `TracedInputError`, keeping `ValueError` for a
traced `DataIssues.raise_if_invalid()` call, and exposing only the
`data.issues` property (no callable `data.issues()` alias). On the
maintainer-created `spec-deviations` branch, update the normative contract in
`spec/SPECIFICATION.md` sections 6.2, 10.2 and 13, its review/decision history,
and D-006 in `DEVIATIONS.md`. Leave D-010 and the other deviation statuses
untouched. This explicit maintainer request authorizes the spec edit.

Public API: clarify that only `InvalidDataError` is exported for data
diagnostics; no code or behavior changes. No tests are added for this
documentation step. Review the diff for internal consistency and run
`git diff --check`. Overwrite `REVIEW.md` with the reading order, invariants,
evidence and uncertainties. Commit this one step on `spec-deviations` and stop.

# A2 constructor API migration — specification revision 9

## Pre-merge bookkeeping step (2026-09-30)

The maintainer requests a clarification of the non-exact equal-endpoint case
and cleanup of deviation IDs/statuses. Change only the `from_bounds` docstring
in `src/chronox_jax/data.py`, the branch D-006 references in
`src/chronox_jax/_data_validation.py` and
`tests/unit/test_data_construction.py`, and `DEVIATIONS.md`. Rename the branch
D-006 to D-011 and mark D-011, D-007, D-008 and D-009 as superseded by
revision 9, leaving final closure to the maintainer. Update `REVIEW.md` and
commit this small step on `a2-constructors`.

Public API: unchanged. Spec: sections 3.1, 6.2 and 13. Tests: the existing
independent A2 equal-endpoint acceptance cases and the full A2 gate; no new
test is needed for a docstring and bookkeeping change. Do not edit the
read-only specification or independent tests.

Branch: `a2-constructors`. A2-R1 exception alignment was committed as
`c6ef7e3`. The maintainer's 2026-09-30 request authorizes the remaining
constructor API migration: remove issue-returning constructor methods, accept
eager and traced observation inputs in the two ordinary forms, and expose
diagnostics through `data.issues`.

## Current step

Spec sections: 3.1–3.3, 6.2–6.3, 10.2 and 12.1–12.2.

Files and public API:

- `src/chronox_jax/data.py`: retain exactly `SurvivalData(...)` and
  `SurvivalData.from_bounds(...)` as construction methods; remove automatic
  host validation and the `with_issues` method; add a typed `issues` property
  decoding the stored issue bitfield. No `from_bounds_with_issues` method.
- `src/chronox_jax/_data_validation.py`: remove obsolete host-only constructor
  validation and references to the removed method, preserving normalization,
  dtype policy, argument-misuse checks and issue attribution.
- Own tests in `tests/unit/`: migrate callers to ordinary construction and
  `data.issues`; check eager, `jit` and `vmap` behavior, invalid-value masks,
  explicit host checkpoints, traced inputs, tree reconstruction and the absent
  methods. Keep tests with code in the same commit.
- `REVIEW.md`: map checks to requirements and report exact gate results.

The revision 9 specification and its acceptance criteria already require this
API; `spec/GOALS.md`, `spec/SPECIFICATION.md` and `tests/spec/` are read-only.
There are no current examples or docs modules to migrate. Historical deviation
entries retain their original wording and status.

## Verification

Run `scripts/check.sh --through A2` to cover Ruff, mypy, own tests and the
independent A2 tests in both dtype modes. Use focused A3 acceptance tests for
the removed-method and tracing checks, without claiming A3 complete. Run
`git diff --check`. If an independent test conflicts with the specification,
record the issue in `DEVIATIONS.md` and leave the test unchanged.

## Later handoff

A3 owns indexing and slicing, value-derived concrete static metadata and
compile-count tests (spec 3.2). The current all-true metadata is only a
conservative interim value. A6 owns likelihood issue propagation and explicit
failure for unimplemented observation modes. Neither is part of this step.

The traced host-checkpoint exception subtype and legacy diagnostic compatibility
remain open in spec section 13 and `DEVIATIONS.md`. This step preserves the
existing traced-checkpoint failure and does not introduce a callable alias.

## Performance amendment on `spec-speed` (2026-09-30)

The maintainer requests adapting the earlier performance draft to the current
specification at `9a403a3`, including revision 10 diagnostic compatibility.
This authorizes edits to the normally read-only specification and goals.

- Update `spec/GOALS.md` with speed, memory efficiency and acceleration.
- Update `spec/SPECIFICATION.md` sections 1, 7.2, 9.2, 10.3, 12 and 14:
  retain pure ordinary construction, `data.issues`, stored-issue propagation,
  conservative traced metadata and simulation fit-skipping semantics. Measure
  prepared-data evaluation and traced construction separately. Keep GPU checks
  and scalable metric algorithms optional, with a small CPU baseline in Slice A.
- Leave the final amendment revision number for acceptance: the current header
  says revision 9 while the history includes revision 10. No API changes,
  implementation edits, new dependencies or changes to independent tests.
- Overwrite `REVIEW.md` with reasoning, reading order, preserved contracts,
  documentation checks and unmeasured performance requirements.
- Checks: whitespace, Markdown/section references, allowed diff scope and exact
  preservation of unrelated statistical/API sections and example code. Runtime
  tests and benchmarks are not run for this documentation-only step.
- Commit this one documentation step on `spec-speed`; do not merge or publish.

## Illustrative simulation guide on `spec-speed` (2026-09-30)

The maintainer approved moving the simulation workflow out of the normative
contract and removing recipe-specific acceptance requirements. This authorizes
the following documentation step, including edits to the specification.

- Move the detailed recipe into `docs/simulation-guide.md`, explicitly
  illustrative and unverified; retain `lax.map` as one example choice.
- Update specification sections 6.2, 11.4, 12.1, 12.3 and 14: retain library
  guarantees, check compiled composition directly, and leave study workflow,
  external solver and loop selection to callers and measurements.
- Public API, goals, implementation, dependencies and tests remain unchanged.
  Add no runtime tests; check whitespace, local links, moved example code and
  preservation of unrelated contracts. Record results in `REVIEW.md`.
- Commit this single documentation step on `spec-speed` and stop.
