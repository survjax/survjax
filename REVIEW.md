# Review: A10 follow-up — Boolean precondition and pair workspace guard

## Reading order

1. `tests/unit/test_concordance_counts.py`: the new
   `test_pair_workspace_has_no_cohort_square`, then the retained hand-counted
   and carry tests. The repeated final high-word assertion is removed.
2. `src/survjax/_concordance.py`: `_pair_counts` now explicitly requires a
   boolean (N,) exact mask. Pair/count arithmetic is unchanged.
3. PLAN.md's follow-up and public integration outline; DEVIATIONS.md's D-027
   acceptance record. Formal spec alignment belongs to another agent/branch.

Read the entire reviewer report `../review/slices/A10.md`, reviewing `04bc1a7`.
Its verdict is merge with no blockers or should-fix findings. The maintainer
requested addressing the review and explicitly assigned spec changes/sync
elsewhere. This commit addresses its optional implementation/reporting nits;
no new public metric, dependency, docs or numerical behavior.

## Findings and invariants

- N-1: document finite (N,) scores/times and boolean (N,) exact masks. This
  private function still requires validated exact/right cohorts; it does not
  accept unchecked event codes as an additional input convention (spec 9.2).
- N-2: inspect the JAX program at N=7, recursively including loop/body params.
  All variable and equation shapes are examined. A cohort vector must be
  present and a (7,7) array must be absent (spec 9.2). This is a regression
  guard for this traced program, not a backend memory measurement or proof
  for arbitrary N. No persistent inspection framework is introduced.
- N-3: the next public integration plan must substitute safe finite values for
  entire invalid/unsupported cohorts before counting, retain cohort length and
  diagnostic masks, and zero the returned count words. No rows are dropped.
  A NaN score must never reach counting and break C+D+T=comparable. This is a
  future integration requirement, not a newly implemented runtime path.
- N-4: remove the redundant high-word assertion/import; the exact Python-integer
  comparison still checks both words after every increment. Rewrite the old
  merge instructions as past events. The tables below are committed evidence;
  historical `/private/tmp` log paths in the plan are optional diagnostics,
  not the sole source of results.

Existing hand counts, exact carry comparisons and tolerance policy are
unchanged. All NumPy/Python count conversions remain host-side in tests.
The new trace inspection does not convert tracers or change global JAX mode.

D-027: the maintainer accepted named uint32 hi/lo count objects and the
previously explained result masks, failed-cohort zero counts, scalar rejection
and valid-empty-cohort undefined behavior. Record that acceptance without
changing the entry's formal open status or implementing a public result type.
S-1 (revision 35 alignment and sync) is explicitly external to this branch;
no `.spec-seen`, spec, exports, independent tests or sync scripts are changed.
S-2's early A10 authorization remains in the plan; no delivery/role text edit.

## Checks run for this follow-up

Every uv command uses `UV_CACHE_DIR=/private/tmp/survjax-uv-cache`.

Focused command, in separate processes with `JAX_ENABLE_X64=1` and `=0`:
`uv run --group test pytest -q tests/unit/test_concordance_counts.py tests/unit/test_observation_inputs.py tests/unit/test_likelihood_entry_points.py`.

| Check | Float64 | Float32 |
| --- | --- | --- |
| Focused suites | 93 passed in 5.90s | 93 passed in 5.40s |
| N-by-N mutant: shape test | 1 expected failure | 1 expected failure |
| N-by-N mutant: mixed-cohort values | 1 passed | 1 passed |
| Mutation run duration | 0.21s | 0.21s |

No focused skips or numerical failures. Removing the redundant assertion made
its NumPy import unused; the initial focused Ruff check found F401. Removed
that import and reran successfully. Final verification runs format/lint,
`uv run mypy src`, `git diff --check`, staged diff/protected-path checks and
an AST comparison excluding docstrings against the parent production module.
Record their actual final outputs in the plan before committing.

## Sensitivity evidence

A temporary copy of src/ replaced only
`same_time = time[index] == time` with
`same_time = (time[:, None] == time[None, :])[index]`.
This preserves the equal-time vector but constructs an N-by-N matrix inside
the loop. Repository source and all independent tests remained untouched.

Separate child processes set PYTHONPATH to the copied src/, set the requested
dtype mode, assert that survjax.__file__ points inside that copy, then run
only the new shape test and `test_mixed_cohort_has_hand_enumerated_counts`.
Both exit 1 with exactly `1 failed, 1 passed`: the shape assertion rejects
(7,7), while the mixed values remain correct. The reported shape set contains
(), (1,), (1,7), (7,), (7,1), and (7,7). This demonstrates the new check catches
memory regression even when a value-only anchor passes.

Optional raw diagnostics/source copy: `/private/tmp/survjax-a10-n2-4wfwe7up`.
The mutation, test names, output and counts above remain available in this
committed review if that temporary directory disappears.

## Limits and stop

The full through-A7 numerical gate was not rerun for this docstring/test-only
follow-up. The reviewer reran it on the unchanged numerical implementation;
that is historical reviewer evidence, not a new check claimed here. No full
through-A10 pass, R reference, docs build, wheel, benchmark or accelerator claim.
No score gradients are promised or tested. The shape guard uses the installed,
locked JAX 0.11.2 trace representation; no untested version matrix is claimed.

Stop after this follow-up commit on A10-cindex. Public integration still needs
its own refined plan after the separately owned alignment and sync land.
