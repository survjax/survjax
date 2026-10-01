# Review: DOC-4 readable reprs for SurvivalData and DataIssues

Branch: `docs-startup`, on top of DOC-3 `3e71f5b`. The spec does not cover
repr; this follows the approved DOC-4 plan with the maintainer's changes:
left censoring prints as the set `(0, 3]`, and each repr lives next to its
class (no separate `_repr.py`), accepting files over 300 lines.

## Read first

1. `src/chronox_jax/diagnostics.py`: `_plural`, `_rule_counts` (now shared by
   `raise_if_invalid` and the repr), `DataIssues.__repr__`.
2. `src/chronox_jax/data.py`: `_observation_text`, `_window_text`, the
   elision constants, `SurvivalData.__repr__`. Now 329 lines (was 252).
3. `tests/unit/test_data_repr.py`: expected strings written by hand.
4. `docs/survival-data.md`: prose updated to the new printouts.

## Invariants

- Both reprs check every leaf for `jax.core.Tracer` first and then print only
  shape (and dtype), so no tracer reaches `np.asarray`. Covered by
  `test_traced_reprs_show_only_shape` under `jax.jit`. Under `vmap` the same
  check applies (batch tracers are tracers) but there is no separate test.
- `repr=False` on both dataclasses only removes the generated `__repr__`.
  Pytree flattening, construction and equality are unchanged
  (`test_repr_change_keeps_six_leaves`). The private fields never print.
- The `raise_if_invalid` message is unchanged (same f-string over the same
  ordered rule list). A2 spec tests that match it still pass.
- Notation: exact `t`; `t+` when not exact, upper is `+inf` and lower is
  finite; anything else `(lower, upper]`, including invalid rows as stored,
  e.g. `(6, nan]`, `(9, 6]`. Numbers use `:g` (6 significant digits).
- The footer and `DataIssues` list `uninformative` separately and mark it
  "(not an error)" (spec 2/6.2).

## Tests and requirements

| Test | Shows |
| --- | --- |
| `test_repr_shows_each_observation_kind` | exact, `12+`, `(0, 3]`, `(6, 9]` (spec 3.2 sets) |
| `test_repr_shows_non_default_window` | `window (2, 8]` only on truncated rows |
| `test_repr_footer_counts_invalid_and_uninformative_rows` | invalid equal endpoints print as `(5, 5]`; footer counts |
| `test_repr_elides_middle_of_long_data` | 25 rows: first 10, `...`, last 5 |
| `test_repr_labels_two_dimensional_rows` | shape in header, index tuples |
| `test_traced_reprs_show_only_shape` | traced fallback for both classes |
| `test_valid_issues_repr`, `test_invalid_issues_repr_lists_firing_rules` | `DataIssues` verdict, rule lines, uninformative note |

## Checks run

- `scripts/check.sh --through A2`: "all gates passed through A2" (ruff,
  mypy, own tests and spec tests through A2, both dtype modes).
- `scripts/build_docs.sh`: both x32 and x64 builds succeeded.
- `git diff --check`: clean.
- Rendered x64 guide: extracted all 16 cell outputs and compared each with its
  prose. All match: the cohort printouts show `12+`, `(6, 9]`, `(0, 3]` and
  the `window (2, inf]` column. The bad-records printout shows the footer
  "4 invalid rows, 1 uninformative". `bad.issues` lists the four failing rules
  plus uninformative row 4.

## Unsure about / for the maintainer

- `docs/playground.ipynb` has uncommitted local changes I did not make. A
  JupyterLab session saved outputs (old-style repr from a kernel started
  before these changes), an empty trailing cell and full kernel metadata.
  There are no source edits. I left it out of this commit and did not revert
  it.
- An uninformative `(0, inf]` row prints as `0+` by the `t+` rule. That is
  correct (right censored at 0), but `(0, inf]` might read more clearly.
- Rule order in `DataIssues` follows the bitfield (`missing_time2` before
  `invalid_time`), not row order.
- `_plural` is private to `diagnostics.py` but imported by `data.py`.
- Diff is about 300 changed lines excluding `PLAN.md`.
