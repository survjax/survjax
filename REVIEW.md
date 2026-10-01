# Review: DOC-3 tutorial-style SurvivalData docs and a clean repr

Branch: `docs-startup` (the checked-out maintainer branch, on top of DOC-2
`726e561`). No new branch was created.

## Read first

1. `src/chronox_jax/data.py`: the only code change. `_issue_bits` and
   `_may_contain` are declared with `field(repr=False)`.
2. `tests/unit/test_data_repr.py`: eager repr, repr under `jax.jit`, six leaves.
3. `docs/survival-data.md`: the rewritten guide. Read it rendered:
   `open docs/_build/html-x64/survival-data.html`.
4. `docs/playground.ipynb`, `docs/index.md` (one-line description rewrapped).

## Invariants

- The dataclass is `init=False`, so `field(repr=False)` without defaults changes
  only `__repr__`. `__init__`, `_assign`, `tree_flatten`/`tree_unflatten` and
  `eq=False` are untouched. `DataIssues` is unchanged.
- Guide cells have no asserts. The build fails on any unexpected exception
  (`nb_execution_raise_on_error`). Only cells tagged `raises-exception` may
  raise, and there are three: the `event`+flags `TypeError` and two
  `raise_if_invalid()` calls.
- One synthetic cohort (six patients, months, progression) runs through the
  page. Each censoring kind is explained for the patient before the call.

## Evidence

- `pytest tests/unit/test_data_repr.py`: 3 passed.
- `scripts/check.sh --through A2`: "all gates passed through A2" (ruff, mypy,
  own tests in both modes: 80 passed/3 skipped and 83 passed; spec tests
  through A2: x64 178 passed/13 skipped, x32 186 passed/5 skipped).
- `UV_CACHE_DIR=/private/tmp/chronox-docs-uv-cache scripts/build_docs.sh`:
  both x32 and x64 "build succeeded", no warnings.
- `git diff --check`: clean.

## Rendered output check (x64, x32 identical except dtype)

I extracted all 14 cell outputs from the generated HTML and compared each with
the prose around it:

| Section | Shown | Prose matches |
| --- | --- | --- |
| Time and status | repr with five public fields; upper `[4, inf, 7, inf]`; truncation 0/inf | yes |
| Between two visits | lower `[..., 6]`, upper `[..., 9]` | yes ("lower=6, upper=9") |
| Before first visit | lower 0, upper 3 for patient 6 | yes |
| Misuse | `TypeError: event cannot be combined with censoring flags` | yes |
| Endpoints | `from_bounds` upper equals the flag form's upper | yes |
| Truncation | `left_trunc [0,0,0,0,2,2]` | yes |
| Bad rows | `invalid_rows [T,T,T,T,F]`; `uninformative [F,F,F,F,T]`; `InvalidDataError` naming missing_time2, invalid_time, invalid_interval, censored_before_entry | yes |
| Status 2 / equal endpoints | `invalid_indicator [T,F]`; `invalid_interval [T]` | yes |
| `jax.jit` | `invalid_rows [F,T]`, then `InvalidDataError: invalid_indicator` | yes |
| Precision | `float64` (x64) / `float32` (x32) | yes |

Not shown on the page: the traced `raise_if_invalid()` `ValueError` and the
float64-unavailable `ValueError` are stated in prose only. The first is
covered by `tests/unit/test_data_issues.py::test_raise_if_invalid_is_host_only`.
I did not check the second in this step.

## R comparison sources

Re-checked against https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html
(fetched 2026-10-01):

- "(start, end]" is in the `time2` argument: "ending time of the interval for
  interval censored or counting process data only. Intervals are assumed to be
  open on the left and closed on the right, `(start, end]`." It covers interval
  censoring, so I kept the claim.
- 1/2 coding guessed via `max(status)==2`; `type="interval"` codes 0–3 and
  "`time2` ... is ignored unless event=3"; interval2 uses `(t,t)` for exact and
  "Infinite values can be represented either by actual infinity (Inf) or NA";
  defaults: "counting process if `time2` is present".

## Unsure about

- **New claim:** the table's "Delayed entry" row maps `Surv(start, stop,
  status)` to `SurvivalData(stop, status, left_trunc=start)`. This is an
  input-convention mapping for one row per subject. It does not cover
  multi-row time-varying data, and with no likelihood yet it is unverified.
  Drop it if you prefer DOC-2's narrower wording.
- R's interval2 writes left censoring as `(-infinity, t2)`. Chronox uses 0.
  The table says "0 for no lower limit" but does not mention R's `-Inf`.
- Rendered tracebacks show local source paths (`~/dev/chronox-dev/...`).
  That's acceptable for a local build.
- **Size:** the diff is about 336 changed lines (including `PLAN.md` and the
  new test), slightly above the ~300 guide, as the plan anticipated.
