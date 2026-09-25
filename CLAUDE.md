# Chronox (`chronox-jax`): working agreement for coding agents

Chronox is a lightweight JAX library for survival likelihoods, predictions and
evaluation. Distribution name `chronox-jax`, import `chronox_jax as cx`. There is
no top-level `chronox` import.

The goals are in `spec/GOALS.md`. The normative contract is
`spec/SPECIFICATION.md` (revision 4). **Both are read-only for you.** If the spec
is ambiguous, wrong or awkward to implement, do not fix it and do not silently
choose: log it in `DEVIATIONS.md` (spec section, what you did, why) and ask.

The spec's links to REVIEW-DISPOSITIONS.md, claude-review.md, `docs/design/` and
`notes/` point into the separate design repo (`chronox-design`). They are
expected to be dead here. Do not go looking for them.

Current target: **Slice A** (spec section 12.2). Do not implement anything
belonging to slices B to F. Unimplemented observation modes must fail
explicitly, never return plausible values (section 12.2).

## How we work (a human reads every line)

The maintainer reviews all code, so optimise for reviewability over speed.

1. **Plan first.** Before code for a step, write or update `PLAN.md`: files to
   create or change, public API touched, tests you will add, spec sections
   covered. Then **stop and wait for approval.**
2. **Small steps.** One concept per step, ending in one commit. Aim for a diff
   of about 300 lines or less, tests included. If a step grows past that,
   split it and say so.
3. **Stop after every step.** Write `REVIEW.md` (overwrite it each time):
   - what to read first, in order;
   - the invariants the code relies on;
   - which tests demonstrate which spec requirement;
   - anything you are unsure about.
   Then stop. Do not begin the next step until told.
4. **Tests come with the code**, in the same commit, never "later".
5. **No scope creep.** No dependencies, abstractions, helper layers, config
   systems or features beyond the spec without asking. Three similar lines beat
   a premature abstraction.
6. **Report honestly.** State exactly which checks you ran and their output. If a
   check was not run (for example R is unavailable), say so. Never claim a
   property is verified because it is plausible.
7. **Commits:** imperative subject line, body says why. One step per commit. No
   force-pushes, no history rewriting, no publishing, no remote setup or
   credentials (spec 12.2: separate actions).

## Code style (optimised for a human reader)

- Plain functions and small frozen dataclasses registered as pytrees. No
  metaclass tricks, no dynamic attribute magic, no decorator stacks.
- Explicit is better than compact. Name intermediate quantities.
- Type-annotate all public APIs, pytree fields, result containers and extension
  interfaces. No shape-typing library, no runtime type-checking decorators
  (spec 10.3). Document array shapes and the time-first grid convention in
  docstrings.
- Files under about 300 lines, one responsibility per module. Create modules only
  when the current step needs them (spec 10.2).
- Comments explain *why* and cite the spec (`# spec 7.2: substitute before
  operating`). Do not restate what the code does.
- Public API is exactly what `__init__.py` exports. Numerical helpers stay in
  underscore modules.

## JAX and numerical rules (from spec sections 3.3, 4.5, 7)

These are the places where plausible code is quietly wrong. Re-read them before
each numerical step.

- **No floors, no epsilons, no clipping.** Never floor `Z`, clip probabilities
  upward, add an epsilon to times, or turn a narrow interval into a density.
- **Safe inputs before each operation.** `jnp.where` on the final value does not
  remove NaN gradients from the unselected branch, and under `vmap` all branches
  run. Substitute safe inputs *before* the operation (the double-`where` pattern),
  including inside helpers. `stop_gradient` goes on substituted values only,
  never on real parameters or active calculations (spec 7.2).
- **Work in log space**: `log1mexp`, `expm1`, `logaddexp`, stable log
  differences. Never subtract nearly equal powers.
- **Failure taxonomy (spec 7.1):** true zero mass gives `-inf`; invalid input or
  parameters give NaN plus an issue flag; precision loss is never relabelled as
  zero.
- **Direct methods are total functions of `t`** and never raise (spec 4.5).
  Observation *validation* is a different, stricter rule (exact times must be
  `> 0`). Do not conflate them.
- **Never convert tracers to NumPy** (`np.asarray(tracer)` and friends). NumPy is
  for host-side inspection and fixed constants only.
- **Never change `jax_enable_x64` globally** in library code. Tests and the
  runner set it, and the library follows it.
- **Integer counts are cast to the float dtype before division** (silent float32
  from `int32 / int32`).
- **Static metadata must not cause recompiles** when values change. This is
  tested (compile-count tests).
- Gradients are only guaranteed w.r.t. parameters, on valid smooth interior
  cases, to second order (spec 7.3). Do not promise more, and do not test more as
  though it were guaranteed.

## Repository layout

```text
spec/                     GOALS.md, SPECIFICATION.md (copied, read-only)
CLAUDE.md                 this file (AGENTS.md is a symlink to it)
PLAN.md  REVIEW.md  DEVIATIONS.md
pyproject.toml  uv.lock  .python-version  .pre-commit-config.yaml
.gitignore  LICENSE
src/chronox_jax/          layout of spec 10.2; only modules Slice A needs
tests/
    conftest.py           dtype handling, shared helpers
    tolerances.py         the only place tolerances are defined
    unit/                 hand-derived and analytical cases
    property/             hypothesis identity tests
    reference/            comparisons against committed fixtures
    fixtures/             committed synthetic data and R outputs (see below)
    install/              wheel/sdist smoke test, run outside the source tree
scripts/
    run_tests.sh          runs the suite once per dtype mode (see below)
    fixtures/             R generator scripts (only these need R)
examples/                 tiny synthetic example for Slice A
docs/
```

## Environment and tools (spec 10.1, 10.3)

- `uv` for environments and dependencies, `uv_build` as build backend. Python
  3.12, CPU only. Runtime deps: `jax`, `numpy` only. Test deps: `pytest`,
  `mpmath`, `hypothesis`. Dev deps: `ruff`, `mypy`, `pre-commit`.
- Ruff rule families `E W F I B C4 UP`, ignore `E501`. Pre-commit hooks as in
  spec 10.1, consistent with any CI. mypy on `src/` with checks on typed
  function bodies.
- Select mutually compatible versions at setup and record them in the lockfile.
  Justify any lower bound by a check. Do not advertise an untested Python/JAX
  matrix. Report the versions you chose.
- Regular tests never need network, R or locally acquired data.
- The core package must import and run without any demo dependency.

Standard commands (keep these working at every commit):

```bash
uv run ruff format --check . && uv run ruff check .
uv run mypy src
scripts/run_tests.sh            # both dtype modes
```

## Testing conventions

**Two dtype modes, separate processes.** JAX's 64-bit mode cannot be toggled
reliably inside one process. `scripts/run_tests.sh` runs the whole suite twice:
once with `JAX_ENABLE_X64=1` and once without. Tests read the mode from the
environment via a fixture in `conftest.py` and never set it themselves. Tests
that are only meaningful in one mode are skipped in the other with an explicit
reason (for example, mpmath tail checks at float64 precision).

**Tolerances live in `tests/tolerances.py` only** (spec 12.1):

| dtype | `atol` | `rtol` |
| --- | --- | --- |
| float64 | `1e-10` | `1e-8` |
| float32 | `1e-5` | `1e-4` |

Derivative and tail tests need their own tolerances, each with a comment
justifying the scale. When probabilities underflow, compare log probabilities.
Never loosen a tolerance to make a test pass. Report the failure instead.

**Where expected values come from.** Every expected number has a stated origin in
a comment or the test name: hand derivation (show the algebra in the comment),
mpmath at high precision, an analytical identity, or a committed R fixture.
**Never derive an expected value by running the code under test.**

**Structure of a test.** One behaviour per test, named for the behaviour
(`test_weibull_shape_one_matches_exponential`), with the spec section in the
docstring. Prefer several small explicit cases to one large parametrised blob
that hides which case failed.

**Transformation checks** for anything differentiable or jit-able:

- eager, `jax.jit` and `jax.vmap` agree;
- `jax.test_util.check_grads(..., order=2, modes=("fwd", "rev"))` on interior
  cases; Hessian where the spec requires it;
- likelihood suite run once eagerly under `jax_debug_nans` on valid mixed
  batches;
- boundary cases from the spec (Weibull `k=1` on a grid starting at `t=0`; the
  gradient of `S(0)` in `k`) must give finite gradients.

**Property tests** (hypothesis) use `derandomize=True` or a fixed seed so runs
are reproducible. They express the identities in spec 12.1, not implementation
details.

**Fixtures (spec 10.4).**

- Synthetic data only. No patient rows, no row-level output derived from patient
  data, ever.
- Each fixture set is a directory with: data (CSV), expected outputs (JSON with a
  header recording seed, R/package versions, and generating script name), and a
  short README. Generator script lives in `scripts/fixtures/`.
- **You cannot regenerate R fixtures in this environment unless R 4.4.2 with
  flexsurv 2.3.2 and survival 3.7-0 is actually available.** If it is not, write
  the generator script and its README, then stop and hand it to the maintainer
  to run. Never write fixture values by hand or from memory, and never
  approximate them with Python code.
- Fixture loading goes through one helper in `tests/conftest.py`. Tests never
  parse files ad hoc.

**Wheel smoke test** (`tests/install/`): build the wheel, install it in a fresh
environment outside the source tree, check imports, exported names and the
`py.typed` marker, and confirm the core imports without demo dependencies.

## Slice A: suggested step order

`PLAN.md` may refine this, but keep the steps this small. Each ends in a commit,
a `REVIEW.md` and a stop.

| Step | Content | Spec |
| --- | --- | --- |
| A0 | Scaffold: git init, uv project, tools, pre-commit, empty package with `py.typed`, `run_tests.sh`, one import smoke test | 10.1 to 10.3 |
| A1 | `SurvivalData` fields, pytree registration, dtype policy | 3.2, 3.3 |
| A2 | Constructors (flag and bounds), boundary rules, `DataIssues` bitfield | 3.1, 3.3, 6.2 |
| A3 | Indexing and slicing, static metadata, compile-count test | 3.2 |
| A4 | Distribution base: `register_family`, param and static handling, `log_survival` plus `reference_time`, autodiff `log_prob` default, direct-method value policy | 4.1, 4.2, 4.5 |
| A5 | `Exponential` and `Weibull` closed forms, boundary values, survival grids | 4.1, 4.5, 5 |
| A6 | Exact and right-censored likelihood without truncation, result containers, explicit failure for unimplemented modes | 6.3, 7.1, 7.2 |
| A7 | Hand-derived, tail and boundary tests; mpmath checks; `jax_debug_nans` run | 7, 12.1 |
| A8 | R fixture generators and committed fixtures (may require maintainer to run R), reference tests | 10.4, 12.1 |
| A9 | Derivative and transformation suite; wheel smoke test; tiny synthetic example | 7.3, 10.3, 12.2 |

Slice A is done only when every item in its completion gate (spec 12.2) has
evidence, and the final `REVIEW.md` lists each gate item with the test that
covers it. Passing Slice A is not evidence for later slices.
