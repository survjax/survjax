# PH.4 — Public PH exports and installed API integration

Independent review: round 1/3

Step start and previous PH.3d final: 5e3868094391e0c5464c92d3775c42902c8ab2cb.
Original worktree launch: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Current AGENTS.md/CLAUDE.md read before work; clean release-ph start.

## Read first

1. PLAN.md PH.4 and spec 4.4,8.7,10.2,12.1–12.2 (revision 40).
2. src/survjax/distributions/__init__.py, then src/survjax/__init__.py:
   import and export the SAME implemented PH class, no new implementation.
3. tests/unit/test_import.py: exact root exports plus identity across all
   three paths and an exact distribution export set.
4. tests/install/test_wheel.py: exact installed exports and public PH eager /
   jit construction/RMST smoke in an isolated environment with runtime deps only.
5. docs/api.md's seven-line block; unchanged test_api_reference.py; the
   code-literal correction in the now-rendered PH class docstring.
6. Actual validation / rendered HTML inspection below.

## Authorization and scope

The maintainer was presented /private/tmp/ph4-api-reference.patch, exactly
seven new docs/api.md lines, and asked whether to apply it alongside public
exports/tests. After an intervening question about wrappers.py (keep it:
AFT is planned for Slice D), the maintainer replied “ok good; now on to the
next bit”. That approves the described next PH export step and its specific
code-step docs exception. Only those seven docs/api.md lines are edited;
no other docs/ source, navigation or build settings. The candidate integration
patch was prepared/syntax-linted before approval, applied only after PLAN.md.
No broader docs permission, publication or spec change is inferred.

The public API addition is exactly one name: ProportionalHazards. Root __all__
has 14 names, distribution __all__ 8. Existing exports all remain; root,
distribution and defining-module names identify the SAME registered frozen
pytree class. Constructor behavior (positional or keyword baseline, keyword-
only log_hazard_ratio) and all calculations remain unchanged. Registering the
class is not duplicated. The only wrapper edit corrects RST markup in its
class docstring: ``(K, *batch_shape)`` was previously parsed as unclosed
emphasis. This is necessary for strict autodoc to build, not a numerical edit.

No private closure/numerical file, dependency, tolerance, fixtures, protected
path, test_api_reference.py or tests/spec edit. No AFT, generic summaries,
sampling, new module, alternate public name or lazy export mechanism. Keep the full unfinished
PLAN history. PH.3d S-1 (spec layout still names the old helper) remains for
the maintainer/designer, with no spec edit or new interpretation here.

## Requirement and test map

- Spec 4.4/10.2: test_public_package_imports keeps exact root export equality,
  adds only PH. test_ph_public_imports_are_the_same_registered_family proves
  root/distribution/defining-module identity and exact distribution exports.
  These are fixed expected sets, not sets derived from implementation output.
- Spec 10.2/10.5: the API reference test is UNCHANGED and still requires an
  autodoc entry for EVERY root public export. The PH block passes it. No
  missing-name allowlist, future-feature skip, conditional docs or weakened
  assertion. Rendered HTML shows the public constructor and all six named
  method targets (rmst, reference_time, log_survival, log_hazard, log_prob,
  log_cdf); inherited direct/grid APIs remain documented on the base page.
- Spec 10.3/12.1–12.2: installed-core smoke runs python -I outside the source
  tree, checks package location, py.typed, exact root exports, runtime metadata
  and absence of demo/dev imports. It checks the root/distribution PH identity,
  accepts a POSITIONAL baseline, and exercises eager RMST plus constructing
  PH and evaluating RMST inside jit. It emits both scalars in JSON and compares
  outside the installed environment with unchanged impl_tolerances constants.
- Independent installed oracle: r=0.5, eta=log(2), so effective rate is 1.
  Physical survival is exp(-t), M(2)=integral_0^2 exp(-t)dt=-expm1(-2).
  Expected value comes from NumPy's exponential integral identity, never from
  wrapper or native RMST output. Existing ATOL/RTOL, no new tolerance.
- No extra first/second-AD test of unchanged arithmetic is added. Full
  existing dedicated PH construction/direct/grid/likelihood/RMST/derivative/
  isolation/tail suites run in the cumulative gate. Identity of the public
  class means those tests cover the exported implementation too.

## Actual validation

- `JAX_ENABLE_X64=1 uv run pytest -q tests/unit/test_import.py tests/unit/test_api_reference.py`:
  **4 passed in 1.80s**. Separate JAX_ENABLE_X64=0: **4 passed in 1.80s**.
- `uv run ruff format --check .`: 135 files formatted;
  `uv run ruff check .`: all checks passed;
  `uv run mypy src`: no issues in 22 source files; `git diff --check`: clean.
- First `scripts/build_docs.sh` invocation, DIRECT with no redirection/env /
  wrapper: **exit 1** on a single Docutils unclosed-inline-emphasis warning
  from PH's class-docstring shape. All notebooks executed, but x64 was not
  reached. The script's strict -n/-W flags were not changed or bypassed.
- Fix only that shape's RST literal markup, then rerun `scripts/build_docs.sh`
  DIRECTLY: **exit 0; build succeeded in BOTH x32 and x64**, forcing executable
  notebooks on both passes. No Sphinx warning suppression, dependency file or
  other docs-source edit. Generated outputs docs/_build/html-x32 and html-x64.
- Inspected the generated API HTML in both outputs with HTMLParser, not just
  source text. Class and six method IDs present. The rendered constructor is
  `ProportionalHazards(baseline, *, log_hazard_ratio)`, higher-hazard convention
  is shown, and the ``(K, *batch_shape)`` literal retains its star/underscore.
  First inspection assumed a section tag for the following heading; Sphinx
  actually emits a span anchor there. Corrected the inspection selector;
  no product-source change was needed.
- Initial `scripts/check_wheel.sh --through A10` passes, with a wheel built
  before the quote-only docstring fix. Log /private/tmp/ph4-wheel-gate.log.
  That initial run passed: own 1282/8 x64 (134.33s),1287/3 float32 (114.82s),
  doctests 4 each, independent 2332/28 x64 (66.63s),2322/38 float32 (62.54s).
- FINAL `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed**
  on final matching-source wheel and final tests/docstring. No source/test/doc
  changes during this final gate:
  - own x64 **1282 passed, 8 skipped in 116.42s**;
  - own float32 **1287 passed, 3 skipped in 169.68s**;
  - doctests **4 passed** per mode (1.39s /1.26s);
  - independent acceptance x64 **2332 passed, 28 skipped in 91.75s**;
  - independent acceptance float32 **2322 passed, 38 skipped in 81.21s**.
  Final log /private/tmp/ph4-final-wheel-gate.log. Retained wheel:
  /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.DdnQr9/survjax-0.0.1.dev0-py3-none-any.whl;
  SHA256 e9aa003d19c07d4290041a300a45288b7b8a942982dae204ceab57b4f19ec46d.
  ZIP inspection confirms both export modules AND the corrected wrapper
  docstring match current source bytes. Environment unchanged: Python 3.12.9,
  JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1; Sphinx 9.1.0 in the locked docs group.
  No dependency file change or new dependency; docs command used existing group.
  Initial version/ZIP probe used system python3 and failed before any change
  (jax unavailable there); reran under uv successfully.

Final implementation/test/doc diff: **51 changed lines**, including only seven
in docs/. PLAN.md /REVIEW.md are excluded under the current working agreement.

## Remaining work and limits

The public surface is now wired; this is not the PH release-completion claim.
The separate executable PH topic and final evidence/acceptance audit remain.
That topic must cover eta direction, broadcasting, direct/grid/likelihood
use and RMST, supported derivatives, the shape/zero-boundary limits, revision-
40 extreme PH direct/likelihood derivative limits and accepted D-032 failure
behavior. Existing optional numerical/test nits remain recorded separately;
no new numerical claim or tolerance relaxation is made by public exposure.
No R, spec fixture regeneration or appendix/PDF rebuild is required in this
step; no corresponding artifact changed. After checks, commit, invoke the
independent-review skill exactly for PH.4, then stop/report its verdict and
final/previous/original SHAs. No merge, push or publication.
