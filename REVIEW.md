# PH.5 — Executable proportional-hazards guide

Independent review: round 2/3

Step start and previous PH.4 final: 2a341c87b870f5f0baba8269e432db01aedc9f6a.
Original worktree launch: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-ph start; current AGENTS.md/CLAUDE.md read. The maintainer asked
for the next part after reviewed PH.4. This guide targets that committed /
reviewed code snapshot; no merge or PH-wide completion is claimed here.

## Read first

1. PLAN.md PH.5; spec 4.4,5,6,7.3,8.7,10.5,12.1–12.2, revision 40.
2. docs/proportional-hazards.md from top to bottom: interpretation /scalar
   baseline, draws x arms, likelihood and ordinary Hessian, delegated RMST,
   numerical limits/diagnostics. Eight executable cells plus one plot.
3. docs/api.md: only the three-line link from its owned PH section.
4. Existing D-031/D-032 entries for provenance of the two DISTINCT numerical
   limitations. Their formal status is unchanged and maintainer-owned.
5. Executed output and inspection evidence below.

## Scope and navigation

Docs-only, one PH topic: final whole-step diff is 247 docs lines: 238 new
guide lines +3 API-link lines +4 additions/2 removals in the existing PH
availability paragraph. Round 1 had 240 lines; round 2 corrects that paragraph
and scopes the general derivative sentence. PLAN/REVIEW excluded as agreed.
No src/, docstring, test, fixture, dependency, config, protected-path, spec,
appendix/PDF, playground or shared-index change. No new API, AFT, sampling,
generic summary, inference/fitting system or dependency. Matplotlib is already
in the locked docs group, and only the guide imports it.

The page's local `orphan: true` metadata keeps it outside the shared root TOC
pending maintainer/A9 navigation integration (release packet serializes
index/conf/playground edits). It does not change global warning handling.
The public API section links to the guide NOW, and the guide links back to
that API section, so it is reachable. Do not interpret the missing root TOC
entry as complete navigation integration. A maintainer/A9 follow-up can add
`proportional-hazards` beside the other guides and remove orphan metadata.
No other agent was messaged or its files edited.

## Requirement and example map

- Spec 4.4: explains eta is dimensionless, eta>0 means higher hazard, covariate
  mapping stays with caller, baseline is positional/keyword and eta keyword-
  only. Exponential r=0.5,eta=log(2) has h=1, S=exp(-t). First cell shows
  baseline /PH survival and PH hazard at months [0,1,2], and asserts S(0)=1.
  Plot uses the same physical comparison, not fitted/patient data. No R or
  external numerical agreement is claimed.
- Spec 4.1/4.4/5: hypothetical baseline (3,1) x predictor (3,2) -> batch (3,2).
  K=S=3, survival_grid shape (3,3,2), with first time row exactly ones. The
  zero row also guards axis order when dimension lengths coincide. Indexing
  draw zero retains two arms. Text distinguishes pointwise trailing broadcast
  from a shared leading time grid; links to existing grids/family guides.
- Spec 6/7.3: synthetic exact event t=1 plus right censor t=2, no truncation.
  The independent NLL is 3*exp(eta)-eta, so at eta=0 value=3, first=2,
  second=3. A jit(value_and_grad) and jit(grad(grad)) cell displays these and
  asserts finiteness. Guarantees are scoped to built-in baselines and ordinary
  valid smooth interiors. Unsupported likelihood rows are explicitly NaN.
- Spec 4.4/8.7: RMST is a time-unit integral; rate-1 case M(2)=1-exp(-2).
  A separate expression -expm1(-2) is shown beside the result. The physical
  law/integral supplies the reference, not another call to RMST. Table gives
  r_PH=r*exp(eta) and b_PH=b*exp(-eta/k). Text explicitly says RMST delegates
  privately while direct methods/likelihood keep PH formulas. Custom without
  supported closure raises; private hook is NOT public extension API. Only
  staged rmst is available among summaries; remaining summary stubs raise.
- Spec 8.7: explicit cutoff axes [:,None,None] -> (3,3,2), zero row exact,
  finite results. States cutoff-invalid/parameter-issue NaNs, M(0)=0,
  parameter tangents 0 and cutoff tangent 1, dM/dtau=S. Distinguishes Weibull
  higher shape restriction (generally raises, some orders/tails may succeed
  correctly) from supported nonshape interior derivatives; advises keeping
  shape OUTSIDE differentiated arguments. Weibull zero guarantees only value /
  first derivatives. No new tolerance or higher-boundary claim.
- Spec 4.7/7.1: negative baseline scale cell gives [False,True] invalid mask,
  `baseline.scale` field prefix, finite valid survival and NaN invalid one.
  Explains host-only explicit raise_if_invalid and construction's stored values.
- Spec 7.3, revision 40 /D-031: directly documents extreme PH baseline/eta /
  mixed first/second derivative failure even with accurate values and finite
  true derivatives, includes the spec's float64 stress example, and says no
  sharp safe domain is known. Post-release stabilization; ordinary contracts
  unchanged; this exception DOES NOT weaken RMST.
- D-032 accepted decision, not a new exception: explicit float32 k=.2,b=1,
  eta=-18,tau=1 in BOTH builds produces clean original issues and NaN RMST.
  It is outside the mandatory cutoff/equivalent-scale accuracy range; e^90
  overflows while true M(1) is close to 1. Describes detectable numerical
  failure, distinct from invalid inputs and from the direct-derivative limit.
  No global x64 toggle, probability floor, changed model or diagnostic flag.

No ad hoc numeric tolerance is added. Shape/boundary/finite/issue assertions
are exact checks of documented facts. Rounded numerical outputs are read
against the stated analytical derivations; this guide does not replace the
existing high-precision numerical regression tests.

## Actual validation

- `scripts/build_docs.sh` invoked DIRECTLY, without redirection, environment
  assignment or shell wrapper, twice: initial draft **exit 0 in both modes**;
  final wording clarification **exit 0 in both modes**, each with fresh forced
  execution and strict -E/-n/-W /raise-on-notebook-error behavior. Both passes
  read 12 source pages. Final PH x64 execution reported 4.30 seconds. No
  error/warning suppression or script/config modification.
- Final HTML is in docs/_build/html-x32/proportional-hazards.html and html-x64.
  HTMLParser inspection confirms all sections, 8 executed outputs per mode,
  expected float32 /float64 ordinary output, (3,3,2) time/draw/arm shapes,
  and reciprocal API/guide links. Checked final built-in scope /unsupported
  likelihood wording in the generated HTML, not just source.
  - ordinary loss output is (3,2,3) in each mode;
  - scalar RMST /independent expression: (0.86466473,0.86466473) float32,
    (0.86466472,0.86466472) float64;
  - invalid scale prints the baseline.scale mask and [finite,nan];
  - final failure is (True,nan), result dtype FLOAT32 even in the x64 build.
- Figure rendered and visually inspected via view_image: baseline blue above
  PH orange for t>0, common survival 1 at zero, labeled month/probability axes,
  clear legend and no clipping/overlap. Both modes produce byte-identical PNG
  in their _images directories:
  90a1e0a54cebfb8edac31f4086b1df2980160d95789160e8d40efbbeb4c6a12e.png.
- `uv run ruff format --check .`: 136 files formatted;
  `uv run ruff check .`: all checks passed;
  `uv run mypy src`: no issues in 22 files;
  `uv run pytest -q tests/unit/test_api_reference.py`: 1 passed in 0.41s;
  `git diff --check`: clean. No assertion/reference-test change.
- Scope audit `git diff --exit-code -- src tests pyproject.toml uv.lock AGENTS.md CLAUDE.md scripts .agents .workmux.yaml docs/index.md docs/conf.py docs/playground.ipynb`:
  exit 0. No library/dependency change. No full numerical/wheel gate rerun in
  this DOCS-ONLY step; PH.4's final source/wheel gate covers the identical code.
  No R, fixture generation or appendix/PDF rebuild. No generated HTML, PNG,
  notebook output or execution cache is committed.

## Round 1 response

- F-1 (should-fix): docs/distributions.md's two-line "PH and AFT planned"
  claim is stale after the public export. Correct only that PH availability
  paragraph: distinguish the local nesting example /hazard_ratio from the
  supported survjax.ProportionalHazards /log_hazard_ratio, link the new guide,
  leave AFT planned. The recipe's Python, its R/reference text and all other
  sections are untouched. Ownership handoffs reserve A10 concordance and A9
  verification/playground/nav work, not this narrow PH-specific correction
  in already-committed material (last file change 726896d).
- N-1 (nit): explicitly restrict the general second-AD sentence to direct
  methods and likelihoods. It no longer could imply Weibull RMST shape-Hessian
  support; the RMST section still states its separate derivative contract.
- N-2 (nit/handoff): root TOC integration remains assigned to maintainer/A9;
  retain local orphan metadata and the working API link. No change here.
- Reviewer independently reproduced the concrete numerical/policy examples
  in both dtypes and reran the docs build successfully in round 1. No code
  or deviation-policy question was raised; D-031/D-032 remain unchanged.

Round-2 final validation:

- `scripts/build_docs.sh` invoked DIRECTLY: **exit 0, both x32 and x64 builds
  succeeded**, strict flags and forced notebook execution unchanged. All
  12 source pages rebuilt; final PH x64 execution reported 4.20s.
- HTMLParser verified F-1's corrected paragraph and new extension-guide link
  in BOTH rendered distributions.html pages. The PH page's derivative wording
  now explicitly names direct methods/likelihoods in both outputs. API and
  extension links reach the guide; all eight executed outputs remain, with
  correct dtype, leading grid shape and explicit float32 NaN failure.
- Figure rechecked: both modes have identical 490x290 PNG, SHA256
  90a1e0a54cebfb8edac31f4086b1df2980160d95789160e8d40efbbeb4c6a12e,
  unchanged from the previously visually inspected figure.
- `uv run ruff format --check .`: 136 formatted; `uv run ruff check .`: all
  checks passed; `uv run mypy src`: no issues in 22 files;
  `uv run pytest -q tests/unit/test_api_reference.py`: 1 passed in 0.26s;
  `git diff --check`: clean. No code or test changes, so no numerical /wheel
  gate rerun. No generated output committed.
- Whole-step diff against PH.4 base: 238+3+4+2 = **247 documentation lines**,
  within budget. New source constraints unchanged; only corrective prose/link
  edits in round 2. Commit fix-up and invoke independent-review round 2/3.

## Remaining work

Maintainer/A9 root navigation integration remains, then final PH evidence /
acceptance audit (independent PH spec tests have not yet been supplied).
Current explicit numerical limits are not resolved by documentation; their
existing formal deviation status stays open. No unsupported future feature
is presented as working. Commit, invoke independent-review PH.5 exactly and
stop/report verdict, findings, final/previous/original SHAs and remaining work.
No merge, push, publish, protected edit or unrequested cross-agent operation.
