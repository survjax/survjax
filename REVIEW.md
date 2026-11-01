# PH.6 — PH guide discovery from documentation home

Independent review: round 1/3

Step start /previous PH.5 final: 414fa845e121eed9ff1cc09f991b920193bdef08.
Original worktree launch: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-ph start; current AGENTS.md/CLAUDE.md read before edits.

## Read first

1. PLAN.md PH.6. The maintainer confirms A9 was already merged and removed,
   explicitly assigns this navigation follow-up HERE, and flags the absence
   of any docs-home PH link. Old A9 ownership deferral is superseded.
2. docs/index.md diff: feature intro, direct PH-guide description/link,
   toctree entry immediately after builtin-families, custom-recipe link label.
3. docs/proportional-hazards.md: ONLY removal of orphan: true.
4. Spec 4.4/8.7/10.5/12.2 and generated-link checks below.

## Exact scope and invariants

One small docs-navigation topic, 15 changed docs lines (index +10/-4,
new guide metadata -1); PLAN/REVIEW excluded under the current agreement.
No library, API, test, dependency, config, playground, spec, deviation,
protected-path or guide-body change. No new numerical test that merely
mirrors a Markdown link. No A9 worktree reopened, no merge/push/publish.

The home intro now mentions the implemented built-in PH wrapper and staged
RMST only on Exponential/Weibull and their PH wrappers, as spec 4.4/8.7 require.
Other summaries remain planned. The obsolete/unverified "Slice A completion
gate has not yet been met" sentence is removed, NOT replaced by a claim that
A9/Slice A/the whole release is complete. This change is current capability
and navigation documentation; final PH evidence audit remains separate.

The home guide-list paragraph links directly to proportional-hazards.md and
summarizes the built-in wrapper's predictor, shapes, likelihoods, restricted
means and derivative limits. The existing extension recipe remains linked,
with its label clarified as the custom-distribution guide. Root toctree now
includes PH beside the built-in family guide, giving the sidebar and ordinary
previous/next page relations. PH.5's N-2 navigation handoff is addressed HERE.

The maintainer is still reviewing the guide, so preserve the WHOLE source
except the orphan marker, not merely the code blocks. Verified bytes against
`git show HEAD:docs/proportional-hazards.md` before commit (HEAD was the step
base): deleting exactly `orphan: true` gives the working-tree bytes. No
sentence, example, assertion, numerical limit or other front matter changed.
The previously reviewed eight executable cells and figure remain intact.

## Actual checks and outputs

- `scripts/build_docs.sh` invoked DIRECTLY, without redirects, environment
  assignments or shell wrappers: **exit 0; x32 and x64 builds succeeded**.
  Strict -E/-n/-W and notebook-error settings unchanged, 12 source pages read
  with fresh notebook execution. PH x64 execution reported 4.60 seconds.
- Parsed BOTH generated index.html main articles with HTMLParser, distinguishing
  article prose, article toctree and out-of-article sidebar anchors. Each has
  the PH target THREE times, one in each required context:
  [(outside article,not toc), (inside article,not toc), (inside article,toc)].
  This verifies actual HOME BODY links, not merely the API or sidebar link.
- Each guide's generated head has prev=builtin-families.html and
  next=likelihood.html, proving normal toctree relations/order. Eight executed
  cell-output containers remain in each guide. The existing PH/API links are
  preserved; no new orphan warning/suppression introduced.
- Guide figure SHA256 remains
  90a1e0a54cebfb8edac31f4086b1df2980160d95789160e8d40efbbeb4c6a12e
  in BOTH builds, identical to the visually inspected PH.5 figure. No plot or
  guide body edited. Generated HTML/notebooks/images/cache remain uncommitted.
- Exact source-byte comparison: guide equals old bytes minus the orphan line.
- `uv run ruff format --check .`: 137 files formatted;
  `uv run ruff check .`: all checks passed;
  `uv run mypy src`: no issues in 22 files;
  `uv run pytest -q tests/unit/test_api_reference.py`: 1 passed in 0.30s;
  `git diff --check`: clean.
- Scope audit `git diff --exit-code -- src tests pyproject.toml uv.lock AGENTS.md CLAUDE.md scripts .agents .workmux.yaml docs/conf.py docs/playground.ipynb`:
  exit 0. No new dependency or runtime change. No core numerical/wheel gate
  rerun for this docs-only navigation change; PH.4 covers the same library.
  No R, fixture regeneration or appendix/PDF rebuild.

## Remaining work

No unresolved navigation ownership or remaining A9 implementation is implied.
The maintainer's human review of the guide continues without body churn.
Next separate task: final PH evidence/acceptance audit, including independent
PH spec tests when supplied. No full-release completion claim is made here.

Commit, invoke independent-review PH.6 exactly, address blocker/should-fix
findings within three rounds if needed, then stop/report verdict, findings,
final/previous/original SHAs and remaining work. No merge or publication.
