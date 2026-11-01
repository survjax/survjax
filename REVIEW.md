# PH.3d — Rename the RMST-only private helper

Independent review: round 1/3

Start and previous PH.3c final: 295834c612351c8ea4cff106f7b0208c446674e2.
Original worktree base: 8757c0fda78e34e6e818249adedd2a144a33c33d.
Clean release-ph before this step; current AGENTS.md/CLAUDE.md read. Preserve
all unfinished PLAN history, including historical references to the old path.

## Read first

1. PLAN.md PH.3d and the maintainer's explicit request to give _numerics.py a
   focused RMST name. Spec 10.2 lists the old illustrative private filename;
   this requested rename supersedes it. No spec edit or new responsibility.
2. `git show --find-renames`: _numerics.py -> _rmst.py is byte-identical.
3. Imports in distributions/exponential.py, weibull.py and wrappers.py.
4. appendices/README.md and appendices/weibull-rmst.md: correct new links,
   current log-ratio/rescaling explanation; companion PDF rebuilt and viewed.
5. Existing native/PH RMST suites and installed-wheel gate evidence below.

## Invariants and scope

_rmst.py names the shared private RMST implementation:
cutoff shape/dtype handling plus staged Weibull RMST. Nothing else is moved
into it; _log_math remains unchanged. The actual filename is _rmst.py.
No helper signature, calculation, branch, dtype, derivative rule, public
export, dependency, tolerance or exception behavior changes. No compatibility
shim for a private unexported path. Three family/wrapper imports are the only
production edits beyond the move. The existing module's SHA256 before and
after the move is 6fd938fe83c43fab859601b59842900e50c94df1bdee55fd07326b4b0d9a6030.
Compared file bytes against `git show HEAD:src/survjax/_numerics.py`, not just
an AST or a passing test. The wheel's _rmst.py bytes match the source too;
_numerics.py is absent from the wheel. No new test solely asserts a filename.

The appendix is permanent mathematical documentation outside docs/ (as
previously requested), not workstream history. New text explains the two
algebraically equivalent log-ratio routes already reviewed in PH.3c:
- separated inputs use log(tau)-log(b) to avoid a full ratio;
- nearby inputs use log1p((n-m)/m), with b=m*2**e, n=tau*2**(-e), the SAME
  exponent for both operands, and O(1) denominator m;
- the nearby condition places the ratio inside [1/2,2], permitting exact
  normal-float subtraction (Sterbenz); safe inactive operands are substituted
  before frexp/ldexp/division/log1p, and real parameter tangents remain active.
Existing series/gamma/mean bounds and shape/zero restrictions are unchanged.
Removed the old hard page break before derivatives: the new paragraph would
otherwise leave an almost empty fourth page and create a fifth. The final
four-page PDF has the same section hierarchy with continuous pagination.

No standard docs/, tests/spec, protected scripts/config, spec, deviations,
fixtures, tests or exports are edited. The maintainer requested continuation
after this naming correction; next export implementation still depends on
explicit narrow docs/api.md permission, requested with a concrete patch.
The working-tree docs/api.md is UNCHANGED. The candidate seven-line diff is
/private/tmp/ph4-api-reference.patch, outside the commit. No approval is
inferred from a pending question or elapsed time.

## Requirement and evidence map

- Spec 8.7/12.1: all existing RMST value/first derivative/supported Hessian,
  eager/jit/vmap, boundary/invalid isolation and shape-restriction tests run
  through the renamed imports in the unchanged fresh-wheel A10 gate.
- Spec 10.2/10.3: private implementation stays private; every source importer
  updated. Ruff/mypy plus fresh installed-wheel smoke cover the import paths;
  explicit archive inspection verifies only the new helper path is shipped.
- Permanent appendix alignment: Pandoc/XeLaTeX compiles, PDF text extracts,
  Poppler renders all pages; final PNGs inspected individually. No clipped
  equations, broken glyphs or overflow. Bounding-box inspection puts all text
  within each 612x792pt page, lowest edge 766.404pt on all four pages.

## Actual validation

- `uv run ruff format --check .`: 135 files formatted;
  `uv run ruff check .`: all checks passed;
  `uv run mypy src`: no issues in 22 source files; `git diff --check`: clean.
- `rg -n '_numerics' src appendices docs scripts tests --glob '*.py' --glob '*.md' --glob '*.sh'`:
  no matches (exit 1). Historical PLAN/archive and read-only spec references
  are intentionally preserved. Existing test files have no direct old import.
- `pandoc appendices/weibull-rmst.md --standalone --number-sections --pdf-engine=xelatex -o appendices/weibull-rmst.pdf`:
  exit 0. Initial render five pages; remove obsolete forced page break and
  rebuild: final four pages, 51166 bytes, 612x792pt letter, no encryption.
  `pdfinfo`, `pdftotext -layout`, `pdftotext -bbox`, and
  `pdftoppm -scale-to 1300 -png` succeeded. Final pages inspected via view_image.
  Temporary renders /private/tmp/ph3d-final-{1,2,3,4}.png; no output in docs/.
- Wheel ZIP inspection: new _rmst.py present, old _numerics.py absent,
  new module content exactly equals source. Retained artifact:
  /var/folders/lz/l2_06bbx7fn4z318dtwnb3x40000gp/T/survjax-wheel.axaKwS/survjax-0.0.1.dev0-py3-none-any.whl;
  SHA256 7a18a4a72f795ddb321a1c3734c69892ce12f9ee351e16e89c81ebc17f6deff8.
- `scripts/check_wheel.sh --through A10`: **exit 0, all gates passed**.
  Production source stayed fixed throughout:
  - own x64 **1281 passed, 8 skipped in 128.59s**;
  - own float32 **1286 passed, 3 skipped in 171.25s**;
  - doctests **4 passed** each (1.22s /1.12s);
  - independent x64 **2332 passed, 28 skipped in 85.78s**;
  - independent float32 **2322 passed, 38 skipped in 73.49s**.
  Log /private/tmp/ph3d-wheel-gate.log. Environment unchanged: Python 3.12.9,
  JAX 0.11.2, NumPy 2.5.3, mpmath 1.4.1. No R or standard-docs build;
  neither fixture nor docs/ source was changed. Standalone PDF build above.
  PDF SHA256 20e1ce340e728b03c5e857f885a2caae19e69f31609424572531d554d11895b0.

## Remaining work

No uncertainty about this naming change. Prior PH.3c optional N-1/N-2 remain
recorded; this step makes no numerical change to address them. No generic
summary, AFT, sampling, module dumping ground, dependency or publishing work.

Next: exports/import/install/API-reference integration with its required
explicit docs/api.md exception, executable PH topic, and final PH evidence /
acceptance audit. The appendix/PDF alignment is completed here. Write final
results, commit, invoke independent-review PH.3d exactly and report its verdict;
continue only as explicitly authorized by the maintainer's “then on to the
next step” instruction and the applicable permission boundary.
