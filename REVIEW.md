# SIM.notebook-cleanup — Lean demo support

Read the newest PLAN section, run.py, and the notebook/docs changes. The
maintainer asked for an interactive notebook and removal of integration-test
infrastructure. The existing no-full-suite/no-reviewer waiver remains in force.
Start/previous final: 8349e7f; original worktree base: 8757c0f.

The cleanup replaces the raw diagnostic exporter and audit reducer with one
runner that writes the tables, histograms and figures used by the tutorial.
Six check modules and the stale dependency-probe document are removed; two
focused statistical checks remain. The generator/calibration functions remain
because MNIST and Bayesian examples import them. The fitter's numerical
operations and acceptance rules are unchanged. Failure reasons and rejected
fits remain in the study outputs. Existing committed full-study evidence is
retained as a historical artifact; new runs do not create audit manifests.

Support Python shrinks from 3,018 lines to 792 (including the two checks).
No dependency or public API changes. The notebook will run a fresh study and
show fresh plots; it never overwrites committed results. The maintainer's new
sentence about a future sample method is preserved.

Validation so far:
- Two focused checks pass (3.93 s): analytic exponential MLE/SE and rejection
  when an arm has no events. Initial test collection needed the example/test
  paths added; no tolerance change was made.
- The simplified full runner completed 12,000 fits. All 12 summary rows match
  saved statistical summaries and failure counts at the standard float64
  tolerances after aligning by scenario/model. Timing columns are excluded.
  Initial comparison failed only because rows were in a different order.
- Ruff import fixes applied to the new notebook and checks.

Remaining authorized delivery: execute the full notebook in a fresh kernel,
execute smoke via both docs builds, finish README/download links and record
final validation. No independent reviewer or full suites will run.

## Notebook delivery and final validation

Support cleanup is committed as b757e88. The 23-cell notebook follows the
accepted prose, preserves the maintainer's sample-method addition, computes
its own single-trial curves and study figures, and exposes CONFIG=full/smoke.
The README and docs link to it. Notebook formatting configuration matches the
other two use cases; checkpoint directories are ignored throughout the repo.

Both full and smoke notebook variants executed successfully in fresh Python 3
kernels via nbclient (300/180-second per-cell limits). Each produced three
figure outputs and an HTML results table with no error outputs. Source has
no execution counts or outputs. Executed review copies are in
/tmp/weibull-notebook-executed.ipynb and /tmp/weibull-notebook-smoke.ipynb.
A final prose-only fix removed the docs self-download paragraph from the
notebook; no code changed after execution.

scripts/build_docs.sh exited 0: both dtype builds passed with the simplified
smoke runner and notebook download. After the prose-only fix, both downloadable
notebook assets were refreshed and verified byte-for-byte against source.
Ruff check, Ruff format check (6 files) and git diff --check pass. No full test
suite or independent reviewer ran. No browser visual inspection is claimed.
All requested work is complete; no substeps remain.
