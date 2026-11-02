# MNIST.18 — Runnable notebook companion

Independent review: round 1/3.

## Read first

1. PLAN.md MNIST.18: user requested a runnable notebook in addition to code/docs.
2. examples/survival_mnist/survival-mnist.ipynb: 11 cells, outputs cleared.
3. Example README launch command and docs page notebook download link.

The notebook runs the example directly in its kernel: data, intercept fitting,
training, prediction, evaluation and figures. CONFIG chooses smoke or full.
It reuses data/model functions and run._figures; editable model source remains
model.py (restart/run-all instructions). No separate trainer, API or dependency.
The untyped loss excerpt is explanatory and matches the example's computation.
Saved full-run results are explicitly distinguished from current notebook results.
Cohort sizes, keys, float32 convention, labels excluded from training, checkpoint
selection and metric definitions remain those of use-case03/spec 11.2.
It is a repository notebook, not a standalone file: launch/dependency and cache
instructions are included. It finds the repository from root or example cwd.
Figures go to a fresh temporary directory, never the committed results folder.
Original mnist2.md remains untouched/untracked.

## Verification

- nbformat schema validates; saved notebook has no outputs/execution counts.
- nbclient executed all cells, smoke from the notebook directory; no cell errors.
  Outputs include 5k/1k/2k sizes, epoch 1, live metrics, a generating-curves PNG,
  CNN source, three live SVGs and committed full metrics/headline SVG.
- A temporary copy with CONFIG=full executed all cells from repository root:
  50k/10k/10k, epoch 7. Live rounded metrics match committed full metrics:
  CNN -1.127/1.368/.689, intercept -1.238/1.188/.500, oracle -1.111/1.380/.698.
- Initial sandbox smoke execution succeeded but kernel shutdown hit a psutil
  PermissionError. Final smoke and full executions used normal kernel permissions
  and succeeded; only the existing IPKernelApp TCP warning remained.
- scripts/build_docs.sh directly: exit 0, both modes build succeeded. Notebook
  download href resolves to a byte-identical copy of the source in both builds.
- Ruff format/check on notebook: 1 already formatted; all checks passed.
  git diff --check clean. No new tests or broad gate: no library/runtime changes.

Executed copies are /tmp/survival-mnist-executed.ipynb and
/tmp/survival-mnist-full-executed.ipynb; no output or data committed.
Implementation files: 235-line notebook plus 13 README/docs lines, below 300.
Starting/previous step 4020dda; original worktree base ee72c83.
No known issue; independent review is the only remaining sub-step.
