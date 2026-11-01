Independent review: round 1/3

# MNIST.11 — Committed full-run artifacts

Read PLAN.md MNIST.11, README.md's full-run section, results/metadata.json,
metrics.csv, history.csv/progress.csv, then the three rendered SVGs. Review
this as generated experimental evidence, not new numerical implementation.

Scope and invariants:
- Implementation source is 222341684e877153122e3a6674250987cfdab7c9, unchanged
  numerically in this step. Changes are workflow records, the example README,
  generated results and export whitespace fixes (see below). No API,
  dependency, protected path, spec or training change.
- One fixed ten-epoch full run, with 50k training, 10k validation and 10k test.
  Prescribed keys/settings and validation-only endpoint selection. No tuning,
  failed-run selection or hand-edited numerical files. Earlier diagnostic was
  approved by the maintainer; this run adds the approved two-view C timing.
- Full CSV/SVG artifacts naturally exceed the 300-line implementation guideline.
  Generated files are reviewed through consistency checks and visual inspection;
  human-authored changes remain small. No code is hidden or minified.
- Exactly metrics.csv, history.csv, progress.csv, metadata.json and fig1/2/3.svg.
  No IDX/cache, label arrays, per-image predictions or model checkpoints committed.
  Figure 1 contains the ten MNIST illustrations required by the design, attributed
  under CC BY-SA 3.0 in the README. All survival outcomes are synthetic.
- D-031/D-032 decisions are implemented and already approved in conversation.
  Their formal statuses remain maintainer-owned; no entries change here.

Generation command (repository root, warm hash-verified cache):

```bash
SURVJAX_EXAMPLES_DATA=/private/tmp/survjax-mnist-verification MPLCONFIGDIR=/private/tmp/survjax-mpl-release-full uv run --group examples python -m examples.survival_mnist.run --config full --out examples/survival_mnist/results
```

Exit 0. Preparation 4.97355 s, training 49.44318 s, on the CPU environment
recorded in metadata. Verification suites ran concurrently; these timings are
illustrative, not an isolated performance comparison. First C invocation may
reuse earlier compilation; warm timing is a median of three synchronized calls.

Acceptance evidence:
- Finite gradients at initialization/after first update: full train completed its
  explicit _require_finite checkpoints. Existing check_mnist_model finite-gradient
  and check_mnist_training checkpoint failure tests remain the regression evidence.
- CNN beats intercept on test mean LL: -1.12626493 > -1.23686779; on C:
  .68943512 > .5. Oracle mean LL -1.11036825 and C .69751966, giving gaps
  .01589668 and .00808454. No claim that the oracle gap must close.
- Learned k 1.36612070 is 1.00575% below 1.38, within the 5% criterion.
- Validation selects earliest best endpoint epoch 7. Test rows did not choose it.
- Metadata/metrics contain actual 50k/10k/10k sizes and test n=10k for all models.
  Realized censoring: train .50036, validation .5079, test .4995.
- History has 10 rows; trace has 3901 data rows (3900 batch losses and 101
  validation samples). Every step is present, last training cell blank, validation
  at 0 and multiples of 39, and every endpoint agrees exactly with history.csv.
  Independent NumPy means of the logged minibatches agree with epoch summaries
  under the unchanged implementation float32 tolerances.
- Both C timing fields agree exactly between CSV/JSON for each model. Seconds:
  CNN first .342211 / warm .229402; intercept .188547 / .206041;
  oracle .191947 / .220775. Warm is not assumed faster under timing noise.
- Source hashes match the loader's verified SHA-256 table; root seed, settings,
  config, selected epoch and repeat count agree across records.
- The full diagnostic's numerical metric fields and history/progress fields are
  reproduced exactly; CSV newlines are normalized to LF. Timing/provenance differs. No experiment change.

Independent artifact check:
`uv run --group examples python /private/tmp/verify_mnist11.py` exited 0:
"PASS: full acceptance, cohort/censoring/hash checks, CSV/metadata/trace
consistency, SVG structure" and "PASS: approved experiment numerical
metrics/history/progress reproduced exactly". Initial harness import failed
because /tmp was sys.path[0]; adding the repository root fixed the harness.
No numerical field was modified. The key checks can be repeated without MNIST:

```python
import csv
import json
import math
from pathlib import Path

root = Path("examples/survival_mnist/results")


def read(name):
    with (root / name).open() as stream:
        return list(csv.DictReader(stream))


rows = read("metrics.csv")
history = read("history.csv")
progress = read("progress.csv")
m = json.loads((root / "metadata.json").read_text())
r = {x["model"]: x for x in rows}
assert float(r["cnn"]["mean_log_likelihood"]) > float(
    r["intercept"]["mean_log_likelihood"]
)
assert float(r["cnn"]["concordance"]) > float(r["intercept"]["concordance"]) == 0.5
assert abs(float(r["cnn"]["shape"]) / 1.38 - 1) <= 0.05
assert all(int(x["n"]) == 10000 for x in rows)
assert m["sizes"] == {"train": 50000, "validation": 10000, "test": 10000}
assert len(history) == 10 and len(progress) == 3901
assert (
    m["selected_epoch"]
    == min(range(10), key=lambda i: float(history[i]["validation_nll"])) + 1
)
assert [int(x["step"]) for x in progress] == list(range(3901))
v = {
    int(x["step"]): float(x["validation_nll"]) for x in progress if x["validation_nll"]
}
assert list(v) == list(range(0, 3901, 39))
for epoch, row in enumerate(history, 1):
    assert float(row["validation_nll"]) == v[390 * epoch]
for row in rows:
    for col, key in (
        ("concordance_seconds", "concordance"),
        ("concordance_warm_seconds", "concordance_warm"),
    ):
        assert math.isfinite(float(row[col])) and float(row[col]) >= 0
        assert float(row[col]) == m["timings_seconds"][key][row["model"]]
```

Visual QA: rendered each committed SVG with installed rsvg-convert into
/private/tmp/survjax-mnist-artifact11/fig1/2/3.png and inspected with view_image.
All ten image/curve pairs are legible, true curves dashed and closely aligned;
boxplots show true markers by digit; convergence trace shows raw batch noise,
full validation plateau/deterioration and epoch 7 selection. No clipping or
layout changes needed. SVG XML parses and Figure 1 has exactly ten image nodes.
PNG previews remain outside Git.

Checks: optional nine-file suites, x64 91 passed/1 skipped (46.73 s), float32
69 passed/23 skipped (37.30 s). Ruff format: 138 files already formatted;
Ruff lint passed; mypy no issues in 21 source files (fresh-wheel gate).
Gate completion appended below. No new tests for generated data; existing
writer/math/selection tests remain, plus explicit artifact verification above.
No docs build or R generation. The demo's executable-page acceptance remains
for the docs step; do not claim the entire use case complete here.
Previous step final: 222341684e877153122e3a6674250987cfdab7c9.
Original worktree base: ee72c837549d14cb26e9983bf6d63ac54563f189.
Remaining sub-step: executable docs with live smoke and committed results.


CSV newline adjustment: git diff --no-index --check identified CRLF as trailing
whitespace in the generated CSVs. Explicit lineterminator="\n" in all three
writer constructors makes future output directly committable. Normalize this
run's CSV newlines only; the artifact checker still compares every numerical
field/history/trace to the approved diagnostic. Extend the existing writer
check to require LF. The first fresh-wheel gate passed before this formatting
fix; final targeted writer checks/static checks cover it. No core code changed.


Cumulative gate: scripts/check_wheel.sh --through A10 exited 0, "check: all
gates passed through A10". Both dtype acceptance suites completed with the
fresh wheel. Own suites: x64 1119 passed/7 skipped (81.73 s), float32 1125
passed/1 skipped (75.56 s); doctests 4 passed per mode. Final acceptance output:
float32 2324 passed/36 skipped (59.97 s). Ruff lint passed and mypy reported no
issues in 21 source files. This gate preceded only CSV/SVG export whitespace normalization
and final workflow/README wording. Final writer checks cover that fix:
float32 4 passed (1.35 s), x64 4 passed (.48 s). Final Ruff format/lint and
staged whitespace checks are clean. No broad repeat justified for newline-only
example output and prose; no core code changed.


Staged whitespace checking also found Matplotlib's trailing path-data spaces
in SVGs. Strip terminal spaces/tabs only, retaining line boundaries and every
path token. Apply the same formatting in the figure writer and pin the exported
format in the existing SVG test. Compare freshly rasterized normalized SVGs
against the pre-normalization PNGs pixel-for-pixel before committing. This is
output formatting only; no figure redesign or numerical field edits.
All three normalized SVGs were re-rendered and their PNG arrays compared with
the pre-normalization PNGs: pixel-identical. Artifact verification still passes.
Final focused export tests: float32 9 passed (10.36 s), x64 9 passed (9.65 s).
Initial staged checking identified generated path whitespace; the final staged
check is clean after normalization. No full training or numerical test repeat
is needed for whitespace-only export changes.
