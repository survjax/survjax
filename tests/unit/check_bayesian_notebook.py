"""Optional: the Bayesian tutorial notebook runs top to bottom (needs docs and examples groups)."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "examples/bayesian_rmst/bayesian-rmst.ipynb"
)


def test_notebook_runs_with_short_chains():
    """02: a fresh kernel runs every cell; 100 draws per chain keep the check short."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup = notebook.cells[1]
    assert setup.source.count("NUM_DRAWS = 1000") == 1
    setup.source = setup.source.replace("NUM_DRAWS = 1000", "NUM_DRAWS = 100")
    # Raises CellExecutionError, naming the cell, on the first failing cell.
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
