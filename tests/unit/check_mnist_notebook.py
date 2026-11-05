"""Optional: the tutorial notebook runs top to bottom (needs MNIST and the docs group)."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "examples/survival_mnist/survival-mnist.ipynb"
)


def test_notebook_runs_in_smoke_config():
    """03: a fresh kernel runs every cell; smoke keeps the check short."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup = notebook.cells[1]
    assert setup.source.count('CONFIG = "full"') == 1
    setup.source = setup.source.replace('CONFIG = "full"', 'CONFIG = "smoke"')
    # Raises CellExecutionError, naming the cell, on the first failing cell.
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
