"""Sphinx configuration for the local Chronox documentation (spec 10.5)."""

project = "Chronox"
extensions = ["myst_nb"]
# The existing simulation guide describes unimplemented likelihood workflows.
exclude_patterns = ["_build", ".ipynb_checkpoints", "simulation-guide.md"]
html_theme = "alabaster"

# Execute source notebooks on every build; committed notebooks have no outputs.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 120
