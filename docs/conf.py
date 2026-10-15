"""Sphinx configuration for the local survjax documentation (spec 10.5)."""

project = "survjax"
extensions = ["myst_nb"]
# The distributions guide writes formulas as $...$ and $$...$$.
myst_enable_extensions = ["dollarmath"]
# Pages link to sections, as in builtin-families.md#precision-in-both-tails.
myst_heading_anchors = 3
# The existing simulation guide describes unimplemented likelihood workflows.
exclude_patterns = ["_build", ".ipynb_checkpoints", "simulation-guide.md"]
html_theme = "alabaster"

# Execute source notebooks on every build; committed notebooks have no outputs.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 120
