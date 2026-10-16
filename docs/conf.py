"""Sphinx configuration for the local survjax documentation (spec 10.5)."""

project = "survjax"
# autodoc and napoleon (Google-style sections) ship with Sphinx: no new dependency.
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
# The distributions guide writes formulas as $...$ and $$...$$.
myst_enable_extensions = ["dollarmath"]
# Pages link to sections, as in builtin-families.md#precision-in-both-tails.
myst_heading_anchors = 3
# The existing simulation guide describes unimplemented likelihood workflows.
exclude_patterns = ["_build", ".ipynb_checkpoints", "simulation-guide.md"]
html_theme = "alabaster"

# API reference: ArrayLike expands to a long union in signatures, and the
# docstrings' Args sections already describe types, so hide hints there.
autodoc_typehints = "none"
autodoc_member_order = "bysource"
# Show defaults as written (static's MISSING), not as object reprs.
autodoc_preserve_defaults = True
# SurvivalData documents its constructor in __init__.
autoclass_content = "both"
# Attribute types come from annotations; JAX's docs are not linked (no network).
nitpick_ignore = [("py:class", "jax.Array")]

# Execute source notebooks on every build; committed notebooks have no outputs.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 120
