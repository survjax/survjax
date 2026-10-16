"""The API reference documents every public export (spec 10.2, 10.5)."""

from pathlib import Path

import survjax

API_PAGE = Path(__file__).parents[2] / "docs" / "api.md"


def test_api_reference_lists_every_export():
    """Spec 10.2: the public API is __all__; each name needs an autodoc entry."""
    page = API_PAGE.read_text()
    missing = [name for name in survjax.__all__ if f":: survjax.{name}\n" not in page]
    assert missing == []
