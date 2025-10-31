from pathlib import Path

import pytest

from ..io_tools import trim_path_overlap


@pytest.mark.parametrize(
    "path,name,trim_name,ensure_part,expected",
    [
        # No overlap
        ("a/b/c", "x/y", True, True, (Path("a/b/c"), Path("x/y"))),
        # Full overlap, trim_name=True
        ("a/b", "a/b", True, False, (Path("a/b"), Path())),
        ("a/b", "a/b", True, True, (Path("a"), Path("b"))),
        # Full overlap, trim_name=False
        ("a/b", "a/b", False, False, (Path(), Path("a/b"))),
        ("a/b", "a/b", False, True, (Path("a"), Path("b"))),
        # Partial overlap
        ("a/b/c", "c/d", True, True, (Path("a/b/c"), Path("d"))),
        ("a/b/c", "c/d", False, True, (Path("a/b"), Path("c/d"))),
        # Empty name
        ("a/b", "", True, False, (Path("a/b"), Path())),
        ("a/b", "", True, True, (Path("a"), Path("b"))),
        # Empty path
        ("", "x/y", True, False, (Path(), Path("x/y"))),
        ("", "x/y", True, True, (Path("x"), Path("y"))),
        # Both singletons, ensure_part=True
        ("a", "b", True, True, (Path("a"), Path("b"))),
    ],
)
def test_trim_path_overlap(path, name, trim_name, ensure_part, expected):
    result = trim_path_overlap(
        Path(path), Path(name), trim_name=trim_name, ensure_part=ensure_part
    )
    assert result == expected
