from __future__ import annotations

from pathlib import Path

from scripts.check_python_size import check_paths


def test_python_size_limits() -> None:
    errors = check_paths((Path("src"), Path("tests"), Path("scripts")))
    assert errors == []

