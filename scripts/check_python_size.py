#!/usr/bin/env python3
"""Enforce repository Python file and function size limits."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable


MAX_FILE_LINES = 800
MAX_FUNCTION_LINES = 200
DEFAULT_ROOTS = (Path("src"), Path("tests"), Path("scripts"))


def python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.py"))


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    line_count = len(source.splitlines())
    if line_count > MAX_FILE_LINES:
        errors.append(f"{path}: {line_count} lines exceeds {MAX_FILE_LINES}")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}:{exc.lineno}: cannot inspect invalid Python: {exc.msg}")
        return errors

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = node.end_lineno or node.lineno
            function_lines = end_line - node.lineno + 1
            if function_lines > MAX_FUNCTION_LINES:
                errors.append(
                    f"{path}:{node.lineno}: {node.name} has {function_lines} lines; "
                    f"limit is {MAX_FUNCTION_LINES}"
                )
    return errors


def check_paths(roots: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in python_files(roots):
        errors.extend(check_file(path))
    return errors


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    roots = tuple(Path(argument) for argument in arguments) or DEFAULT_ROOTS
    errors = check_paths(roots)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    checked = sum(1 for _ in python_files(roots))
    print(
        f"Python size check passed: {checked} files, "
        f"file <= {MAX_FILE_LINES} lines, function <= {MAX_FUNCTION_LINES} lines"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

