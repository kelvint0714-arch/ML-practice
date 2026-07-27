"""Execute the Day 2–28 tutorial notebooks from top to bottom.

The safe default is a read-only run: every Notebook executes in memory and
the source file is left unchanged. Pass ``--in-place`` explicitly when the
maintainer intentionally wants to refresh the shipped tutorial outputs.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "curriculum" / "core"
DAY_PATTERN = re.compile(r"day(?P<number>\d{2})_.+")


def selected_notebooks(first_day: int, last_day: int) -> list[Path]:
    """Return canonical tutorial notebooks inside the selected Day range."""

    notebooks: list[tuple[int, Path]] = []
    for day_directory in CORE_ROOT.glob("day??_*"):
        match = DAY_PATTERN.fullmatch(day_directory.name)
        if match is None:
            continue
        number = int(match.group("number"))
        if first_day <= number <= last_day:
            notebooks.append((number, day_directory / "tutorial.ipynb"))
    notebooks.sort()
    return [path for _, path in notebooks]


def clear_runtime_state(notebook: nbformat.NotebookNode) -> None:
    """Remove stale outputs and machine-dependent execution timing metadata."""

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.execution_count = None
        cell.outputs = []
        cell.metadata.pop("execution", None)


def execute_notebook(path: Path, timeout: int, in_place: bool) -> None:
    """Execute one notebook; optionally replace the source atomically."""

    notebook = nbformat.read(path, as_version=4)
    clear_runtime_state(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
        record_timing=False,
    )
    client.execute()
    if not in_place:
        return

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".ipynb",
        prefix=f".{path.stem}.",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        nbformat.write(notebook, temporary)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute core tutorial notebooks. The default checks them without "
            "editing source files; --in-place explicitly refreshes saved outputs."
        )
    )
    parser.add_argument("--first-day", type=int, default=2)
    parser.add_argument("--last-day", type=int, default=28)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace source Notebooks with fresh outputs.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute selected notebooks and report every failure."""

    args = parse_args()
    if not 2 <= args.first_day <= args.last_day <= 28:
        raise SystemExit("Choose a range inside Day 2–28.")

    notebooks = selected_notebooks(args.first_day, args.last_day)
    expected_count = args.last_day - args.first_day + 1
    if len(notebooks) != expected_count:
        missing = [str(path.relative_to(REPO_ROOT)) for path in notebooks if not path.exists()]
        print(
            f"Expected {expected_count} tutorial notebooks, found {len(notebooks)}.",
            file=sys.stderr,
        )
        if missing:
            print(f"Missing: {missing}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for index, path in enumerate(notebooks, start=1):
        relative = path.relative_to(REPO_ROOT)
        if not path.is_file():
            failures.append(f"{relative}: file is missing")
            print(f"[{index}/{len(notebooks)}] missing {relative}")
            continue
        mode = "refresh" if args.in_place else "check"
        print(f"[{index}/{len(notebooks)}] {mode} {relative}")
        try:
            execute_notebook(
                path,
                timeout=args.timeout,
                in_place=args.in_place,
            )
        except (CellExecutionError, TimeoutError, OSError, ValueError) as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")

    if failures:
        print("\nNotebook execution failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    if args.in_place:
        print(f"Refreshed {len(notebooks)} tutorial notebooks successfully.")
    else:
        print(
            f"Checked {len(notebooks)} tutorial notebooks successfully; "
            "source files were not changed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
