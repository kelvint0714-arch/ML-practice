"""Execute active-learning Unit 01–09 tutorial notebooks top to bottom.

The safe default executes in memory and leaves source files unchanged.
Use ``--in-place`` only when a maintainer intentionally refreshes the
instructor-supplied outputs.
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
ACTIVE_ROOT = REPO_ROOT / "curriculum" / "active_learning"
UNIT_PATTERN = re.compile(r"unit(?P<number>\d{2})_.+")


def selected_notebooks(first_unit: int, last_unit: int) -> list[Path]:
    """Return canonical Unit notebooks in numeric order."""

    notebooks: list[tuple[int, Path]] = []
    for unit_directory in ACTIVE_ROOT.glob("unit??_*"):
        match = UNIT_PATTERN.fullmatch(unit_directory.name)
        if match is None:
            continue
        number = int(match.group("number"))
        if first_unit <= number <= last_unit:
            notebooks.append((number, unit_directory / "tutorial.ipynb"))
    notebooks.sort()
    return [path for _, path in notebooks]


def clear_runtime_state(notebook: nbformat.NotebookNode) -> None:
    """Remove stale outputs and machine-dependent timing metadata."""

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.execution_count = None
        cell.outputs = []
        cell.metadata.pop("execution", None)


def execute_notebook(
    path: Path,
    timeout: int,
    kernel_name: str,
    in_place: bool,
) -> None:
    """Execute one Unit notebook and optionally replace it atomically."""

    source_permissions = path.stat().st_mode & 0o777
    notebook = nbformat.read(path, as_version=4)
    clear_runtime_state(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
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
    temporary_path.chmod(source_permissions)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute active-learning Unit tutorials. The default validates "
            "without editing; --in-place refreshes saved outputs."
        )
    )
    parser.add_argument("--first-unit", type=int, default=1)
    parser.add_argument("--last-unit", type=int, default=9)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute selected notebooks and report all failures."""

    args = parse_args()
    if not 1 <= args.first_unit <= args.last_unit <= 9:
        raise SystemExit("Choose a range inside Unit 01–09.")

    notebooks = selected_notebooks(args.first_unit, args.last_unit)
    expected_count = args.last_unit - args.first_unit + 1
    if len(notebooks) != expected_count:
        print(
            f"Expected {expected_count} Unit notebooks, found {len(notebooks)}.",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    for index, path in enumerate(notebooks, start=1):
        relative = path.relative_to(REPO_ROOT)
        mode = "refresh" if args.in_place else "check"
        print(f"[{index}/{len(notebooks)}] {mode} {relative}")
        if not path.is_file():
            failures.append(f"{relative}: file is missing")
            continue
        try:
            execute_notebook(
                path,
                timeout=args.timeout,
                kernel_name=args.kernel_name,
                in_place=args.in_place,
            )
        except (CellExecutionError, TimeoutError, OSError, ValueError) as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")

    if failures:
        print("\nUnit notebook execution failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    action = "Refreshed" if args.in_place else "Checked"
    suffix = "" if args.in_place else "; source files were not changed"
    print(f"{action} {len(notebooks)} Unit notebooks successfully{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
