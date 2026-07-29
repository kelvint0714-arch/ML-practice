"""Set an honest status for one learner experiment workspace.

Examples:

    python scripts/set_workspace_status.py day02 in_progress
    python scripts/set_workspace_status.py day02 completed
    python scripts/set_workspace_status.py unit01 in_progress

The command updates only the experiment copy. It never edits curriculum
source notebooks or marks ``curriculum/PROGRESS.md`` automatically.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from start_day import (
    EXPERIMENTS_ROOT,
    REPO_ROOT,
    day_title,
    find_day_directory,
    render_notes as render_day_notes,
)
from start_unit import (
    find_unit_directory,
    render_notes as render_unit_notes,
    unit_title,
)


ITEM_PATTERN = re.compile(r"(?P<kind>day|unit)(?P<number>\d{1,2})", re.IGNORECASE)
STATUS_LABELS = {
    "not_started": ("待本人运行", "待本人填写"),
    "in_progress": ("进行中", "正在记录"),
    "completed": ("已完成（需通过任务卡自测）", "本人已填写"),
}


def parse_item(item: str) -> tuple[str, int]:
    """Parse an item such as ``day02`` or ``unit01``."""

    match = ITEM_PATTERN.fullmatch(item.strip())
    if match is None:
        raise SystemExit("Item must look like day02 or unit01.")
    return match.group("kind").lower(), int(match.group("number"))


def workspace_paths(kind: str, number: int) -> tuple[Path, Path, str]:
    """Return workspace, notebook, and untouched notes template."""

    if kind == "day":
        if not 2 <= number <= 35:
            raise SystemExit("Day workspace status is available for Day 02–35.")
        source_directory = find_day_directory(number)
        workspace = EXPERIMENTS_ROOT / source_directory.name
        starter_notes = render_day_notes(day_title(source_directory))
    else:
        if not 1 <= number <= 9:
            raise SystemExit("Unit workspace status is available for Unit 01–09.")
        source_directory = find_unit_directory(number)
        workspace = EXPERIMENTS_ROOT / "active_learning" / source_directory.name
        starter_notes = render_unit_notes(unit_title(source_directory))

    notebook = workspace / f"{source_directory.name}.ipynb"
    return workspace, notebook, starter_notes


def code_cells(notebook: dict) -> list[dict]:
    """Return all code cells from a Notebook document."""

    return [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def has_saved_execution(notebook: dict) -> bool:
    """Return whether any code cell contains an execution count or output."""

    return any(
        cell.get("execution_count") is not None or cell.get("outputs", []) != []
        for cell in code_cells(notebook)
    )


def validate_completed(
    notebook: dict,
    notes_path: Path,
    starter_notes: str,
) -> None:
    """Reject a completed claim when basic execution evidence is absent."""

    cells = code_cells(notebook)
    if not cells or any(cell.get("execution_count") is None for cell in cells):
        raise SystemExit(
            "Cannot mark completed: run every code cell from a clean kernel first."
        )
    for cell in cells:
        if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
            raise SystemExit(
                "Cannot mark completed: the Notebook still contains an error output."
            )

    notes = notes_path.read_text(encoding="utf-8")
    status_line = re.compile(r"^> 状态：.*$", re.MULTILINE)
    normalized_notes = status_line.sub("", notes).strip()
    normalized_starter = status_line.sub("", starter_notes).strip()
    if normalized_notes == normalized_starter:
        raise SystemExit(
            "Cannot mark completed: explain the run in notes.md before completion."
        )


def update_status_line(path: Path, replacement: str) -> None:
    """Replace the generated status line in a README or notes file."""

    text = path.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"^> 状态：.*$",
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise SystemExit(f"Cannot find generated status line in {path}.")
    path.write_text(updated, encoding="utf-8")


def set_status(item: str, status: str) -> Path:
    """Validate and update one experiment workspace status."""

    kind, number = parse_item(item)
    workspace, notebook_path, starter_notes = workspace_paths(kind, number)
    readme_path = workspace / "README.md"
    notes_path = workspace / "notes.md"
    for required in (readme_path, notes_path, notebook_path):
        if not required.is_file():
            raise SystemExit(
                f"Workspace file is missing: {required.relative_to(REPO_ROOT)}"
            )

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if status == "not_started" and has_saved_execution(notebook):
        raise SystemExit(
            "Cannot mark not_started while the Notebook contains saved execution."
        )
    if status == "completed":
        validate_completed(notebook, notes_path, starter_notes)

    metadata = notebook.setdefault("metadata", {})
    metadata["artifact_role"] = "learner_workspace"
    metadata["workspace_status"] = status
    metadata["learner_evidence"] = status == "completed"
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    readme_label, notes_label = STATUS_LABELS[status]
    update_status_line(readme_path, f"> 状态：**{readme_label}**")
    update_status_line(notes_path, f"> 状态：{notes_label}")
    print(
        f"Updated {workspace.relative_to(REPO_ROOT)} to {status}. "
        "curriculum/PROGRESS.md was not changed."
    )
    return workspace


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Set one Day/Unit experiment workspace status honestly."
    )
    parser.add_argument("item", help="Workspace identifier, for example day02.")
    parser.add_argument(
        "status",
        choices=tuple(STATUS_LABELS),
        help="not_started, in_progress, or completed.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""

    args = parse_args()
    set_status(args.item, args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
