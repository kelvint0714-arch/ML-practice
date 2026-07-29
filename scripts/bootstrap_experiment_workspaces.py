"""Create and index every unstarted Day/Unit experiment workspace.

This is a repository-maintenance command, not a progress-completion command.
It prepares full, output-cleared notebook copies for Day 02–35 and active
learning Unit 01–09. Existing learner notes and executed notebooks are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from start_day import (
    REPO_ROOT,
    create_workspace as create_day_workspace,
    day_title,
    find_day_directory,
    render_experiment_readme as render_day_readme,
    render_notes as render_day_notes,
)
from start_unit import (
    create_workspace as create_unit_workspace,
    find_unit_directory,
    render_experiment_readme as render_unit_readme,
    render_notes as render_unit_notes,
    unit_title,
)


EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
INDEX_PATH = EXPERIMENTS_ROOT / "INDEX.md"


def is_unexecuted(notebook: dict) -> bool:
    """Return whether every code cell is still an empty-output starter cell."""

    code_cells = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    return bool(code_cells) and all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in code_cells
    )


def normalize_unstarted_notebook(path: Path, dry_run: bool) -> None:
    """Correct metadata on a pristine copy without touching learner results."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    metadata = notebook.setdefault("metadata", {})
    if metadata.get("workspace_status") in {"in_progress", "completed"}:
        print(f"keep    {path.relative_to(REPO_ROOT)} (learner status set)")
        return
    if not is_unexecuted(notebook):
        print(f"keep    {path.relative_to(REPO_ROOT)} (contains learner output)")
        return

    expected = {
        "artifact_role": "learner_workspace",
        "learner_evidence": False,
        "workspace_status": "not_started",
    }
    if all(metadata.get(key) == value for key, value in expected.items()):
        print(f"keep    {path.relative_to(REPO_ROOT)} (starter metadata current)")
        return

    print(f"update  {path.relative_to(REPO_ROOT)} (mark not_started)")
    if dry_run:
        return
    metadata.update(expected)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def is_pristine_workspace(notebook_path: Path) -> bool:
    """Return whether generated text can be refreshed without resetting work."""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = notebook.get("metadata", {})
    return (
        is_unexecuted(notebook)
        and metadata.get("workspace_status") == "not_started"
        and metadata.get("learner_evidence") is False
    )


def refresh_placeholder_notes(path: Path, expected: str, dry_run: bool) -> None:
    """Refresh only untouched generated notes, never learner-authored text."""

    current = path.read_text(encoding="utf-8")
    previous_template = expected.replace("\n> 状态：待本人填写\n", "")
    if current not in {expected, previous_template}:
        print(f"keep    {path.relative_to(REPO_ROOT)} (contains learner notes)")
        return
    if current == expected:
        print(f"keep    {path.relative_to(REPO_ROOT)} (starter status current)")
        return
    print(f"update  {path.relative_to(REPO_ROOT)} (mark pending)")
    if not dry_run:
        path.write_text(expected, encoding="utf-8")


def sync_day(day_number: int, dry_run: bool) -> tuple[Path, str]:
    """Create one Day workspace and refresh only its generated README."""

    day_directory = find_day_directory(day_number)
    destination = EXPERIMENTS_ROOT / day_directory.name
    create_day_workspace(day_number, dry_run=dry_run)

    readme = destination / "README.md"
    readme_content = render_day_readme(day_title(day_directory), day_directory)
    if not dry_run:
        notebook_path = destination / f"{day_directory.name}.ipynb"
        normalize_unstarted_notebook(notebook_path, dry_run=False)
        if is_pristine_workspace(notebook_path):
            print(f"update  {readme.relative_to(REPO_ROOT)} (generated status)")
            readme.write_text(readme_content, encoding="utf-8")
            refresh_placeholder_notes(
                destination / "notes.md",
                render_day_notes(day_title(day_directory)),
                dry_run=False,
            )
        else:
            print(f"keep    {readme.relative_to(REPO_ROOT)} (learner workspace)")
    else:
        print(f"update  {readme.relative_to(REPO_ROOT)} (if still pristine)")
    return destination, day_title(day_directory)


def sync_unit(unit_number: int, dry_run: bool) -> tuple[Path, str]:
    """Create one Unit workspace and refresh only its generated README."""

    unit_directory = find_unit_directory(unit_number)
    destination = EXPERIMENTS_ROOT / "active_learning" / unit_directory.name
    create_unit_workspace(unit_number, dry_run=dry_run)

    readme = destination / "README.md"
    readme_content = render_unit_readme(unit_title(unit_directory), unit_directory)
    if not dry_run:
        notebook_path = destination / f"{unit_directory.name}.ipynb"
        normalize_unstarted_notebook(notebook_path, dry_run=False)
        if is_pristine_workspace(notebook_path):
            print(f"update  {readme.relative_to(REPO_ROOT)} (generated status)")
            readme.write_text(readme_content, encoding="utf-8")
            refresh_placeholder_notes(
                destination / "notes.md",
                render_unit_notes(unit_title(unit_directory)),
                dry_run=False,
            )
        else:
            print(f"keep    {readme.relative_to(REPO_ROOT)} (learner workspace)")
    else:
        print(f"update  {readme.relative_to(REPO_ROOT)} (if still pristine)")
    return destination, unit_title(unit_directory)


def render_index(
    day_entries: list[tuple[int, Path, str]],
    unit_entries: list[tuple[int, Path, str]],
) -> str:
    """Return the full learner-facing experiment workspace index."""

    lines = [
        "# 实验工作区完整索引",
        "",
        "> 这里的“已准备”只表示完整起始代码已经放入 `experiments/`；",
        "> 除 Day 01 参考基线外，其余工作区都尚未代表本人完成或复现。",
        "",
        "## 已运行参考基线",
        "",
        "| 编号 | 内容 | 状态 |",
        "|---|---|---|",
        "| Day 01 | [ESOL 传统机器学习参考基线]"
        "(esol/day01_baseline/README.md) | 已运行参考制品 |",
        "",
        "## Day 02–28：核心算法路线",
        "",
        "| 编号 | 主题 | 工作区 | 初始状态 |",
        "|---|---|---|---|",
    ]
    for number, destination, title in day_entries:
        if number > 28:
            continue
        relative_destination = destination.relative_to(EXPERIMENTS_ROOT)
        notebook = f"{destination.name}.ipynb"
        lines.append(
            f"| Day {number:02d} | {title} | "
            f"[README]({relative_destination}/README.md) · "
            f"[Notebook]({relative_destination}/{notebook}) | 待本人运行 |"
        )

    lines.extend(
        [
            "",
            "## Day 29–35：可选 GNN 路线",
            "",
            "| 编号 | 主题 | 工作区 | 初始状态 |",
            "|---|---|---|---|",
        ]
    )
    for number, destination, title in day_entries:
        if number < 29:
            continue
        relative_destination = destination.relative_to(EXPERIMENTS_ROOT)
        notebook = f"{destination.name}.ipynb"
        lines.append(
            f"| Day {number:02d} | {title} | "
            f"[README]({relative_destination}/README.md) · "
            f"[Notebook]({relative_destination}/{notebook}) | 待本人运行 |"
        )

    lines.extend(
        [
            "",
            "## Unit 01–09：主动学习专题",
            "",
            "| 编号 | 主题 | 工作区 | 初始状态 |",
            "|---|---|---|---|",
        ]
    )
    for number, destination, title in unit_entries:
        relative_destination = destination.relative_to(EXPERIMENTS_ROOT)
        notebook = f"{destination.name}.ipynb"
        lines.append(
            f"| Unit {number:02d} | {title} | "
            f"[README]({relative_destination}/README.md) · "
            f"[Notebook]({relative_destination}/{notebook}) | 待本人运行 |"
        )

    lines.extend(
        [
            "",
            "## 正确使用方式",
            "",
            "1. 按 [`curriculum/PROGRESS.md`](../curriculum/PROGRESS.md) 的顺序学习；",
            "2. 打开对应工作区 README，再从空内核运行 Notebook；",
            "3. 把解释写入 `notes.md`，把本人生成的文件放入 `results/`；",
            "4. 通过任务卡自测后，才更新进度清单和工作区状态。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare all Day 02–35 and active-learning Unit 01–09 starter "
            "workspaces without overwriting learner work."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the workspace operations without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    """Build missing workspaces, normalize pristine copies, and write the index."""

    args = parse_args()
    day_entries = [
        (number, *sync_day(number, dry_run=args.dry_run))
        for number in range(2, 36)
    ]
    unit_entries = [
        (number, *sync_unit(number, dry_run=args.dry_run))
        for number in range(1, 10)
    ]
    index_content = render_index(day_entries, unit_entries)
    print(f"update  {INDEX_PATH.relative_to(REPO_ROOT)}")
    if not args.dry_run:
        INDEX_PATH.write_text(index_content, encoding="utf-8")
        print(
            "Prepared 34 Day workspaces and 9 active-learning Unit workspaces. "
            "All pristine copies remain marked not_started."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
