"""Create a learner-owned starter workspace for one curriculum day.

The curriculum keeps instructor-provided tutorial notebooks under
``curriculum/core`` and ``curriculum/optional_gnn``. This script copies
exactly one tutorial into ``experiments`` so that running or editing it does
not alter the source lesson. A newly copied notebook is explicitly marked
``not_started`` and is not learner evidence until the learner runs, checks,
and explains it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "curriculum" / "core"
OPTIONAL_GNN_ROOT = REPO_ROOT / "curriculum" / "optional_gnn"
DAY_ROOTS = (CORE_ROOT, OPTIONAL_GNN_ROOT)
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DAY_DIRECTORY_PATTERN = re.compile(r"day(?P<number>\d{2})_(?P<slug>.+)")


def readable_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_day_directory(day_number: int) -> Path:
    """Return the unique curriculum Day directory for ``day_number``."""

    matches = sorted(
        match
        for day_root in DAY_ROOTS
        for match in day_root.glob(f"day{day_number:02d}_*")
    )
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one curriculum directory for Day {day_number}, "
            f"found {len(matches)}."
        )
    return matches[0]


def day_title(day_directory: Path) -> str:
    """Read the first Markdown heading from a Day task card."""

    readme = day_directory / "README.md"
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return day_directory.name


def render_experiment_readme(title: str, day_directory: Path) -> str:
    """Return a learner-facing experiment README."""

    curriculum_path = day_directory.relative_to(REPO_ROOT)
    return f"""# {title}：实验工作区

> 状态：**待本人运行**

本目录预先提供完整起始代码，方便按学习目录直接开始。Notebook 的教学
预存输出已经清空；目录存在只表示“代码已准备”，不表示实验已经完成，
也不表示其中结果已经由本人复现。

## 课程来源

- 任务卡：[`{curriculum_path}/README.md`](../../{curriculum_path}/README.md)
- 教学 Notebook：[`{curriculum_path}/tutorial.ipynb`](../../{curriculum_path}/tutorial.ipynb)

## 完成规则

1. 从空内核运行本目录 Notebook；
2. 不跳过断言和检查；
3. 在 `notes.md` 用自己的话解释输入、动作、输出；
4. 将当天要求的 CSV、JSON 或图片放入 `results/`；
5. 写清结果能说明什么、不能说明什么；
6. 完成自测后再更新 `curriculum/PROGRESS.md`。

完成前不要改写上面的状态。教学示例不是粘合剂实验结果，也不能替代
本人运行记录。
"""


def render_notes(title: str) -> str:
    """Return an empty but structured learning record."""

    return f"""# {title}：学习记录

> 状态：待本人填写

## 今天完成了什么

-

## 输入、动作、输出

- 输入：
- 动作：
- 输出：

## 我能独立解释的算法概念

-

## 我今天新认识的 Python 语法

-

## 结果怎样解释

-

## 当前结果不能说明什么

-

## 遇到的报错及解决方式

- 完整报错：
- 原因：
- 怎样修复：

## 我还不能独立回答的问题

-
"""


def write_new_text(path: Path, content: str, dry_run: bool) -> None:
    """Write ``content`` only when ``path`` does not already exist."""

    if path.exists():
        print(f"keep    {readable_path(path)}")
        return
    print(f"create  {readable_path(path)}")
    if not dry_run:
        path.write_text(content, encoding="utf-8")


def copy_clean_notebook(source: Path, destination: Path, dry_run: bool) -> None:
    """Copy a tutorial as an unexecuted, learner-owned workspace."""

    if destination.exists():
        print(f"keep    {readable_path(destination)}")
        return
    print(f"copy    {readable_path(destination)} (clear saved outputs)")
    if dry_run:
        return

    notebook = json.loads(source.read_text(encoding="utf-8"))
    notebook_metadata = notebook.setdefault("metadata", {})
    notebook_metadata.pop("course_artifact", None)
    notebook_metadata["artifact_role"] = "learner_workspace"
    notebook_metadata["learner_evidence"] = False
    notebook_metadata["workspace_status"] = "not_started"
    notebook_metadata["source_tutorial"] = source.relative_to(REPO_ROOT).as_posix()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
        cell_metadata = cell.get("metadata", {})
        cell_metadata.pop("execution", None)
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def create_workspace(day_number: int, dry_run: bool = False) -> Path:
    """Create one Day experiment workspace without overwriting learner work."""

    if not 2 <= day_number <= 35:
        raise SystemExit("Learning workspaces are available for Day 2–35.")

    day_directory = find_day_directory(day_number)
    match = DAY_DIRECTORY_PATTERN.fullmatch(day_directory.name)
    if match is None:
        raise SystemExit(f"Unexpected Day directory name: {day_directory.name}")

    tutorial = day_directory / "tutorial.ipynb"
    if not tutorial.is_file():
        raise SystemExit(
            f"Day {day_number} tutorial is missing: "
            f"{tutorial.relative_to(REPO_ROOT)}"
        )

    destination = EXPERIMENTS_ROOT / day_directory.name
    results_directory = destination / "results"
    print(f"Day {day_number}: {day_title(day_directory)}")
    print(f"target  {destination.relative_to(REPO_ROOT)}")

    if not dry_run:
        results_directory.mkdir(parents=True, exist_ok=True)

    write_new_text(
        destination / "README.md",
        render_experiment_readme(day_title(day_directory), day_directory),
        dry_run,
    )
    write_new_text(
        destination / "notes.md",
        render_notes(day_title(day_directory)),
        dry_run,
    )
    write_new_text(
        results_directory / "README.md",
        (
            "# 结果文件\n\n"
            "只保存由本目录 Notebook 实际生成、且能够解释来源的 CSV、"
            "JSON 或图片。不要复制教学 Notebook 的预存输出冒充个人结果。\n"
        ),
        dry_run,
    )

    notebook_destination = destination / f"{day_directory.name}.ipynb"
    copy_clean_notebook(tutorial, notebook_destination, dry_run)

    if dry_run:
        print("Dry run only: no files were written.")
    else:
        print("Workspace ready. Open its README.md and run the Notebook.")
    return destination


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create one learner-owned Day 2–35 experiment workspace."
    )
    parser.add_argument("day", type=int, help="Day number, from 2 to 35.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned files without writing anything.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""

    args = parse_args()
    create_workspace(args.day, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
