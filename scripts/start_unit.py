"""Create a learner-owned starter workspace for one active-learning Unit.

The curriculum notebook remains an instructor-supplied artifact. This script
copies it to ``experiments/active_learning`` with outputs cleared and never
overwrites existing learner files. A new copy is explicitly marked
``not_started`` and is not learner evidence until it has been run, checked,
and explained.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = REPO_ROOT / "curriculum" / "active_learning"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments" / "active_learning"
UNIT_PATTERN = re.compile(r"unit(?P<number>\d{2})_(?P<slug>.+)")


def readable_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_unit_directory(unit_number: int) -> Path:
    """Return the unique curriculum directory for one Unit."""

    matches = sorted(ACTIVE_ROOT.glob(f"unit{unit_number:02d}_*"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one Unit {unit_number:02d} directory, "
            f"found {len(matches)}."
        )
    return matches[0]


def unit_title(unit_directory: Path) -> str:
    """Read the first heading from one Unit task card."""

    readme = unit_directory / "README.md"
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return unit_directory.name


def render_experiment_readme(title: str, unit_directory: Path) -> str:
    """Return the learner-facing workspace README."""

    curriculum_path = unit_directory.relative_to(REPO_ROOT)
    return f"""# {title}：实验工作区

> 状态：**待本人运行**

本目录预先提供完整起始代码，方便按学习目录直接开始。Notebook 的教学
预存输出已经清空；目录存在只表示“代码已准备”，不表示实验已经完成，
也不表示其中结果已经由本人复现。

## 课程来源

- 任务卡：[`{curriculum_path}/README.md`](../../../{curriculum_path}/README.md)
- 教学 Notebook：[`{curriculum_path}/tutorial.ipynb`](../../../{curriculum_path}/tutorial.ipynb)

## 完成规则

1. 从空内核运行本目录 Notebook；
2. 不删除断言、Random 基线或标签揭示线；
3. 在 `notes.md` 用自己的话解释输入、动作、输出；
4. 个人 CSV、JSON 或图片只放在本目录 `results/`；
5. 写清结果能说明什么、不能说明什么；
6. 完成自测后只更新全仓库唯一清单 `curriculum/PROGRESS.md`。

完成前不要改写上面的状态。教学人工数据不是粘合剂实验结果，离线
Oracle 也不代表真实实验已经完成。
"""


def render_notes(title: str) -> str:
    """Return a structured but empty learning record."""

    return f"""# {title}：学习记录

> 状态：待本人填写

## 今天完成了什么

-

## 输入、动作、输出

- 输入：
- 动作：
- 输出：

## 标签权限与 Oracle

- query 前可见：
- query 后返回：

## 我能独立解释的算法概念

-

## 我今天新认识的 Python 语法

-

## 结果能说明什么

-

## 结果不能说明什么

-

## 报错、原因与修复

- 完整报错：
- 原因：
- 修复：

## 我仍不能独立回答的问题

-
"""


def write_new_text(path: Path, content: str, dry_run: bool) -> None:
    """Write a text file only when it does not already exist."""

    if path.exists():
        print(f"keep    {readable_path(path)}")
        return
    print(f"create  {readable_path(path)}")
    if not dry_run:
        path.write_text(content, encoding="utf-8")


def copy_clean_notebook(
    source: Path,
    destination: Path,
    dry_run: bool,
) -> None:
    """Copy a tutorial without outputs and mark it as learner-owned."""

    if destination.exists():
        print(f"keep    {readable_path(destination)}")
        return
    print(f"copy    {readable_path(destination)} (clear saved outputs)")
    if dry_run:
        return

    notebook = json.loads(source.read_text(encoding="utf-8"))
    metadata = notebook.setdefault("metadata", {})
    metadata.pop("course_artifact", None)
    metadata["artifact_role"] = "learner_workspace"
    metadata["learner_evidence"] = False
    metadata["workspace_status"] = "not_started"
    metadata["source_tutorial"] = source.relative_to(REPO_ROOT).as_posix()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
        cell.get("metadata", {}).pop("execution", None)

    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def create_workspace(unit_number: int, dry_run: bool = False) -> Path:
    """Create one Unit workspace without overwriting learner work."""

    if not 1 <= unit_number <= 9:
        raise SystemExit("Active-learning workspaces are available for Unit 01–09.")

    unit_directory = find_unit_directory(unit_number)
    match = UNIT_PATTERN.fullmatch(unit_directory.name)
    if match is None:
        raise SystemExit(f"Unexpected Unit directory name: {unit_directory.name}")

    tutorial = unit_directory / "tutorial.ipynb"
    if not tutorial.is_file():
        raise SystemExit(
            f"Unit {unit_number:02d} tutorial is missing: "
            f"{tutorial.relative_to(REPO_ROOT)}"
        )

    destination = EXPERIMENTS_ROOT / unit_directory.name
    results_directory = destination / "results"
    print(f"Unit {unit_number:02d}: {unit_title(unit_directory)}")
    print(f"target  {destination.relative_to(REPO_ROOT)}")

    if not dry_run:
        results_directory.mkdir(parents=True, exist_ok=True)

    write_new_text(
        destination / "README.md",
        render_experiment_readme(unit_title(unit_directory), unit_directory),
        dry_run,
    )
    write_new_text(
        destination / "notes.md",
        render_notes(unit_title(unit_directory)),
        dry_run,
    )
    write_new_text(
        results_directory / "README.md",
        (
            "# 结果文件\n\n"
            "只保存由本目录 Notebook 实际生成、能够解释来源的 CSV、JSON "
            "或图片。不要复制课程预存输出冒充个人实验结果。\n"
        ),
        dry_run,
    )
    copy_clean_notebook(
        tutorial,
        destination / f"{unit_directory.name}.ipynb",
        dry_run,
    )

    if dry_run:
        print("Dry run only: no files were written.")
    else:
        print("Workspace ready. Open its README.md and run the Notebook.")
    return destination


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create one learner-owned active-learning Unit workspace."
    )
    parser.add_argument("unit", type=int, help="Unit number, from 1 to 9.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""

    args = parse_args()
    create_workspace(args.unit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
