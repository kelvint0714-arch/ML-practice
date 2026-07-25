"""Validate the Markdown-only daily curriculum structure.

This script does not mark any learning day complete and does not execute the
machine-learning experiments. It checks the repository layout, navigation,
local links, day continuity, required task-card sections, and Python fences.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_ROOT = REPO_ROOT / "curriculum"
CORE_ROOT = CURRICULUM_ROOT / "core"
OPTIONAL_GNN_ROOT = CURRICULUM_ROOT / "optional_gnn"
DAY_PATTERN = re.compile(r"day(\d{2})_[^/]+$")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PYTHON_FENCE_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)

REQUIRED_TOP_LEVEL = (
    "curriculum",
    "data",
    "docs",
    "experiments",
    "scripts",
)

REQUIRED_PHRASES = (
    "## 今天为什么学",
    "## 前置条件",
    "## 今日产出",
    "## 核心概念",
    "## 分步骤任务",
    "## 常见错误",
    "## 完成标准",
    "## 自测问题",
)


def check_repository_layout(errors: list[str]) -> None:
    """Report missing top-level directories in the documented layout."""

    for name in REQUIRED_TOP_LEVEL:
        if not (REPO_ROOT / name).is_dir():
            errors.append(f"Missing required top-level directory: {name}/")


def collect_day_directories(
    root: Path,
    expected_numbers: range,
    errors: list[str],
) -> list[Path]:
    """Return a route's Day directories and report gaps or duplicate numbers."""

    if not root.is_dir():
        errors.append(f"Missing curriculum route directory: {root}")
        return []

    day_dirs: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = DAY_PATTERN.fullmatch(path.name)
        if match:
            day_dirs.append((int(match.group(1)), path))

    day_dirs.sort()
    numbers = [number for number, _ in day_dirs]
    expected = list(expected_numbers)
    if numbers != expected:
        errors.append(
            f"{root.relative_to(REPO_ROOT)}/ must contain Day directories "
            f"{expected[0]:02d}–{expected[-1]:02d}; found {numbers}."
        )

    for number, path in day_dirs:
        if not (path / "README.md").exists():
            errors.append(f"Day {number:02d} is missing README.md: {path}")

    return [path for _, path in day_dirs]


def check_task_card_sections(day_dirs: list[Path], errors: list[str]) -> None:
    """Check the common structure for Day 02 onward."""

    for day_dir in day_dirs:
        match = DAY_PATTERN.fullmatch(day_dir.name)
        assert match is not None
        number = int(match.group(1))
        if number == 1:
            continue

        readme = day_dir / "README.md"
        if not readme.exists():
            continue

        text = readme.read_text(encoding="utf-8")
        for phrase in REQUIRED_PHRASES:
            if phrase not in text:
                errors.append(f"{readme}: missing section '{phrase}'.")

        if "核心代码" not in text and "核心文档" not in text:
            errors.append(f"{readme}: missing a core code/document skeleton.")
        if "上一天" not in text:
            errors.append(f"{readme}: missing previous-day navigation.")
        if number < 35 and "下一天" not in text:
            errors.append(f"{readme}: missing next-day navigation.")


def normalized_link_target(source: Path, raw_target: str) -> Path | None:
    """Resolve a local Markdown target or return None for external links."""

    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None

    target = unquote(target.split("#", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def check_markdown_links(errors: list[str]) -> None:
    """Check every local Markdown link after URL decoding."""

    for markdown in REPO_ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        if "/Users/" in text or "file://" in text:
            errors.append(f"{markdown}: contains a machine-specific local path.")

        for raw_target in LINK_PATTERN.findall(text):
            target = normalized_link_target(markdown, raw_target)
            if target is not None and not target.exists():
                errors.append(
                    f"{markdown}: broken local link '{raw_target}' "
                    f"-> '{target}'."
                )


def check_python_fences(errors: list[str]) -> int:
    """Compile curriculum Python fences without executing them."""

    fence_count = 0
    for markdown in CURRICULUM_ROOT.rglob("README.md"):
        text = markdown.read_text(encoding="utf-8")
        for index, code in enumerate(PYTHON_FENCE_PATTERN.findall(text), start=1):
            fence_count += 1
            try:
                compile(code, f"{markdown}:python-fence-{index}", "exec")
            except SyntaxError as exc:
                errors.append(
                    f"{markdown}: Python fence {index} has invalid syntax: {exc}"
                )
    return fence_count


def main() -> int:
    errors: list[str] = []
    check_repository_layout(errors)
    core_days = collect_day_directories(CORE_ROOT, range(1, 29), errors)
    gnn_days = collect_day_directories(OPTIONAL_GNN_ROOT, range(29, 36), errors)
    day_dirs = core_days + gnn_days
    check_task_card_sections(day_dirs, errors)
    check_markdown_links(errors)
    fence_count = check_python_fences(errors)

    if errors:
        print("Curriculum validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Curriculum validation passed: "
        f"{len(core_days)} core days, {len(gnn_days)} optional GNN days, "
        f"and {fence_count} Python fences checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
