"""Small, read-only helpers for learners.

The command has two jobs:

* ``status`` reads the canonical curriculum checklist and reports the next item.
* ``doctor`` checks Python, pinned dependencies, and repository structure.

Neither command installs packages, edits progress, or runs model training.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_FILE = REPO_ROOT / "curriculum" / "PROGRESS.md"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_repository.py"
TRACK_REQUIREMENTS = {
    "core": REPO_ROOT / "requirements-learning.txt",
    "active-learning": REPO_ROOT / "requirements-active-learning.txt",
    "gnn": REPO_ROOT / "requirements-gnn.txt",
}
CHECKBOX_PATTERN = re.compile(
    r"^- \[(?P<mark>[^]])\]\s+\*\*(?P<title>[^*]+)\*\*"
    r"(?P<rest>.*)$",
    re.MULTILINE,
)
LINK_PATTERN = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")
PIN_PATTERN = re.compile(r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s;]+)$")
DONE_MARKS = {"x", "X", "√", "✓"}


@dataclass(frozen=True)
class LessonItem:
    """One checkbox from the canonical learning checklist."""

    title: str
    completed: bool
    route: str
    target: str | None


@dataclass(frozen=True)
class DependencyResult:
    """Installed state for one pinned dependency."""

    name: str
    expected: str
    installed: str | None


def route_for_title(title: str) -> str:
    """Map a checklist title to the core, active-learning, or GNN route."""

    if title.startswith("Unit "):
        return "active-learning"
    match = re.match(r"Day (\d+)", title)
    if match and int(match.group(1)) >= 29:
        return "gnn"
    return "core"


def parse_progress(text: str) -> list[LessonItem]:
    """Parse learning checkboxes without changing the Markdown source."""

    items: list[LessonItem] = []
    for match in CHECKBOX_PATTERN.finditer(text):
        title = match.group("title").rstrip("：: ")
        link = LINK_PATTERN.search(match.group("rest"))
        items.append(
            LessonItem(
                title=title,
                completed=match.group("mark") in DONE_MARKS,
                route=route_for_title(title),
                target=link.group("target") if link else None,
            )
        )
    return items


def progress_items() -> list[LessonItem]:
    """Read and parse the canonical progress file."""

    try:
        text = PROGRESS_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read {PROGRESS_FILE}: {exc}") from exc
    items = parse_progress(text)
    if not items:
        raise SystemExit("No learning checkboxes were found in curriculum/PROGRESS.md.")
    return items


def print_progress(route: str = "all") -> int:
    """Print completion counts and the first pending lesson."""

    items = progress_items()
    selected = items if route == "all" else [item for item in items if item.route == route]
    if not selected:
        raise SystemExit(f"No checklist items found for route {route!r}.")

    completed = sum(item.completed for item in selected)
    percent = completed / len(selected) * 100
    print("ML-Learning 学习进度")
    print(f"范围：{route} | 已完成：{completed}/{len(selected)} ({percent:.1f}%)")

    if route == "all":
        for route_name in ("core", "active-learning", "gnn"):
            route_items = [item for item in items if item.route == route_name]
            route_done = sum(item.completed for item in route_items)
            print(f"- {route_name}: {route_done}/{len(route_items)}")

    next_item = next((item for item in selected if not item.completed), None)
    if next_item is None:
        print("下一项：当前范围已全部完成。请回顾阶段产物和结论边界。")
        return 0

    print(f"下一项：{next_item.title}")
    if next_item.target:
        target = (PROGRESS_FILE.parent / next_item.target).resolve()
        try:
            display_target = target.relative_to(REPO_ROOT)
        except ValueError:
            display_target = target
        print(f"打开：{display_target}")
    print("提示：完成自测后，由本人更新 curriculum/PROGRESS.md。")
    return 0


def load_pinned_requirements(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    """Load exact pins, following local ``-r`` includes."""

    resolved = path.resolve()
    visited = set() if seen is None else seen
    if resolved in visited:
        return {}
    visited.add(resolved)

    pins: dict[str, str] = {}
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            include_name = line.split(maxsplit=1)[1]
            pins.update(load_pinned_requirements(resolved.parent / include_name, visited))
            continue
        match = PIN_PATTERN.fullmatch(line)
        if match:
            pins[match.group("name")] = match.group("version")
    return pins


def dependency_results(
    requirement_file: Path,
    version_reader: Callable[[str], str] = metadata.version,
) -> list[DependencyResult]:
    """Return installed versions without importing heavy ML libraries."""

    results: list[DependencyResult] = []
    for name, expected in load_pinned_requirements(requirement_file).items():
        try:
            installed = version_reader(name)
        except metadata.PackageNotFoundError:
            installed = None
        results.append(DependencyResult(name, expected, installed))
    return results


def print_doctor(track: str, run_repository_check: bool = True) -> int:
    """Print actionable environment diagnostics and return a shell status."""

    print("ML-Learning 环境诊断")
    print(f"学习路线：{track}")
    print(f"Python：{sys.version.split()[0]} ({sys.executable})")

    failures = 0
    warnings = 0
    if sys.version_info[:2] == (3, 10):
        print("[OK] Python 3.10 与课程验证版本一致")
    elif sys.version_info >= (3, 10):
        warnings += 1
        print("[WARN] 课程在 Python 3.10 上验证；当前版本可能可用但未锁定验证")
    else:
        failures += 1
        print("[MISSING] 需要 Python 3.10 或更高版本")

    required_paths = (PROGRESS_FILE, CHECK_SCRIPT, TRACK_REQUIREMENTS[track])
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        failures += len(missing_paths)
        for path in missing_paths:
            print(f"[MISSING] 课程文件：{path}")
    else:
        print("[OK] 仓库结构和路线文件存在")

    requirement_file = TRACK_REQUIREMENTS[track]
    if requirement_file.is_file():
        for result in dependency_results(requirement_file):
            if result.installed is None:
                failures += 1
                print(f"[MISSING] {result.name}（课程版本 {result.expected}）")
            elif result.installed != result.expected:
                warnings += 1
                print(
                    f"[WARN] {result.name} {result.installed}；"
                    f"课程验证版本 {result.expected}"
                )
            else:
                print(f"[OK] {result.name} {result.installed}")

    if run_repository_check and CHECK_SCRIPT.is_file():
        completed = subprocess.run(
            [sys.executable, str(CHECK_SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            print("[OK] 仓库课程结构检查通过")
        else:
            failures += 1
            print("[MISSING] 仓库课程结构检查失败")
            output = (completed.stdout + completed.stderr).strip()
            if output:
                print(output)

    print(f"诊断汇总：{failures} 个阻断问题，{warnings} 个版本提醒")
    if failures:
        print(f"下一步：python -m pip install -r {requirement_file.name}")
        return 1
    print("环境已具备当前路线的基础运行条件。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Read-only learning progress and environment helpers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show progress and the next item.")
    status.add_argument(
        "--route",
        choices=("all", "core", "active-learning", "gnn"),
        default="all",
        help="Limit the progress summary to one route.",
    )

    doctor = subparsers.add_parser("doctor", help="Check one course environment.")
    doctor.add_argument(
        "--track",
        choices=tuple(TRACK_REQUIREMENTS),
        default="core",
        help="Dependency set to check.",
    )
    doctor.add_argument(
        "--skip-repository-check",
        action="store_true",
        help="Only check Python and installed dependencies.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the learner helper."""

    args = build_parser().parse_args(argv)
    if args.command == "status":
        return print_progress(args.route)
    return print_doctor(args.track, not args.skip_repository_check)


if __name__ == "__main__":
    raise SystemExit(main())
