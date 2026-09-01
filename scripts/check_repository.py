"""Validate the ML-Learning curriculum without running model training.

The checker covers repository layout, Day/Unit continuity, Markdown links,
Python code fences, saved Notebook state, tutorial outputs, and the retained
ESOL teaching baseline. It never edits learner progress.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_ROOT = REPO_ROOT / "curriculum"
CORE_ROOT = CURRICULUM_ROOT / "core"
OPTIONAL_GNN_ROOT = CURRICULUM_ROOT / "optional_gnn"
ACTIVE_LEARNING_ROOT = CURRICULUM_ROOT / "active_learning"
DAY_PATTERN = re.compile(r"day(\d{2})_[^/]+$")
UNIT_PATTERN = re.compile(r"unit(\d{2})_[^/]+$")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PYTHON_FENCE_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)

REQUIRED_ROOT_FILES = (
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "README.md",
    "requirements-gnn.txt",
    "requirements-learning.txt",
    "requirements-active-learning.txt",
    "requirements.txt",
)
REQUIRED_TOP_LEVEL = (
    ".github",
    "curriculum",
    "data",
    "docs",
    "learning_outputs",
    "scripts",
    "tests",
)
REQUIRED_PUBLIC_GUIDES = {
    "README.md": (
        "## 这套课程适合谁",
        "## 学完能做到什么",
        "## 10 分钟开始学习",
        "curriculum/LEARNING_PATHS.md",
        "scripts/learn.py doctor",
    ),
    "curriculum/LEARNING_PATHS.md": (
        "## 先做 30 秒选择",
        "## 每个学习单元怎样算完成",
        "## 路线 D：主动学习",
    ),
    "docs/getting_started.md": (
        "## 第二步：建立核心课程环境",
        "## 第三步：查看下一项任务",
        "scripts/learn.py doctor",
    ),
    "docs/teaching_guide.md": (
        "## 统一验收量表",
        "## 课程维护原则",
    ),
    "CONTRIBUTING.md": (
        "## 教学内容的最低结构",
        "## Notebook 规范",
        "## 本地检查",
    ),
}
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

ESOL_ROOT = CORE_ROOT / "day07_integrated_baseline" / "reference_baseline"
ESOL_NOTEBOOK = ESOL_ROOT / "esol_baseline.ipynb"
ESOL_METRICS = ESOL_ROOT / "results" / "baseline_metrics.csv"
ESOL_CONFIG = ESOL_ROOT / "results" / "run_config.json"
EXPECTED_CORE_DAY_DIRS = (
    "day01_beginner",
    "day02_metrics",
    "day03_ridge",
    "day04_decision_tree",
    "day05_random_forest",
    "day06_gradient_boosting",
    "day07_integrated_baseline",
    "day08_split_protocol",
    "day09_cross_validation",
    "day10_seed_stability",
    "day11_pipeline_leakage",
    "day12_tuning_without_test",
    "day13_fair_comparison",
    "day14_ml_stage_report",
    "day15_tensor_shape",
    "day16_mlp_forward",
    "day17_loss_optimizer",
    "day18_batch_epoch_loop",
    "day19_mlp_esol",
    "day20_regularization_early_stopping",
    "day21_mlp_vs_ml",
    "day22_hybrid_risk",
    "day23_oof_stacking",
    "day24_hybrid_ablation",
    "day25_uncertainty",
    "day26_active_learning",
    "day27_paper_to_schema",
    "day28_capstone_handoff",
)
EXPECTED_GNN_DAY_DIRS = (
    "day29_graph_basics",
    "day30_pyg_data",
    "day31_gcn",
    "day32_graphsage",
    "day33_gat",
    "day34_gin_graph_classification",
    "day35_molecular_graph_gate",
)
EXPECTED_ACTIVE_LEARNING_UNIT_DIRS = (
    "unit01_foundations",
    "unit02_surrogates_uncertainty",
    "unit03_acquisition_functions",
    "unit04_multiround_loop",
    "unit05_benchmark_protocol",
    "unit06_batch_diversity_constraints",
    "unit07_neural_graph_surrogates",
    "unit08_physics_closed_loop",
    "unit09_capstone",
)
LEARNING_PACKAGE_FILES = (
    "01_concepts.md",
    "02_algorithm_walkthrough.md",
    "tutorial.ipynb",
    "03_exercises.md",
    "04_reference_answers.md",
)
DAY13_OUTPUT_ROOT = CORE_ROOT / "day13_fair_comparison" / "tutorial_outputs"
DAY14_OUTPUT_ROOT = CORE_ROOT / "day14_ml_stage_report" / "tutorial_outputs"
DAY13_FOLD_METRICS = DAY13_OUTPUT_ROOT / "fold_metrics.csv"
DAY13_MODEL_SUMMARY = DAY13_OUTPUT_ROOT / "model_summary.csv"
DAY14_STAGE_SUMMARY = DAY14_OUTPUT_ROOT / "ml_stage_summary.csv"
DAY14_REPORT_PREVIEW = DAY14_OUTPUT_ROOT / "report_preview.md"
TUTORIAL_OUTPUT_PROVENANCE_MARKERS = (
    "artifact_kind: deterministic_synthetic_tutorial",
    "learner_evidence: false",
    "不是 ESOL",
    "下游任务",
)
EXPECTED_TUTORIAL_MODELS = {
    "dummy",
    "ridge",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
}
LEGACY_DAY_SLUGS = (
    "day02_safe_rerun",
    "day03_metrics",
    "day06_ridge_scaling",
    "day07_gradient_boosting",
    "day09_cross_validation_oof",
)


def relative(path: Path) -> str:
    """Return a readable repository-relative path."""

    return str(path.relative_to(REPO_ROOT))


def is_generated_or_internal(path: Path) -> bool:
    """Skip Git internals and local generated directories."""

    ignored_parts = {
        ".cache",
        ".git",
        ".ipynb_checkpoints",
        "__pycache__",
    }
    return any(part in ignored_parts for part in path.relative_to(REPO_ROOT).parts)


def check_repository_layout(errors: list[str]) -> None:
    """Report missing root files and top-level directories."""

    for name in REQUIRED_ROOT_FILES:
        if not (REPO_ROOT / name).is_file():
            errors.append(f"Missing required root file: {name}")
    for name in REQUIRED_TOP_LEVEL:
        if not (REPO_ROOT / name).is_dir():
            errors.append(f"Missing required top-level directory: {name}/")


def check_public_guides(errors: list[str]) -> None:
    """Keep the learner and contributor entry points complete."""

    for relative_path, required_phrases in REQUIRED_PUBLIC_GUIDES.items():
        path = REPO_ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing public guide: {relative_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in required_phrases:
            if phrase not in text:
                errors.append(
                    f"{relative_path}: missing learner-facing marker {phrase!r}."
                )


def collect_day_directories(
    root: Path,
    expected_numbers: range,
    errors: list[str],
) -> list[Path]:
    """Return one route's Day directories and report gaps."""

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
            f"{relative(root)}/ must contain Day directories "
            f"{expected[0]:02d}–{expected[-1]:02d}; found {numbers}."
        )

    for number, path in day_dirs:
        if not (path / "README.md").exists():
            errors.append(f"Day {number:02d} is missing README.md: {path}")
    return [path for _, path in day_dirs]


def collect_unit_directories(errors: list[str]) -> list[Path]:
    """Return active-learning Unit directories and report gaps."""

    if not ACTIVE_LEARNING_ROOT.is_dir():
        errors.append(
            "Missing curriculum route directory: "
            f"{relative(ACTIVE_LEARNING_ROOT)}"
        )
        return []

    unit_dirs: list[tuple[int, Path]] = []
    for path in ACTIVE_LEARNING_ROOT.iterdir():
        if not path.is_dir():
            continue
        match = UNIT_PATTERN.fullmatch(path.name)
        if match:
            unit_dirs.append((int(match.group(1)), path))

    unit_dirs.sort()
    numbers = [number for number, _ in unit_dirs]
    expected = list(range(1, 10))
    if numbers != expected:
        errors.append(
            f"{relative(ACTIVE_LEARNING_ROOT)}/ must contain Unit directories "
            f"01–09; found {numbers}."
        )

    for number, path in unit_dirs:
        if not (path / "README.md").exists():
            errors.append(f"Unit {number:02d} is missing README.md: {path}")
    return [path for _, path in unit_dirs]


def check_task_card_sections(day_dirs: list[Path], errors: list[str]) -> None:
    """Check the common task-card structure for Day 02 onward."""

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
                errors.append(f"{relative(readme)}: missing section '{phrase}'.")
        if "核心代码" not in text and "核心文档" not in text:
            errors.append(f"{relative(readme)}: missing core code/document skeleton.")
        if "上一天" not in text:
            errors.append(f"{relative(readme)}: missing previous-day navigation.")
        if number < 35 and "下一天" not in text:
            errors.append(f"{relative(readme)}: missing next-day navigation.")


def check_unit_task_card_sections(
    unit_dirs: list[Path],
    errors: list[str],
) -> None:
    """Check the common active-learning Unit task-card structure."""

    for unit_dir in unit_dirs:
        match = UNIT_PATTERN.fullmatch(unit_dir.name)
        assert match is not None
        number = int(match.group(1))
        readme = unit_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")

        for phrase in REQUIRED_PHRASES:
            if phrase not in text:
                errors.append(f"{relative(readme)}: missing section '{phrase}'.")
        if "核心代码" not in text and "核心文档" not in text:
            errors.append(f"{relative(readme)}: missing core code/document skeleton.")
        if "上一单元" not in text:
            errors.append(
                f"{relative(readme)}: missing previous-Unit navigation."
            )
        if number < 9 and "下一单元" not in text:
            errors.append(f"{relative(readme)}: missing next-Unit navigation.")


def check_complete_learning_packages(
    day_dirs: list[Path],
    errors: list[str],
) -> tuple[int, int]:
    """Validate the detailed Day 02–35 lesson files and tutorial notebooks."""

    notebook_count = 0
    code_cell_count = 0
    for day_dir in day_dirs:
        match = DAY_PATTERN.fullmatch(day_dir.name)
        assert match is not None
        number = int(match.group(1))
        if number == 1:
            continue

        missing = [
            name
            for name in LEARNING_PACKAGE_FILES
            if not (day_dir / name).is_file()
        ]
        if missing:
            errors.append(
                f"{relative(day_dir)}/: incomplete learning package; "
                f"missing {missing}."
            )
            continue

        readme = day_dir / "README.md"
        linked_targets = local_link_targets(readme)
        linked_sequence = local_link_target_sequence(readme)
        package_positions: list[int] = []
        for name in LEARNING_PACKAGE_FILES:
            expected = (day_dir / name).resolve()
            if expected not in linked_targets:
                errors.append(
                    f"{relative(readme)}: must link to learning-package file "
                    f"'{name}'."
                )
                continue
            package_positions.append(linked_sequence.index(expected))
        if (
            len(package_positions) == len(LEARNING_PACKAGE_FILES)
            and package_positions != sorted(package_positions)
        ):
            errors.append(
                f"{relative(readme)}: learning-package links must follow "
                f"{list(LEARNING_PACKAGE_FILES)}."
            )

        notebook_path = day_dir / "tutorial.ipynb"
        notebook_count += 1
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{relative(notebook_path)}: invalid Notebook JSON: {exc}")
            continue

        if notebook.get("nbformat") != 4:
            errors.append(
                f"{relative(notebook_path)}: expected nbformat 4; "
                f"found {notebook.get('nbformat')}."
            )

        course_artifact = notebook.get("metadata", {}).get("course_artifact", {})
        if course_artifact.get("kind") != "supplied_tutorial":
            errors.append(
                f"{relative(notebook_path)}: metadata.course_artifact.kind must "
                "be 'supplied_tutorial'."
            )
        if course_artifact.get("learner_evidence") is not False:
            errors.append(
                f"{relative(notebook_path)}: course tutorial metadata must set "
                "learner_evidence=false."
            )

        kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
        expected_display_name = (
            "Python 3 (gnn)" if number >= 29 else "Python 3 (esol)"
        )
        if kernelspec.get("display_name") != expected_display_name:
            errors.append(
                f"{relative(notebook_path)}: kernelspec.display_name must be "
                f"{expected_display_name!r}; found "
                f"{kernelspec.get('display_name')!r}."
            )
        if kernelspec.get("name") != "python3":
            errors.append(
                f"{relative(notebook_path)}: kernelspec.name must remain the "
                "portable value 'python3'."
            )

        cells = notebook.get("cells", [])
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "markdown"
        )
        for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
            if heading not in markdown_text:
                errors.append(
                    f"{relative(notebook_path)}: missing tutorial section "
                    f"'{heading}'."
                )
        if f"Day {number}" not in markdown_text:
            errors.append(
                f"{relative(notebook_path)}: must identify itself as Day {number}."
            )

        code_cells = [
            cell for cell in cells if cell.get("cell_type") == "code"
        ]
        code_cell_count += len(code_cells)
        if len(code_cells) < 2:
            errors.append(
                f"{relative(notebook_path)}: expected at least two focused code "
                "cells."
            )
            continue

        execution_counts = [
            cell.get("execution_count") for cell in code_cells
        ]
        expected_counts = list(range(1, len(code_cells) + 1))
        if execution_counts != expected_counts:
            errors.append(
                f"{relative(notebook_path)}: expected consecutive execution "
                f"counts {expected_counts}; found {execution_counts}."
            )

        for cell_index, cell in enumerate(code_cells, start=1):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(
                        f"{relative(notebook_path)}: code cell {cell_index} "
                        f"contains saved error {output.get('ename')}: "
                        f"{output.get('evalue')}"
                    )

        serialized = json.dumps(notebook, ensure_ascii=False)
        if any(
            marker in serialized
            for marker in ("/Users/", "/tmp/", "/private/tmp/", "file://")
        ):
            errors.append(
                f"{relative(notebook_path)}: contains a machine-specific path."
            )

    return notebook_count, code_cell_count


def check_complete_unit_packages(
    unit_dirs: list[Path],
    errors: list[str],
) -> tuple[int, int]:
    """Validate Unit 01–09 files and executed tutorial notebooks."""

    notebook_count = 0
    code_cell_count = 0
    for unit_dir in unit_dirs:
        match = UNIT_PATTERN.fullmatch(unit_dir.name)
        assert match is not None
        number = int(match.group(1))

        missing = [
            name
            for name in LEARNING_PACKAGE_FILES
            if not (unit_dir / name).is_file()
        ]
        if missing:
            errors.append(
                f"{relative(unit_dir)}/: incomplete Unit learning package; "
                f"missing {missing}."
            )
            continue

        readme = unit_dir / "README.md"
        linked_targets = local_link_targets(readme)
        linked_sequence = local_link_target_sequence(readme)
        package_positions: list[int] = []
        for name in LEARNING_PACKAGE_FILES:
            expected = (unit_dir / name).resolve()
            if expected not in linked_targets:
                errors.append(
                    f"{relative(readme)}: must link to Unit package file "
                    f"'{name}'."
                )
                continue
            package_positions.append(linked_sequence.index(expected))
        if (
            len(package_positions) == len(LEARNING_PACKAGE_FILES)
            and package_positions != sorted(package_positions)
        ):
            errors.append(
                f"{relative(readme)}: Unit package links must follow "
                f"{list(LEARNING_PACKAGE_FILES)}."
            )

        notebook_path = unit_dir / "tutorial.ipynb"
        notebook_count += 1
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{relative(notebook_path)}: invalid Notebook JSON: {exc}")
            continue

        if notebook.get("nbformat") != 4:
            errors.append(
                f"{relative(notebook_path)}: expected nbformat 4; "
                f"found {notebook.get('nbformat')}."
            )

        course_artifact = notebook.get("metadata", {}).get("course_artifact", {})
        expected_metadata = {
            "kind": "supplied_tutorial",
            "learner_evidence": False,
            "route": "active_learning",
            "unit": number,
        }
        for key, expected in expected_metadata.items():
            if course_artifact.get(key) != expected:
                errors.append(
                    f"{relative(notebook_path)}: metadata.course_artifact."
                    f"{key} must be {expected!r}; found "
                    f"{course_artifact.get(key)!r}."
                )

        kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
        if kernelspec.get("display_name") != "Python 3 (esol)":
            errors.append(
                f"{relative(notebook_path)}: kernelspec.display_name must be "
                "'Python 3 (esol)'."
            )
        if kernelspec.get("name") != "python3":
            errors.append(
                f"{relative(notebook_path)}: kernelspec.name must remain "
                "'python3'."
            )

        cells = notebook.get("cells", [])
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "markdown"
        )
        for heading in (
            "## Goal",
            "## Setup",
            "## Steps",
            "## Checks",
            "## Next Steps",
        ):
            if heading not in markdown_text:
                errors.append(
                    f"{relative(notebook_path)}: missing tutorial section "
                    f"'{heading}'."
                )
        if f"Unit {number:02d}" not in markdown_text:
            errors.append(
                f"{relative(notebook_path)}: must identify itself as "
                f"Unit {number:02d}."
            )

        code_cells = [
            cell for cell in cells if cell.get("cell_type") == "code"
        ]
        code_cell_count += len(code_cells)
        if len(code_cells) < 2:
            errors.append(
                f"{relative(notebook_path)}: expected at least two focused "
                "code cells."
            )
            continue

        execution_counts = [
            cell.get("execution_count") for cell in code_cells
        ]
        expected_counts = list(range(1, len(code_cells) + 1))
        if execution_counts != expected_counts:
            errors.append(
                f"{relative(notebook_path)}: expected consecutive execution "
                f"counts {expected_counts}; found {execution_counts}."
            )

        for cell_index, cell in enumerate(code_cells, start=1):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(
                        f"{relative(notebook_path)}: code cell {cell_index} "
                        f"contains saved error {output.get('ename')}: "
                        f"{output.get('evalue')}"
                    )

        serialized = json.dumps(notebook, ensure_ascii=False)
        if any(
            marker in serialized
            for marker in ("/Users/", "/tmp/", "/private/tmp/", "file://")
        ):
            errors.append(
                f"{relative(notebook_path)}: contains a machine-specific path."
            )

    return notebook_count, code_cell_count


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


def local_link_targets(markdown: Path) -> set[Path]:
    """Return normalized local link targets from one Markdown file."""

    text = markdown.read_text(encoding="utf-8")
    targets = {
        normalized
        for raw_target in LINK_PATTERN.findall(text)
        if (normalized := normalized_link_target(markdown, raw_target)) is not None
    }
    return targets


def local_link_target_sequence(markdown: Path) -> list[Path]:
    """Return local Markdown targets in their first-seen document order."""

    text = markdown.read_text(encoding="utf-8")
    return [
        normalized
        for raw_target in LINK_PATTERN.findall(text)
        if (normalized := normalized_link_target(markdown, raw_target)) is not None
    ]


def check_curriculum_routes(
    core_days: list[Path],
    gnn_days: list[Path],
    errors: list[str],
) -> None:
    """Check canonical slugs, route coverage, and previous/next navigation."""

    expected_core_paths = [CORE_ROOT / name for name in EXPECTED_CORE_DAY_DIRS]
    expected_gnn_paths = [
        OPTIONAL_GNN_ROOT / name for name in EXPECTED_GNN_DAY_DIRS
    ]
    if core_days != expected_core_paths:
        errors.append(
            "Core Day directory names do not match the canonical algorithm-first "
            f"route: {[path.name for path in core_days]}."
        )
    if gnn_days != expected_gnn_paths:
        errors.append(
            "Optional GNN directory names do not match the canonical route: "
            f"{[path.name for path in gnn_days]}."
        )

    route_expectations = {
        CURRICULUM_ROOT / "PROGRESS.md": expected_core_paths + expected_gnn_paths,
        CORE_ROOT / "README.md": expected_core_paths,
        OPTIONAL_GNN_ROOT / "README.md": expected_gnn_paths,
    }
    for route_file, expected_paths in route_expectations.items():
        if not route_file.is_file():
            errors.append(f"Missing curriculum route file: {relative(route_file)}.")
            continue
        targets = local_link_targets(route_file)
        expected_readmes = {path / "README.md" for path in expected_paths}
        missing = expected_readmes - targets
        if missing:
            errors.append(
                f"{relative(route_file)}: missing canonical Day links "
                f"{[relative(path) for path in sorted(missing)]}."
            )

    all_days = core_days + gnn_days
    by_number = {
        int(DAY_PATTERN.fullmatch(path.name).group(1)): path
        for path in all_days
        if DAY_PATTERN.fullmatch(path.name)
    }
    progress = (CURRICULUM_ROOT / "PROGRESS.md").resolve()

    for number, day_dir in by_number.items():
        readme = day_dir / "README.md"
        targets = local_link_targets(readme)
        if progress not in targets:
            errors.append(
                f"{relative(readme)}: must link back to curriculum/PROGRESS.md."
            )

        if number > 1:
            previous_dir = by_number.get(number - 1)
            if previous_dir is None:
                continue
            expected_previous = (previous_dir / "README.md").resolve()
            if expected_previous not in targets:
                errors.append(
                    f"{relative(readme)}: missing canonical previous Day "
                    f"{relative(expected_previous)}."
                )

        if number < 28 or 29 <= number < 35:
            next_dir = by_number.get(number + 1)
            if next_dir is None:
                continue
            expected_next = (next_dir / "README.md").resolve()
            if expected_next not in targets:
                errors.append(
                    f"{relative(readme)}: missing canonical next Day "
                    f"{relative(expected_next)}."
                )
        elif number == 28:
            gnn_gate = (OPTIONAL_GNN_ROOT / "README.md").resolve()
            if gnn_gate not in targets:
                errors.append(
                    f"{relative(readme)}: Day 28 must route through the GNN gate."
                )

    markdown_files = [
        path
        for path in REPO_ROOT.rglob("*.md")
        if not is_generated_or_internal(path)
    ]
    for markdown in markdown_files:
        text = markdown.read_text(encoding="utf-8")
        for legacy_slug in LEGACY_DAY_SLUGS:
            if legacy_slug in text:
                errors.append(
                    f"{relative(markdown)}: contains legacy Day slug "
                    f"'{legacy_slug}'."
                )


def check_active_learning_routes(
    unit_dirs: list[Path],
    errors: list[str],
) -> None:
    """Check Unit slugs, route coverage, and previous/next navigation."""

    expected_unit_paths = [
        ACTIVE_LEARNING_ROOT / name
        for name in EXPECTED_ACTIVE_LEARNING_UNIT_DIRS
    ]
    if unit_dirs != expected_unit_paths:
        errors.append(
            "Active-learning Unit directory names do not match the canonical "
            f"route: {[path.name for path in unit_dirs]}."
        )

    route_files = (
        ACTIVE_LEARNING_ROOT / "README.md",
        ACTIVE_LEARNING_ROOT / "PROGRESS.md",
        CURRICULUM_ROOT / "PROGRESS.md",
    )
    expected_readmes = {
        path / "README.md" for path in expected_unit_paths
    }
    for route_file in route_files:
        if not route_file.is_file():
            errors.append(
                f"Missing active-learning route file: {relative(route_file)}."
            )
            continue
        targets = local_link_targets(route_file)
        missing = expected_readmes - targets
        if missing:
            errors.append(
                f"{relative(route_file)}: missing canonical Unit links "
                f"{[relative(path) for path in sorted(missing)]}."
            )

    topic_index = ACTIVE_LEARNING_ROOT / "PROGRESS.md"
    if topic_index.is_file():
        topic_text = topic_index.read_text(encoding="utf-8")
        if re.search(r"^- \[[ xX]\].*Unit", topic_text, re.MULTILINE):
            errors.append(
                f"{relative(topic_index)}: must remain a route index without "
                "duplicate Unit completion checkboxes; use curriculum/PROGRESS.md."
            )

    progress = (CURRICULUM_ROOT / "PROGRESS.md").resolve()
    by_number = {
        int(UNIT_PATTERN.fullmatch(path.name).group(1)): path
        for path in unit_dirs
        if UNIT_PATTERN.fullmatch(path.name)
    }
    for number, unit_dir in by_number.items():
        readme = unit_dir / "README.md"
        targets = local_link_targets(readme)
        if progress not in targets:
            errors.append(
                f"{relative(readme)}: must link back to "
                "curriculum/PROGRESS.md."
            )

        if number > 1:
            previous_dir = by_number.get(number - 1)
            if previous_dir is None:
                continue
            expected_previous = (previous_dir / "README.md").resolve()
            if expected_previous not in targets:
                errors.append(
                    f"{relative(readme)}: missing canonical previous Unit "
                    f"{relative(expected_previous)}."
                )

        if number < 9:
            next_dir = by_number.get(number + 1)
            if next_dir is None:
                continue
            expected_next = (next_dir / "README.md").resolve()
            if expected_next not in targets:
                errors.append(
                    f"{relative(readme)}: missing canonical next Unit "
                    f"{relative(expected_next)}."
                )


def check_learning_artifact_contract(errors: list[str]) -> None:
    """Validate the Day 13 fixture and the derived Day 14 teaching report."""

    day13_path = CORE_ROOT / "day13_fair_comparison" / "README.md"
    day14_path = CORE_ROOT / "day14_ml_stage_report" / "README.md"
    if not day13_path.is_file() or not day14_path.is_file():
        errors.append("Day 13/14 task cards are missing; cannot check artifact contract.")
        return

    day13 = day13_path.read_text(encoding="utf-8")
    day14 = day14_path.read_text(encoding="utf-8")
    shared_markers = (
        "learning_outputs/day13_fair_comparison/results",
        "fold_metrics.csv",
        '"model"',
        '"fold"',
        '"split"',
        '"mae"',
        '"rmse"',
        '"r2"',
    )
    for marker in shared_markers:
        if marker not in day13 or marker not in day14:
            errors.append(
                "Day 13/14 artifact contract is inconsistent; missing shared "
                f"marker {marker!r}."
            )

    for output_root in (DAY13_OUTPUT_ROOT, DAY14_OUTPUT_ROOT):
        provenance = output_root / "README.md"
        if not provenance.is_file():
            errors.append(
                f"Missing tutorial-output provenance: {relative(provenance)}."
            )
            continue
        provenance_text = provenance.read_text(encoding="utf-8")
        for marker in TUTORIAL_OUTPUT_PROVENANCE_MARKERS:
            if marker not in provenance_text:
                errors.append(
                    f"{relative(provenance)}: missing provenance marker "
                    f"{marker!r}."
                )

    fold_columns, fold_rows = read_csv_artifact(DAY13_FOLD_METRICS, errors)
    required_fold_columns = {
        "model",
        "fold",
        "split",
        "mae",
        "rmse",
        "r2",
    }
    missing_fold_columns = required_fold_columns - fold_columns
    if missing_fold_columns:
        errors.append(
            f"{relative(DAY13_FOLD_METRICS)}: missing columns "
            f"{sorted(missing_fold_columns)}."
        )
    if len(fold_rows) != 25:
        errors.append(
            f"{relative(DAY13_FOLD_METRICS)}: expected 25 model/fold rows; "
            f"found {len(fold_rows)}."
        )

    fold_keys: set[tuple[str, int]] = set()
    fold_numbers_by_model: dict[str, set[int]] = {}
    for row_number, row in enumerate(fold_rows, start=2):
        model = row.get("model", "")
        try:
            fold = int(row.get("fold", ""))
        except (TypeError, ValueError):
            errors.append(
                f"{relative(DAY13_FOLD_METRICS)}:{row_number}: invalid fold "
                f"{row.get('fold')!r}."
            )
            continue
        key = (model, fold)
        if key in fold_keys:
            errors.append(
                f"{relative(DAY13_FOLD_METRICS)}:{row_number}: duplicate "
                f"model/fold {key}."
            )
        fold_keys.add(key)
        fold_numbers_by_model.setdefault(model, set()).add(fold)
        if row.get("split") != "cv_valid":
            errors.append(
                f"{relative(DAY13_FOLD_METRICS)}:{row_number}: split must be "
                "'cv_valid'."
            )
        for column in ("mae", "rmse", "r2"):
            check_finite_csv_value(
                DAY13_FOLD_METRICS,
                row_number,
                column,
                row.get(column),
                errors,
            )
    if set(fold_numbers_by_model) != EXPECTED_TUTORIAL_MODELS:
        errors.append(
            f"{relative(DAY13_FOLD_METRICS)}: expected models "
            f"{sorted(EXPECTED_TUTORIAL_MODELS)}; found "
            f"{sorted(fold_numbers_by_model)}."
        )
    for model, folds in sorted(fold_numbers_by_model.items()):
        if folds != {1, 2, 3, 4, 5}:
            errors.append(
                f"{relative(DAY13_FOLD_METRICS)}: model {model!r} must contain "
                f"folds 1–5; found {sorted(folds)}."
            )

    summary_columns, summary_rows = read_csv_artifact(DAY13_MODEL_SUMMARY, errors)
    required_summary_columns = {
        "model",
        "mae_mean",
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_max",
        "r2_mean",
        "n_folds",
    }
    missing_summary_columns = required_summary_columns - summary_columns
    if missing_summary_columns:
        errors.append(
            f"{relative(DAY13_MODEL_SUMMARY)}: missing columns "
            f"{sorted(missing_summary_columns)}."
        )
    check_summary_rows(
        DAY13_MODEL_SUMMARY,
        summary_rows,
        (
            "mae_mean",
            "rmse_mean",
            "rmse_std",
            "rmse_min",
            "rmse_max",
            "r2_mean",
        ),
        errors,
    )

    stage_columns, stage_rows = read_csv_artifact(DAY14_STAGE_SUMMARY, errors)
    required_stage_columns = {
        "model",
        "mae_mean",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "n_folds",
        "source_file",
        "protocol",
    }
    missing_stage_columns = required_stage_columns - stage_columns
    if missing_stage_columns:
        errors.append(
            f"{relative(DAY14_STAGE_SUMMARY)}: missing columns "
            f"{sorted(missing_stage_columns)}."
        )
    check_summary_rows(
        DAY14_STAGE_SUMMARY,
        stage_rows,
        (
            "mae_mean",
            "rmse_mean",
            "rmse_std",
            "r2_mean",
        ),
        errors,
    )
    expected_source = relative(DAY13_FOLD_METRICS)
    expected_protocol = "deterministic_synthetic_random_kfold5_tutorial"
    for row_number, row in enumerate(stage_rows, start=2):
        if row.get("source_file") != expected_source:
            errors.append(
                f"{relative(DAY14_STAGE_SUMMARY)}:{row_number}: source_file "
                f"must be {expected_source!r}."
            )
        if row.get("protocol") != expected_protocol:
            errors.append(
                f"{relative(DAY14_STAGE_SUMMARY)}:{row_number}: protocol must "
                f"be {expected_protocol!r}."
            )
    if not DAY14_REPORT_PREVIEW.is_file():
        errors.append(
            f"Missing Day 14 tutorial report: {relative(DAY14_REPORT_PREVIEW)}."
        )
    else:
        report = DAY14_REPORT_PREVIEW.read_text(encoding="utf-8")
        report_markers = (
            "人工教程",
            "不代表 ESOL",
            "下游任务",
            expected_source,
            "计时：只保留在 Day 13 运行时内存变量中，不进入预存展示或 CSV fixture",
        )
        for marker in report_markers:
            if marker not in report:
                errors.append(
                    f"{relative(DAY14_REPORT_PREVIEW)}: missing report marker "
                    f"{marker!r}."
                )


def read_csv_artifact(
    path: Path,
    errors: list[str],
) -> tuple[set[str], list[dict[str, str]]]:
    """Read one required CSV fixture and return its columns and rows."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        errors.append(f"{relative(path)}: cannot read CSV fixture: {exc}")
        return set(), []
    return columns, rows


def check_finite_csv_value(
    path: Path,
    row_number: int,
    column: str,
    raw_value: object,
    errors: list[str],
) -> None:
    """Report a non-finite numeric CSV value."""

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        errors.append(
            f"{relative(path)}:{row_number}: {column} must be numeric; "
            f"found {raw_value!r}."
        )
        return
    if not math.isfinite(value):
        errors.append(
            f"{relative(path)}:{row_number}: {column} must be finite; "
            f"found {raw_value!r}."
        )


def check_summary_rows(
    path: Path,
    rows: list[dict[str, str]],
    numeric_columns: tuple[str, ...],
    errors: list[str],
) -> None:
    """Validate the shared five-model tutorial summary shape."""

    if len(rows) != 5:
        errors.append(
            f"{relative(path)}: expected 5 model summary rows; found {len(rows)}."
        )
    models = {row.get("model", "") for row in rows}
    if models != EXPECTED_TUTORIAL_MODELS:
        errors.append(
            f"{relative(path)}: expected models "
            f"{sorted(EXPECTED_TUTORIAL_MODELS)}; found {sorted(models)}."
        )
    for row_number, row in enumerate(rows, start=2):
        try:
            n_folds = int(row.get("n_folds", ""))
        except (TypeError, ValueError):
            n_folds = None
        if n_folds != 5:
            errors.append(
                f"{relative(path)}:{row_number}: n_folds must be 5; "
                f"found {row.get('n_folds')!r}."
            )
        for column in numeric_columns:
            check_finite_csv_value(
                path,
                row_number,
                column,
                row.get(column),
                errors,
            )


def check_markdown_files(errors: list[str]) -> None:
    """Check local links, machine-specific paths, and control characters."""

    for markdown in REPO_ROOT.rglob("*.md"):
        if is_generated_or_internal(markdown):
            continue
        text = markdown.read_text(encoding="utf-8")
        if "/Users/" in text or "file://" in text:
            errors.append(f"{relative(markdown)}: contains a machine-specific path.")

        bad_controls = sorted(
            {ord(character) for character in text}
            - {9, 10, 13}
        )
        bad_controls = [code for code in bad_controls if code < 32]
        if bad_controls:
            errors.append(
                f"{relative(markdown)}: contains ASCII control characters "
                f"{bad_controls}."
            )

        for raw_target in LINK_PATTERN.findall(text):
            target = normalized_link_target(markdown, raw_target)
            if target is not None and not target.exists():
                errors.append(
                    f"{relative(markdown)}: broken local link '{raw_target}' "
                    f"-> '{target}'."
                )


def check_python_fences(errors: list[str]) -> int:
    """Compile executable curriculum Python fences without executing them."""

    fence_count = 0
    markdown_files = set(CURRICULUM_ROOT.rglob("README.md"))
    beginner_root = CORE_ROOT / "day01_beginner"
    markdown_files.update(beginner_root.glob("[0-9][0-9]_*.md"))
    markdown_files.add(beginner_root / "exercises.md")
    for route_root in (CORE_ROOT, OPTIONAL_GNN_ROOT):
        for day_dir in route_root.glob("day??_*"):
            match = DAY_PATTERN.fullmatch(day_dir.name)
            if match is None or int(match.group(1)) == 1:
                continue
            for name in LEARNING_PACKAGE_FILES:
                if name.endswith(".md"):
                    markdown_files.add(day_dir / name)
    for unit_dir in ACTIVE_LEARNING_ROOT.glob("unit??_*"):
        if UNIT_PATTERN.fullmatch(unit_dir.name) is None:
            continue
        for name in LEARNING_PACKAGE_FILES:
            if name.endswith(".md"):
                markdown_files.add(unit_dir / name)

    for markdown in sorted(markdown_files):
        if not markdown.is_file():
            continue
        text = markdown.read_text(encoding="utf-8")
        for index, code in enumerate(PYTHON_FENCE_PATTERN.findall(text), start=1):
            fence_count += 1
            try:
                compile(code, f"{relative(markdown)}:python-fence-{index}", "exec")
            except SyntaxError as exc:
                errors.append(
                    f"{relative(markdown)}: Python fence {index} "
                    f"has invalid syntax: {exc}"
                )
    return fence_count


def check_json_files(errors: list[str]) -> None:
    """Parse tracked-style JSON files outside generated directories."""

    for path in REPO_ROOT.rglob("*.json"):
        if is_generated_or_internal(path):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{relative(path)}: invalid JSON: {exc}")


def check_esol_notebook(errors: list[str]) -> int:
    """Validate the saved reference Notebook and ensure it has no error output."""

    if not ESOL_NOTEBOOK.is_file():
        errors.append(f"Missing ESOL Notebook: {relative(ESOL_NOTEBOOK)}")
        return 0

    try:
        notebook = json.loads(ESOL_NOTEBOOK.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{relative(ESOL_NOTEBOOK)}: invalid Notebook JSON: {exc}")
        return 0

    markdown_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )
    if "课程 Day 7" not in markdown_text:
        errors.append(
            f"{relative(ESOL_NOTEBOOK)}: must identify itself as the course "
            "Day 7 integrated reference."
        )
    if "Day 2 在不改变数据划分和指标口径" in markdown_text:
        errors.append(
            f"{relative(ESOL_NOTEBOOK)}: contains the legacy Day 2 next-step "
            "instruction."
        )

    code_cells = [
        cell for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    expected_counts = list(range(1, len(code_cells) + 1))
    if execution_counts != expected_counts:
        errors.append(
            f"{relative(ESOL_NOTEBOOK)}: expected consecutive execution counts "
            f"{expected_counts}; found {execution_counts}."
        )

    for cell_index, cell in enumerate(code_cells, start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    f"{relative(ESOL_NOTEBOOK)}: code cell {cell_index} "
                    f"contains saved error {output.get('ename')}: "
                    f"{output.get('evalue')}"
                )
    return len(code_cells)


def check_esol_results(errors: list[str]) -> None:
    """Validate the Day 1 result table and reproducibility configuration."""

    required_columns = {
        "model",
        "split",
        "n_samples",
        "seed",
        "mae_logS",
        "rmse_logS",
        "r2",
        "source_commit",
    }
    try:
        with ESOL_METRICS.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        errors.append(f"{relative(ESOL_METRICS)}: cannot read CSV: {exc}")
        return

    missing_columns = required_columns - columns
    if missing_columns:
        errors.append(
            f"{relative(ESOL_METRICS)}: missing columns "
            f"{sorted(missing_columns)}."
        )
    if len(rows) != 10:
        errors.append(
            f"{relative(ESOL_METRICS)}: expected 10 model/split rows; "
            f"found {len(rows)}."
        )

    try:
        config = json.loads(ESOL_CONFIG.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{relative(ESOL_CONFIG)}: cannot read configuration: {exc}")
        return

    required_config = {
        "best_model_by_validation_rmse",
        "branch",
        "model_params",
        "source_commit",
        "split_sizes",
        "working_tree_dirty_during_run",
    }
    missing_config = required_config - set(config)
    if missing_config:
        errors.append(
            f"{relative(ESOL_CONFIG)}: missing keys {sorted(missing_config)}."
        )
    if config.get("working_tree_dirty_during_run") is not False:
        errors.append(
            f"{relative(ESOL_CONFIG)}: saved run must come from a clean worktree."
        )

    valid_rows = [row for row in rows if row.get("split") == "valid"]
    try:
        best_row = min(valid_rows, key=lambda row: float(row["rmse_logS"]))
    except (ValueError, KeyError):
        errors.append(
            f"{relative(ESOL_METRICS)}: validation RMSE values are invalid."
        )
        return
    if best_row.get("model") != config.get("best_model_by_validation_rmse"):
        errors.append(
            "ESOL best-model mismatch between baseline_metrics.csv "
            "and run_config.json."
        )


def main() -> int:
    errors: list[str] = []
    check_repository_layout(errors)
    check_public_guides(errors)
    core_days = collect_day_directories(CORE_ROOT, range(1, 29), errors)
    gnn_days = collect_day_directories(OPTIONAL_GNN_ROOT, range(29, 36), errors)
    day_dirs = core_days + gnn_days
    unit_dirs = collect_unit_directories(errors)

    check_task_card_sections(day_dirs, errors)
    tutorial_count, tutorial_code_cells = check_complete_learning_packages(
        day_dirs, errors
    )
    check_unit_task_card_sections(unit_dirs, errors)
    unit_tutorial_count, unit_code_cells = check_complete_unit_packages(
        unit_dirs, errors
    )
    check_curriculum_routes(core_days, gnn_days, errors)
    check_active_learning_routes(unit_dirs, errors)
    check_markdown_files(errors)
    fence_count = check_python_fences(errors)
    check_json_files(errors)
    code_cell_count = check_esol_notebook(errors)
    check_esol_results(errors)
    check_learning_artifact_contract(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Repository validation passed: "
        f"{len(core_days)} core days, {len(gnn_days)} optional GNN days, "
        f"{len(unit_dirs)} active-learning Units, {fence_count} Python fences, "
        f"{tutorial_count} Day notebooks with {tutorial_code_cells} code cells, "
        f"{unit_tutorial_count} Unit notebooks with {unit_code_cells} code "
        f"cells, {code_cell_count} ESOL reference code cells, retained ESOL "
        "artifacts, public guides, and tutorial outputs checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
