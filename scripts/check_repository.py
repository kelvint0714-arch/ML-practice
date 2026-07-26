"""Validate the repository without running machine-learning training.

The checker covers repository layout, curriculum continuity, Markdown links,
Python code fences, saved Notebook state, ESOL result artifacts, and the
tracked adhesive workbook package. It never marks a learning day complete.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_ROOT = REPO_ROOT / "curriculum"
CORE_ROOT = CURRICULUM_ROOT / "core"
OPTIONAL_GNN_ROOT = CURRICULUM_ROOT / "optional_gnn"
DAY_PATTERN = re.compile(r"day(\d{2})_[^/]+$")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PYTHON_FENCE_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)

REQUIRED_ROOT_FILES = (
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "requirements-learning.txt",
    "requirements.txt",
)
REQUIRED_TOP_LEVEL = (
    ".github",
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

ESOL_ROOT = REPO_ROOT / "experiments" / "esol" / "day01_baseline"
ESOL_NOTEBOOK = ESOL_ROOT / "esol_baseline.ipynb"
ESOL_METRICS = ESOL_ROOT / "results" / "baseline_metrics.csv"
ESOL_CONFIG = ESOL_ROOT / "results" / "run_config.json"
ADHESIVE_WORKBOOK = (
    REPO_ROOT
    / "data"
    / "adhesive"
    / "templates"
    / "粘合剂重要化学性质_数据格式_v0.3.xlsx"
)
EXPECTED_WORKSHEETS = (
    "01_核心化学性质",
    "02_字段说明",
    "03_公开依据",
)
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

    route_files = (
        CURRICULUM_ROOT / "PROGRESS.md",
        CORE_ROOT / "README.md",
    )
    expected_readmes = {
        path / "README.md" for path in expected_core_paths + expected_gnn_paths
    }
    for route_file in route_files:
        if not route_file.is_file():
            errors.append(f"Missing curriculum route file: {relative(route_file)}.")
            continue
        targets = local_link_targets(route_file)
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


def check_learning_artifact_contract(errors: list[str]) -> None:
    """Keep the Day 13 output and Day 14 input contract synchronized."""

    day13_path = CORE_ROOT / "day13_fair_comparison" / "README.md"
    day14_path = CORE_ROOT / "day14_ml_stage_report" / "README.md"
    if not day13_path.is_file() or not day14_path.is_file():
        errors.append("Day 13/14 task cards are missing; cannot check artifact contract.")
        return

    day13 = day13_path.read_text(encoding="utf-8")
    day14 = day14_path.read_text(encoding="utf-8")
    shared_markers = (
        "experiments/day13_fair_comparison/results",
        "fold_metrics.csv",
        '"model"',
        '"fold"',
        '"split"',
        '"mae"',
        '"rmse"',
        '"r2"',
        '"fit_seconds"',
    )
    for marker in shared_markers:
        if marker not in day13 or marker not in day14:
            errors.append(
                "Day 13/14 artifact contract is inconsistent; missing shared "
                f"marker {marker!r}."
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


def check_adhesive_workbook(errors: list[str]) -> None:
    """Validate the XLSX package and reject hidden author/software metadata."""

    required_members = {
        "[Content_Types].xml",
        "_rels/.rels",
        "xl/workbook.xml",
        "xl/styles.xml",
    }
    try:
        with zipfile.ZipFile(ADHESIVE_WORKBOOK) as workbook:
            names = set(workbook.namelist())
            missing = required_members - names
            if missing:
                errors.append(
                    f"{relative(ADHESIVE_WORKBOOK)}: missing XLSX members "
                    f"{sorted(missing)}."
                )
            corrupt_member = workbook.testzip()
            if corrupt_member is not None:
                errors.append(
                    f"{relative(ADHESIVE_WORKBOOK)}: corrupt member "
                    f"{corrupt_member}."
                )

            workbook_xml = workbook.read("xl/workbook.xml").decode(
                "utf-8", errors="replace"
            )
            for sheet_name in EXPECTED_WORKSHEETS:
                if sheet_name not in workbook_xml:
                    errors.append(
                        f"{relative(ADHESIVE_WORKBOOK)}: missing worksheet "
                        f"'{sheet_name}'."
                    )

            forbidden_metadata = (
                "lastModifiedBy",
                "dc:creator",
                "KSOProductBuildVer",
                'name="ICV"',
            )
            for name in names:
                if not name.startswith("docProps/") or not name.endswith(".xml"):
                    continue
                metadata = workbook.read(name).decode("utf-8", errors="replace")
                for marker in forbidden_metadata:
                    if marker in metadata:
                        errors.append(
                            f"{relative(ADHESIVE_WORKBOOK)}: hidden metadata "
                            f"'{marker}' remains in {name}."
                        )
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        errors.append(f"{relative(ADHESIVE_WORKBOOK)}: invalid XLSX package: {exc}")


def main() -> int:
    errors: list[str] = []
    check_repository_layout(errors)
    core_days = collect_day_directories(CORE_ROOT, range(1, 29), errors)
    gnn_days = collect_day_directories(OPTIONAL_GNN_ROOT, range(29, 36), errors)
    day_dirs = core_days + gnn_days

    check_task_card_sections(day_dirs, errors)
    check_curriculum_routes(core_days, gnn_days, errors)
    check_markdown_files(errors)
    fence_count = check_python_fences(errors)
    check_json_files(errors)
    code_cell_count = check_esol_notebook(errors)
    check_esol_results(errors)
    check_adhesive_workbook(errors)
    check_learning_artifact_contract(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Repository validation passed: "
        f"{len(core_days)} core days, {len(gnn_days)} optional GNN days, "
        f"{fence_count} Python fences, {code_cell_count} executed Notebook "
        "code cells, ESOL artifacts, and the adhesive workbook checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
