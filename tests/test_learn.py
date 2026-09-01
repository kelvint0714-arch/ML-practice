"""Tests for the read-only learner helper."""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("learn", REPO_ROOT / "scripts" / "learn.py")
assert SPEC is not None and SPEC.loader is not None
learn = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = learn
SPEC.loader.exec_module(learn)


class ProgressTests(unittest.TestCase):
    def test_repository_progress_has_expected_routes(self) -> None:
        items = learn.progress_items()

        self.assertEqual(len(items), 49)
        self.assertEqual(sum(item.route == "core" for item in items), 33)
        self.assertEqual(sum(item.route == "active-learning" for item in items), 9)
        self.assertEqual(sum(item.route == "gnn" for item in items), 7)

    def test_status_reports_next_item_without_editing_progress(self) -> None:
        before = learn.PROGRESS_FILE.read_text(encoding="utf-8")
        output = io.StringIO()

        with redirect_stdout(output):
            result = learn.print_progress("all")

        self.assertEqual(result, 0)
        completed = sum(item.completed for item in learn.progress_items())
        self.assertIn(f"已完成：{completed}/49", output.getvalue())
        self.assertIn("下一项：", output.getvalue())
        self.assertEqual(before, learn.PROGRESS_FILE.read_text(encoding="utf-8"))

    def test_parser_accepts_common_completion_marks(self) -> None:
        items = learn.parse_progress(
            "- [x] **Day 2：**完成 [任务](core/day02/README.md)\n"
            "- [ ] **Unit 01：**完成 [任务](active/unit01/README.md)\n"
        )

        self.assertTrue(items[0].completed)
        self.assertFalse(items[1].completed)
        self.assertEqual(items[1].route, "active-learning")


class RequirementTests(unittest.TestCase):
    def test_active_learning_requirements_follow_includes(self) -> None:
        pins = learn.load_pinned_requirements(
            REPO_ROOT / "requirements-active-learning.txt"
        )

        self.assertEqual(pins["deepchem"], "2.8.0")
        self.assertEqual(pins["matplotlib"], "3.10.9")
        self.assertEqual(pins["nbclient"], "0.11.0")


if __name__ == "__main__":
    unittest.main()
