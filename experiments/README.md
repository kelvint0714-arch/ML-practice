# 实验目录

`experiments/` 现在同时保存两类内容：

1. 已运行的参考实验；
2. Day 02–35 与主动学习 Unit 01–09 的完整起始工作区。

先打开[实验工作区完整索引](INDEX.md)。索引已经列出全部 34 个 Day
工作区和 9 个主动学习 Unit 工作区，每项都能直接进入 README 或
Notebook。

| 范围 | 数量 | 当前含义 |
|---|---:|---|
| [ESOL 参考基线 E01](esol/day01_baseline/README.md) | 1 | 已运行参考制品 |
| Day 02–28 核心路线 | 27 | 完整代码已准备，待本人运行 |
| Day 29–35 可选 GNN | 7 | 完整代码已准备，待本人运行 |
| 主动学习 Unit 01–09 | 9 | 完整代码已准备，待本人运行 |

## “补齐”与“完成”的区别

每个新工作区都包含：

```text
README.md
dayXX_topic.ipynb 或 unitXX_topic.ipynb
notes.md
results/README.md
```

Notebook 不是空壳：代码与对应课程教程完整一致，但教师预存的输出已经
清除，并标记为 `workspace_status=not_started`。所以这次补齐解决的是
“后续没有实验文件”的问题，不会把尚未亲自运行的内容冒充为已完成成果。

真正开始时，按 [`curriculum/PROGRESS.md`](../curriculum/PROGRESS.md)
的第一个未完成项依次学习。运行后把解释写进 `notes.md`，把本人生成的
CSV、JSON 或图片放入 `results/`。

开始和完成时分别更新工作区状态：

```bash
python scripts/set_workspace_status.py day02 in_progress
python scripts/set_workspace_status.py day02 completed
```

`completed` 会检查 Notebook 是否全部执行且 `notes.md` 是否已经填写，
但不会自动勾选学习进度。

## 维护命令

缺少某一个工作区时仍可单独建立：

```bash
python scripts/start_day.py 2
python scripts/start_unit.py 1
```

维护者需要一次检查并补建全部工作区时运行：

```bash
python scripts/bootstrap_experiment_workspaces.py
```

脚本不会覆盖已执行的 Notebook 或本人笔记。详细运行方法和环境边界见
[完整 Day/Unit 学习包使用方法](../curriculum/shared/day_package_guide.md)。
