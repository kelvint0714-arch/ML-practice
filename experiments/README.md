# 实验目录

`experiments/` 只保存真正运行过或正在执行的实验；课程任务卡留在 `curriculum/`。

| 实验 | 目标 | 状态 |
|---|---|---|
| [ESOL 参考基线 E01](esol/day01_baseline/README.md) | 复现 ECFP＋传统机器学习回归流程；课程 Day 7 使用 | 已运行，可复现 |

Day 2–28 的核心材料位于 `curriculum/core/`，Day 29–35 的可选 GNN 材料位于 `curriculum/optional_gnn/`，主动学习 Unit 1–9 位于 `curriculum/active_learning/`。这些都是教材，不代表本人完成。开始某一天时运行：

```bash
python scripts/start_day.py 2
```

开始主动学习 Unit：

```bash
python scripts/start_unit.py 1
```

两个脚本都只创建当前模块的个人实验目录，并复制对应教学 Notebook；复制时会清除教材中的预存输出，已有个人文件不会被覆盖。详细区别和运行方法见[完整 Day/Unit 学习包使用方法](../curriculum/shared/day_package_guide.md)。
