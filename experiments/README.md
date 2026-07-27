# 实验目录

`experiments/` 只保存真正运行过或正在执行的实验；课程任务卡留在 `curriculum/`。

| 实验 | 目标 | 状态 |
|---|---|---|
| [ESOL 参考基线 E01](esol/day01_baseline/README.md) | 复现 ECFP＋传统机器学习回归流程；课程 Day 7 使用 | 已运行，可复现 |

Day 2–28 的完整教学材料和可运行 Notebook 位于 `curriculum/core/`，它们是教材，不代表本人完成。开始后续某一天时运行：

```bash
python scripts/start_day.py 2
```

脚本只创建当前一天的个人实验目录，并复制对应教学 Notebook；复制时会清除教材中的预存输出，已有个人文件不会被覆盖。详细区别和运行方法见[完整 Day 学习包使用方法](../curriculum/shared/day_package_guide.md)。
