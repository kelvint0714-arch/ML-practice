# Day 14：传统机器学习阶段报告

> 状态：待学习。阶段报告必须基于真实运行记录，不能照抄本任务说明。

## 学习文件导航

按顺序完成以下五个课程文件：

1. [概念讲义](01_concepts.md)
2. [算法推演](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习题](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 审计 Day 13 已实际运行的课程教程 CSV，并生成明确标注为人工数据的报告预览。它不能代替学习者基于本人 `experiments/` 证据撰写的阶段报告。

## 今天为什么学

跑出很多数字不等于完成研究。
导师真正需要看到的是：问题是什么、数据怎样处理、比较是否公平、结果支持什么结论、还缺什么证据。

今天不增加新模型，而是审计 Day 1–Day 13 的记录，形成一份可复查的传统机器学习阶段报告。

## 前置条件

- 至少理解 Dummy、Ridge、决策树、随机森林和 Boosting；
- 能解释 Pipeline、交叉验证和数据泄漏；
- 知道均值与标准差比单次最好分数更可靠；
- 保留了自己实际运行产生的配置、结果或学习记录。

## 今日产出

完成后应当产生：

1. 一张统一模型结果表；
2. 一份 `experiments/day14_ml_stage_report/report.md` 阶段报告；
3. 一页“可以得出 / 不能得出”的结论边界；
4. 下一阶段学习 MLP 的明确入口；
5. 一份缺失证据清单。

如果前面某天尚未实际运行，应在报告中写“未完成”，不能补造分数。

## 核心概念

### 1. 证据、解释和决定

- 证据：保存的参数、数据划分、每折分数、运行环境；
- 解释：为什么出现过拟合、波动或模型差异；
- 决定：下一步继续验证什么；
- 边界：当前证据不能支持哪些说法。

这四类内容应分开写，避免把猜测包装成结果。

### 2. 报告最小结构

```text
研究问题
→ 数据与目标
→ 划分和防泄漏协议
→ 候选模型与参数
→ 结果表
→ 观察与局限
→ 下一步
```

### 3. 当前项目边界

ESOL 的目标是分子水溶解度，不是粘合剂强度。
公开数据练习能够证明算法流程是否跑通，不能证明已经完成真实粘合剂预测。

## 分步骤任务

1. 列出 Day 1–Day 13 哪些已经亲自完成，哪些只是阅读。
2. 从 Day 13 的 `fold_metrics.csv` 找到公平比较的逐折指标，并把其他协议的结果作为单独证据引用。
3. 检查所有结果是否使用相同目标空间和指标单位。
4. 删除或隔离协议不一致、来源不明的行。
5. 汇总模型、划分、种子、RMSE、MAE、R² 和折数。
6. 标注每个数字是单次结果还是交叉验证汇总。
7. 写三条有数字支持的观察。
8. 写至少三条不能得出的结论。
9. 写进入 MLP 前还需要补齐的知识。
10. 让别人仅凭报告找到原始证据，但不上传无权限数据。

## 核心代码骨架

下面直接读取 Day 13 规定的真实产物。文件不存在时应先返回 Day 13，而不是创建虚假结果。

```python
from pathlib import Path
import pandas as pd

# 从仓库根目录运行这段代码。
fold_metrics_path = Path(
    "experiments/day13_fair_comparison/results/fold_metrics.csv"
)
results_dir = Path("experiments/day14_ml_stage_report/results")
report_path = Path("experiments/day14_ml_stage_report/report.md")
results_dir.mkdir(parents=True, exist_ok=True)

if not fold_metrics_path.exists():
    raise FileNotFoundError(
        f"请先完成 Day 13：{fold_metrics_path}"
    )

fold_metrics = pd.read_csv(fold_metrics_path)
required_columns = {
    "model", "fold", "split", "mae", "rmse", "r2"
}
missing = required_columns - set(fold_metrics.columns)

if missing:
    raise ValueError(
        f"{fold_metrics_path} 缺少列：{sorted(missing)}"
    )

if set(fold_metrics["split"]) != {"cv_valid"}:
    raise ValueError("Day 13 文件中出现了不一致的 split。")

if fold_metrics.duplicated(["model", "fold"]).any():
    raise ValueError("Day 13 文件中存在重复的 model/fold。")

expected_models = {
    "dummy", "ridge", "decision_tree",
    "random_forest", "gradient_boosting",
}
if len(fold_metrics) != 25:
    raise ValueError("Day 13 文件必须恰好包含 5 个模型 × 5 折 = 25 行。")
if set(fold_metrics["model"]) != expected_models:
    raise ValueError("Day 13 文件的模型集合不完整或出现额外模型。")

expected_folds = {1, 2, 3, 4, 5}
if set(fold_metrics["fold"]) != expected_folds:
    raise ValueError("Day 13 文件的折编号不是 1–5。")

fold_counts = fold_metrics.groupby("model")["fold"].nunique()
if not (fold_counts == 5).all():
    raise ValueError("至少一个模型没有完整的5折结果。")

summary = (
    fold_metrics.groupby("model")
    .agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        n_folds=("fold", "count"),
    )
    .reset_index()
    .sort_values("rmse_mean")
)

summary.to_csv(
    results_dir / "ml_stage_summary.csv",
    encoding="utf-8",
    index=False,
)
print(summary)
print("请把阶段报告写到：", report_path)
```

## 只解释今天新增的语法

- `required_columns - set(fold_metrics.columns)` 是集合差，得到缺失列名。
- `raise FileNotFoundError(...)` 主动停止，而不是静默跳过缺失证据。
- `set(fold_metrics["split"])` 检查文件中实际出现的划分名称。
- `.duplicated(["model", "fold"])` 检查同一个模型、同一折是否重复；
- `.nunique()` 检查每个模型是否恰好具有5个不同折；
- `.agg(...)` 同时计算均值、标准差和折数。

Day 1 的固定验证结果与 Day 13 的交叉验证结果采用不同协议，不应直接混成同一个平均值。报告中可以并列说明，但主比较表以 Day 13 的逐折文件为准。

## 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 把不同标签尺度的 RMSE 合并 | 数字不可比较 | 分开报告并写明单位 |
| 只粘贴最好分数 | 丢失稳定性证据 | 保留每折和汇总 |
| 没运行却写“已完成” | 研究记录失真 | 标记待完成 |
| 把 ESOL 写成粘合剂数据 | 误导导师 | 明确公开练习边界 |
| 报告没有源文件位置 | 无法复查 | 保存来源与配置 |

## 完成标准

- [ ] `experiments/day14_ml_stage_report/report.md` 中的每个数字都能追溯到真实文件；
- [ ] 我区分了单次分数和均值±标准差；
- [ ] 我写明数据、目标、单位和划分；
- [ ] 我没有把 ESOL 结果称为粘合剂结果；
- [ ] 我列出了尚未完成和仍不确定的内容；
- [ ] 下一阶段为什么学习 MLP 已经写清楚。

## 自测问题

1. 为什么阶段报告不应该只展示排行榜？
2. 哪些信息能让别人复现实验？
3. 一条合理的“不能得出”结论是什么？
4. 如果两个 CSV 的 RMSE 单位不同，可以直接求平均吗？
5. 尚未运行的实验应该怎样写进报告？

## 上一天 / 下一天

- 上一天：[Day 13：公平比较多个传统模型](../day13_fair_comparison/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 15：用 NumPy 理解张量与形状](../day15_tensor_shape/README.md)
