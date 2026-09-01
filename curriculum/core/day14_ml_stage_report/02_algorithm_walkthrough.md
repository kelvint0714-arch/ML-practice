# Day 14 算法推演：审计、汇总、再写结论

## 1. 先定位源文件

教程读取 Day 13 随附 notebook 实际运行产生的：

```text
curriculum/core/day13_fair_comparison/tutorial_outputs/fold_metrics.csv
```

它是课程教程输出，不是学习者 `learning_outputs/` 证据。学习者正式报告应改为读取本人运行得到的：

```text
learning_outputs/day13_fair_comparison/results/fold_metrics.csv
```

教程不会在源文件缺失时生成随机替代分数。

## 2. 检查 schema

```python
required_columns = {
    "model", "fold", "split", "mae", "rmse", "r2"
}
missing = required_columns - set(fold_metrics.columns)
if missing:
    raise ValueError(f"缺少列：{sorted(missing)}")
```

集合差直接找出缺失列。

## 3. 检查行级完整性

```python
if fold_metrics.duplicated(["model", "fold"]).any():
    raise ValueError("存在重复 model/fold")

if set(fold_metrics["split"]) != {"cv_valid"}:
    raise ValueError("出现不一致 split")

expected_models = {
    "dummy", "ridge", "decision_tree",
    "random_forest", "gradient_boosting",
}
if len(fold_metrics) != 25:
    raise ValueError("必须恰好有 5 个模型 × 5 折 = 25 行")
if set(fold_metrics["model"]) != expected_models:
    raise ValueError("模型集合不完整或出现额外模型")

fold_counts = fold_metrics.groupby("model")["fold"].nunique()
if not (fold_counts == 5).all():
    raise ValueError("至少一个模型折数不完整")
```

还应检查：

```python
metric_columns = ["mae", "rmse", "r2"]
if not np.isfinite(fold_metrics[metric_columns]).all().all():
    raise ValueError("出现 NaN 或无穷")
```

## 4. 汇总但保留源文件

```python
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
```

汇总表应带来源说明，不能脱离逐折文件单独传播。

## 5. 只写输出支持的观察

安全做法是先从表中计算：

- 观察到的最低平均 RMSE 模型；
- Dummy 的平均 RMSE；
- 每个模型 RMSE 标准差；
- 复杂模型是否每折都优于 Dummy。

然后使用限定语言：

> 在当前教程人工数据和固定 5 折中，模型 A 的平均折内验证 RMSE 最低。

不要写：

> 模型 A 已证明是最好的材料算法。

## 6. 报告生成与人工审阅

notebook 可以生成一个基于真实教程输出的 `report_preview.md`，但它只是演示草稿。保存前应人工确认：

- 每个数字与表一致；
- 数据名称没有被错误替换；
- “可能原因”明确标为假设；
- 未完成项没有被代码自动填成完成；
- 输出路径没有冒充 `learning_outputs/`。

## 7. 缺失证据清单

示例：

```text
- 尚未在 ESOL ECFP 上运行本比较；
- 普通随机 KFold 不代表新骨架泛化；
- 未执行嵌套交叉验证；
- 未评估严格未见测试集；
- 尚未获得真实下游任务数据。
```

缺失项不是失败，而是报告可信度的一部分。

## 8. 可复查交付

正式学习者交付应至少包含：

```text
learning_outputs/day14_ml_stage_report/
├── README.md
├── notes.md
├── report.md
└── results/
    └── ml_stage_summary.csv
```

这些文件应由学习者实际运行和审阅后创建。课程只提供方法和教程，不预先代填完成状态。
