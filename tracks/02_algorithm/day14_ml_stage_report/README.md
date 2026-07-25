# Day 14：传统机器学习阶段报告

> 状态：待学习。阶段报告必须基于真实运行记录，不能照抄本任务说明。

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
2. 一份阶段报告 Markdown；
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
2. 找到每次运行的指标文件和配置文件。
3. 检查所有结果是否使用相同目标空间和指标单位。
4. 删除或隔离协议不一致、来源不明的行。
5. 汇总模型、划分、种子、RMSE、MAE、R² 和耗时。
6. 标注每个数字是单次结果还是交叉验证汇总。
7. 写三条有数字支持的观察。
8. 写至少三条不能得出的结论。
9. 写进入 MLP 前还需要补齐的知识。
10. 让别人仅凭报告找到原始证据，但不上传无权限数据。

## 核心代码骨架

先把你实际存在的 CSV 路径填入列表；不要为了让代码通过而创建虚假文件。

```python
from pathlib import Path
import pandas as pd

metric_paths = [
    Path("请替换为你实际生成的结果.csv"),
]

required_columns = {
    "model", "split", "rmse", "mae", "r2"
}
tables = []

for path in metric_paths:
    if not path.exists():
        raise FileNotFoundError(f"找不到真实结果文件：{path}")
    table = pd.read_csv(path)
    table = table.rename(columns={
        "mae_logS": "mae", "rmse_logS": "rmse"
    })
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"{path} 缺少列：{sorted(missing)}")
    table["source_file"] = str(path)
    tables.append(table)

all_results = pd.concat(tables, ignore_index=True)
valid_rows = all_results.loc[
    all_results["split"].isin(["valid", "cv_valid"])
].copy()

summary = (
    valid_rows.groupby("model")[["rmse", "mae", "r2"]]
    .agg(["mean", "std", "count"])
    .sort_values(("rmse", "mean"))
)

summary.to_csv("ml_stage_summary.csv", encoding="utf-8")
print(summary)
```

## 只解释今天新增的语法

- `required_columns - set(table.columns)` 是集合差，得到缺失列名。
- `raise FileNotFoundError(...)` 主动停止，而不是静默跳过缺失证据。
- `pd.concat(..., ignore_index=True)` 把结构一致的表按行合并。
- `.isin([...])` 判断一列的值是否属于允许集合。
- `.agg(["mean", "std", "count"])` 同时计算均值、标准差和样本数。

代码中的占位路径故意不能直接使用，它提醒你只汇总真实产物。

## 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 把不同标签尺度的 RMSE 合并 | 数字不可比较 | 分开报告并写明单位 |
| 只粘贴最好分数 | 丢失稳定性证据 | 保留每折和汇总 |
| 没运行却写“已完成” | 研究记录失真 | 标记待完成 |
| 把 ESOL 写成粘合剂数据 | 误导导师 | 明确公开练习边界 |
| 报告没有源文件位置 | 无法复查 | 保存来源与配置 |

## 完成标准

- [ ] 报告中的每个数字都能追溯到真实文件；
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
- 下一天：[Day 15：用 NumPy 理解张量与形状](../day15_tensor_shape/README.md)
