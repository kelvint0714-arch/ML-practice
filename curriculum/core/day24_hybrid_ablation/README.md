# Day 24：用消融实验判断“结合”是否真的有用

> 状态：待学习、待运行。所有结果位置必须由实际运行填写。

## 本日完整学习包

1. [概念精讲](01_concepts.md)
2. [算法走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 是课程附带的消融演示，不是学习者运行记录；其中 ESOL 输出不能写成下游任务结论。

## 今天为什么学

混合模型分数提高时，我们还不知道收益来自哪个部分。

收益可能来自 MLP、简单平均、第二层模型或原始特征直通。

消融实验通过一次只删除或改变一个组件，
帮助判断复杂结构是否有必要。

## 前置条件

- 完成 [Day 23](../day23_oof_stacking/README.md)；
- 已保存 Day 07 冻结随机森林、统一 MLP、平均和 scaffold-aware stacking 的定义；
- 使用同一个数据划分和评价指标；
- 能解释控制变量；
- 测试集仍未用于结构选择。

## 今日产出

1. 一份运行前固定的消融矩阵；
2. 每个方案的 RMSE、MAE、`R²` 和耗时；
3. 相对最强单模型的差值；
4. 一段复杂度—收益判断；
5. 一个冻结的候选方案，或明确决定保留单模型。

## 核心概念

### 1. 什么是消融

消融不是随机试很多参数。
它从完整方案出发，
每次删除一个组件或替换一个清楚的机制。

### 2. 对照必须可解释

推荐的最小对照包括：

- 传统模型单独；
- MLP 单独；
- 两者简单平均；
- 只有树作为一级模型、但仍保留 OOF + Ridge 二层的 `stack_tree_only`；
- 树和 MLP 共同作为一级模型的 `stack_tree_mlp`。

`tree_only` 与 `stack_tree_only` 必须是真正不同的拟合过程：
后者需要从 OOF 树预测训练二层 Ridge，不能把树预测复制一遍改名。

### 3. 一次只改一项

如果同时改变特征、划分、参数和模型，
即使分数变化也无法归因。

### 4. 负结果同样重要

如果 stacking 没有稳定超过简单平均，
就没有必要为了“看起来高级”保留它。

## 分步骤任务

1. 冻结 Day 23 的完整方案。
2. 在运行前列出所有消融行。
3. 为每一行写出“唯一变化项”。
4. 复用相同训练和验证索引。
5. 复用相同评价函数。
6. 不临时改变基础模型参数。
7. 运行传统模型单独方案。
8. 运行 MLP 单独方案。
9. 运行简单平均。
10. 运行 `stack_tree_only`。
11. 运行 `stack_tree_mlp`，以“加入 MLP”作为唯一变化。
12. 记录训练耗时和模型数量。
13. 计算相对基准 RMSE 差值。
14. 用多个种子核对方向是否稳定。
15. 写出保留或放弃混合结构的理由。

## 核心代码骨架

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error

predictions = {
    "tree_only": tree.predict(X_valid),
    "mlp_only": mlp.predict(X_valid),
}
predictions["simple_mean"] = (
    predictions["tree_only"] + predictions["mlp_only"]
) / 2
predictions["stack_tree_only"] = stack_tree_only.predict(X_valid)
predictions["stack_tree_mlp"] = stack_tree_mlp.predict(X_valid)

rows = []
baseline_rmse = root_mean_squared_error(
    y_valid, predictions["tree_only"]
)
for name, prediction in predictions.items():
    rmse = root_mean_squared_error(y_valid, prediction)
    rows.append({
        "variant": name,
        "rmse": rmse,
        "delta_vs_tree": rmse - baseline_rmse,
        "mae": mean_absolute_error(y_valid, prediction),
        "r2": r2_score(y_valid, prediction),
    })

ablation = pd.DataFrame(rows).sort_values("rmse")
print(ablation)
```

## 只解释今天新增的语法

- `predictions` 字典把方案名称对应到预测数组。
- `predictions["simple_mean"] = ...` 向已有字典增加新项目。
- `delta_vs_tree` 是当前 RMSE 减去基准 RMSE。
- 差值小于零表示 RMSE 比基准低。
- `variant` 表示消融方案，不是新的数据集。
- 代码假设两个单模型和两个 stack 都用 Day 23 相同冻结配方、
  scaffold groups 与 GroupKFold split 列表拟合。

## 建议的消融记录表

| 方案 | 树模型 | MLP | 二层模型 | 原始特征直通 | 唯一变化 |
|---|---:|---:|---:|---:|---|
| 树单模型 | 是 | 否 | 否 | 否 | 基准 |
| MLP 单模型 | 否 | 是 | 否 | 否 | 模型类型 |
| 简单平均 | 是 | 是 | 否 | 否 | 固定等权 |
| stack_tree_only | 是 | 否 | 是 | 否 | 在树 OOF 预测上拟合二层 |
| stack_tree_mlp | 是 | 是 | 是 | 否 | 相对上一行只加入 MLP |

这两行都必须真实调用 `StackingRegressor.fit()`；
禁止用 `tree.predict()` 伪造 `stack_tree_only`。
是否保留完整结构，要同时看指标、种子稳定性、耗时、可解释性，
以及简单平均是否已达到近似效果。

## 下游任务的消融边界

真实数据到达后，还可能比较：

- 仅配方特征；
- 配方加固化条件；
- 再加入基材和表面处理；
- 再加入可获得的理化性质。

这些是特征组消融，
但前提是字段由领域团队确认且预测时确实可获得。
不能为了凑特征组从不同材料体系论文中拼接数值。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 每个消融使用不同划分 | 差异无法归因 | 固定索引 |
| 同时调参和删组件 | 多因素混杂 | 一次只改一项 |
| 只保留最好一行 | 失去消融证据 | 保存完整矩阵 |
| 只看一次种子 | 小样本偶然性 | 做稳定性核对 |
| 无改善仍强行结合 | 复杂度无收益 | 接受负结果 |

## 完成标准

- [ ] 消融矩阵在运行前写好；
- [ ] 每一行只有一个明确变化；
- [ ] 所有方案复用相同外部划分；
- [ ] 保存完整指标而非只留最好结果；
- [ ] 已比较简单平均与 stacking；
- [ ] `stack_tree_only` 不是 `tree_only` 的预测别名；
- [ ] 结论同时讨论收益、稳定性和复杂度；
- [ ] 未使用测试集选择消融方案。

## 自测问题

1. 消融实验与普通调参有什么区别？
2. 为什么简单平均是必要对照？
3. 同时换划分和模型会造成什么问题？
4. `delta_vs_tree < 0` 在本例中代表什么？
5. 没有改善时，最合理的结论是什么？

## 上一天 / 下一天

- 上一天：[Day 23：OOF Stacking](../day23_oof_stacking/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 25：启发式不确定性](../day25_uncertainty/README.md)
