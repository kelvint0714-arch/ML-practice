# Day 11 概念讲义：Pipeline 与数据泄漏

## 学习目标

完成本日后，你应能：

- 识别会从数据中学习状态的预处理步骤；
- 解释为什么预处理也必须遵守训练边界；
- 使用 `Pipeline` 串联填补、缩放和模型；
- 区分 `fit`、`transform` 与 `predict`；
- 检查已拟合 Pipeline 内部保存的训练统计量。

## 1. 泄漏是什么

数据泄漏是指模型开发阶段获得了按协议本不应获得的信息。泄漏不一定直接把 `y_valid` 交给模型；让预处理器提前看到验证特征分布也会污染验证边界。

错误流程：

```text
全部 X（包含未来验证行）
→ 拟合填补器和缩放器
→ 再划分 train / valid
→ 训练并评价
```

正确流程：

```text
先划分 train / valid
→ 只在 X_train 拟合填补器、缩放器和模型
→ 用已学到的统计量转换 X_valid
→ 评价
```

## 2. 哪些对象会学习

| 对象 | `fit()` 保存什么 | 验证阶段应做什么 |
|---|---|---|
| `SimpleImputer` | 每列填充值 `statistics_` | 仅 `transform` |
| `StandardScaler` | 每列 `mean_`、`scale_` | 仅 `transform` |
| 特征选择器 | 被保留列或阈值 | 仅 `transform` |
| PCA | 主成分方向 | 仅 `transform` |
| Ridge | 回归系数与截距 | 仅 `predict` |

凡是通过数据估计状态的步骤，都必须在允许的训练区域内拟合。

## 3. Pipeline 如何保护边界

```python
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
```

调用：

```python
pipeline.fit(X_train, y_train)
```

相当于按顺序：

1. `imputer.fit_transform(X_train)`；
2. `scaler.fit_transform(填补后的训练数据)`；
3. `ridge.fit(缩放后的训练数据, y_train)`。

随后：

```python
pipeline.predict(X_valid)
```

只会使用已经保存的填充值与缩放参数转换验证数据，然后预测；它不会在验证集上重新拟合。

## 4. Pipeline 不会替你做什么

Pipeline 不会自动：

- 划分训练、验证与测试；
- 判断分组字段是否合理；
- 防止你把验证行传给 `.fit()`；
- 防止标签本身含未来信息；
- 判断某一列是否在现实预测时可获得；
- 保证指标与科学问题一致。

它是组织和复用正确流程的工具，不是自动研究审查器。

## 5. 为什么“验证集没有标签泄漏”仍可能有问题

即使只用 `X_valid` 计算全表均值，也让训练流程知道了未来输入分布。影响有时很小，有时很大；无论分数是否上升，协议都已失去独立性。不能以“这次结果差不多”为泄漏辩护。

## 6. 交叉验证中的 Pipeline

若进行 K 折，应把所有需学习的预处理放进 Pipeline，再把整条 Pipeline 交给 `cross_validate` 或 `GridSearchCV`。这样每一折都会重新拟合自己的填补器、缩放器与模型。

错误：

```text
全训练区域 fit_transform
→ 对已处理数据做 K 折
```

正确：

```text
原始训练区域
→ K 折每轮内部重新 fit Pipeline
```

## 7. 检查内部状态

拟合后可以查看：

```python
pipeline.named_steps["imputer"].statistics_
pipeline.named_steps["scaler"].mean_
pipeline.named_steps["ridge"].coef_
```

这些数组应来自训练数据。验证数据即使分布不同，也不应改变它们。

## 8. 证据边界

本日 notebook 故意创建带缺失值和分布变化的人工数据，只验证防泄漏流程。错误流程与正确流程的成绩谁高不是核心结论；核心结论是只有后者保持了验证边界。
