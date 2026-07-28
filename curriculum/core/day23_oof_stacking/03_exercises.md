# Day 23 练习

## A. 形状题

训练集有 902 个样本，基础模型为冻结随机森林、MLP 两个：

1. `passthrough=False` 时 OOF 矩阵是什么形状？
2. 若 `passthrough=True` 且原始输入 1024 维，第二层输入是什么形状？
3. 为什么第二种方案的正则化更重要？

## B. 分组覆盖

写代码验证 GroupKFold 中每个样本恰好进入一次 holdout，
并验证每折训练 scaffold 与 holdout scaffold 互斥。

## C. API 审计

解释以下参数：

```python
StackingRegressor(
    estimators=...,
    final_estimator=...,
    cv=...,
    passthrough=False,
)
```

并说明为什么课程禁止 `cv="prefit"`。
再说明为什么要把 `GroupKFold(...).split(...)` 先变成显式列表。

## D. 代码改错

```python
scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
mlp = MLPRegressor().fit(X_scaled, y_train)
stack = StackingRegressor(
    estimators=[("mlp", mlp)],
    cv="prefit",
)
stack.fit(X_scaled, y_train)
```

至少指出两处风险，并给出正确结构。

## E. 结果题

若 tree、MLP、mean、stack 的验证 RMSE 分别为 1.70、1.85、1.68、1.69，你会保留哪一个作为下一步候选？说明为何这仍不是最终结论。
