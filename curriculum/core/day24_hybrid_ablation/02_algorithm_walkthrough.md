# Day 24 算法走读：建立可审计消融矩阵

## 步骤 1：运行前定义变体

```python
variant_specs = [
    {"variant": "tree_only", "change": "baseline"},
    {"variant": "mlp_only", "change": "replace tree with MLP"},
    {"variant": "simple_mean", "change": "fixed equal-weight ensemble"},
    {"variant": "stack_tree_only", "change": "OOF tree + Ridge meta"},
    {"variant": "stack_tree_mlp", "change": "add MLP to previous stack"},
]
```

先保存这张表，再开始拟合。不要看到结果后删除“不好看”的行。

## 步骤 2：独立创建模型

使用 `clone()` 或工厂函数，避免变体共享已经拟合的状态：

```python
tree_single = clone(tree)
mlp_single = clone(mlp)
stack_tree_only = StackingRegressor(
    estimators=[("tree", clone(tree))],
    final_estimator=Ridge(alpha=1.0),
    cv=inner_splits,
)
stack_tree_mlp = StackingRegressor(
    estimators=[("tree", clone(tree)), ("mlp", clone(mlp))],
    final_estimator=Ridge(alpha=1.0),
    cv=inner_splits,
)
```

所有对象使用相同冻结参数。

## 步骤 3：记录拟合成本

```python
started = perf_counter()
model.fit(X_train, y_train)
fit_seconds = perf_counter() - started
```

简单平均需要拟合两个单模型；若已在同一次实验中拟合，可复用它们的验证预测，但应在表中说明成本口径。
凡是包含 MLP 的拟合都应在局部 `catch_warnings(record=True)` 中执行，
统计并展示 `ConvergenceWarning`；不要用全局过滤器隐藏。

## 步骤 4：集中保存预测

```python
predictions = {
    "tree_only": tree_pred,
    "mlp_only": mlp_pred,
    "simple_mean": (tree_pred + mlp_pred) / 2,
    "stack_tree_only": stack_tree_only_pred,
    "stack_tree_mlp": stack_tree_mlp_pred,
}
```

把“生成预测”和“计算指标”分开，减少每个变体使用不同函数的风险。
断言两个 stack 的一级模型数分别为 1 和 2，并检查每个 split 的
scaffold groups 互斥。

## 步骤 5：统一评价

```python
baseline_rmse = root_mean_squared_error(y_valid, predictions["tree_only"])
rows = []
for name, pred in predictions.items():
    rmse = root_mean_squared_error(y_valid, pred)
    rows.append({
        "variant": name,
        "rmse": rmse,
        "delta_vs_tree": rmse - baseline_rmse,
        "mae": mean_absolute_error(y_valid, pred),
        "r2": r2_score(y_valid, pred),
    })
```

结果按 `variant_specs` 原始顺序保存一份，再额外生成按 RMSE 排序的查看表。不要让排序丢失预注册顺序。

## 步骤 6：稳定性

理想做法是在预先声明的多个种子上重复整个矩阵：

```text
每个 seed：
    创建所有模型
    使用同一外部划分
    运行完整变体集合
汇总每个变体的均值、标准差和逐 seed 差值
```

只改变计划中的模型随机种子，不让每个变体使用不同样本。

## 步骤 7：做决定

按以下优先级：

1. 数据/泄漏审计是否通过；
2. 相对最强单模型是否有稳定收益；
3. 是否超过简单平均；
4. 收益是否值得额外成本；
5. 若没有，明确冻结较简单方案。

## 报告模板

> `stack_tree_mlp` 在【实际运行协议】下相对 `stack_tree_only` 的 ΔRMSE
> 为【值】，相对简单平均为【值】。在【种子数】次重复中改善方向出现
> 【次数】次。考虑耗时和可审计性，当前决定【保留/放弃】MLP 组件。
> 此结论只适用于 ESOL 练习。
