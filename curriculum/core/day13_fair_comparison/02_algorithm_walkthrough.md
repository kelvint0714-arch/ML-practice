# Day 13 算法推演：用同一组折比较五个模型

## 1. 冻结协议

教程在人工回归数据上比较：

1. Dummy；
2. Ridge；
3. 浅决策树；
4. 小型随机森林；
5. 梯度提升。

全部使用相同原始特征、相同 5 折和相同三项指标。参数为教学固定基线，不经测试集选择。

## 2. 为每个模型建立无泄漏流程

```python
models = {
    "dummy": make_pipeline(
        SimpleImputer(strategy="median"),
        DummyRegressor(strategy="mean"),
    ),
    "ridge": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    ),
    # 其他模型也各自放在无泄漏 Pipeline 中。
}
```

Ridge 多一个缩放器是算法需要，不是不公平。关键是缩放器位于 Pipeline 中，并在相同折边界内拟合。

## 3. 一次返回多个指标

```python
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

scores = cross_validate(
    model,
    X,
    y,
    cv=cv_splits,
    scoring=scoring,
    return_train_score=False,
)
```

为了让每个模型严格复用同一批索引，教程先把折转换成列表：

```python
cv_splits = list(cv.split(X))
```

列表中的索引不随模型变化。

## 4. 展开为逐折长表

`cross_validate` 返回每个指标的数组。对第 `i` 折：

```python
fold_rows.append({
    "model": model_name,
    "fold": i + 1,
    "split": "cv_valid",
    "mae": -scores["test_mae"][i],
    "rmse": -scores["test_rmse"][i],
    "r2": scores["test_r2"][i],
    "fit_seconds": scores["fit_time"][i],
})
```

MAE/RMSE 取负恢复普通误差；R² 不取负。

## 5. 完整性检查

若 5 个模型、5 折，应有 25 行：

```python
assert len(fold_metrics) == 25
assert not fold_metrics.duplicated(["model", "fold"]).any()
assert set(fold_metrics["split"]) == {"cv_valid"}
assert (fold_metrics.groupby("model")["fold"].nunique() == 5).all()
```

还应检查数值有限：

```python
assert np.isfinite(fold_metrics[["mae", "rmse", "r2"]]).all().all()
```

## 6. 汇总

```python
model_summary = (
    fold_metrics.groupby("model")
    .agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
    )
    .reset_index()
    .sort_values("rmse_mean")
)
```

排序只是帮助阅读。若两模型平均 RMSE 很接近，应返回逐折差异与标准差，而不是放大名次。

## 7. 保存证据时记录来源

课程 notebook 为了演示会把 CSV 保存到课程目录下的 `tutorial_outputs/`。这些是随附教学运行产物，不是学习者自己的 `experiments/` 证据。

学习者正式完成 Day 13 时，应在：

```text
experiments/day13_fair_comparison/results/
```

保存本人运行的逐折与汇总表，同时记录数据、seed、版本和执行日期。`fit_seconds`
只保留在运行时内存变量中，不进入预存展示或需要严格比较的 fixture。不要直接把教程输出复制成“我完成的实验”。

## 8. 结论模板

> 在固定人工教学数据、相同随机 5 折、相同输入信息和预先声明的五个基线配置下，模型按平均折内验证 RMSE 排序如表所示。逐折波动和 Dummy 对照共同构成证据；该比较只验证程序，不代表 ESOL 或粘合剂任务的模型排名。

实际第一名与数值只能在执行后引用。
