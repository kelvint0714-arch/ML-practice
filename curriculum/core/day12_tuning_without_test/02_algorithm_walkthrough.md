# Day 12 算法推演：训练内选择，外部验证一次

## 1. 先声明协议

运行前写下：

```text
候选 alpha：0.01, 0.1, 1, 10, 100
主指标：训练内 5 折 RMSE
折规则：KFold(shuffle=True, random_state=42)
预处理：中位数填补 + 标准化
外部验证：参数冻结后评价一次
测试集：不创建、不使用
```

这份声明能区分“预先设计”与“看分数后修改”。

## 2. 每个候选实际训练几次

5 个 `alpha` × 5 折 = 25 次折内拟合。

默认 `refit=True` 时，搜索结束后还会用最佳参数在完整训练区域拟合一次，因此至少 26 次。不能把“五折交叉验证”简单理解成无论多少候选都只训练五次。

## 3. 单个候选的一折

对 `alpha=1.0`、第 1 折：

```text
第1折训练行
→ fit 中位数
→ transform
→ fit 均值/尺度
→ transform
→ fit Ridge(alpha=1)

第1折验证行
→ 用训练中位数 transform
→ 用训练尺度 transform
→ predict
→ 计算 RMSE
```

第 2 折会得到全新的填补器、缩放器和 Ridge。

## 4. 建立搜索

```python
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    return_train_score=True,
    refit=True,
)
search.fit(X_train, y_train)
```

`search.fit` 只能接收训练区域。外部验证数组不能出现在这一行。

## 5. 整理候选表

```python
candidate_table = (
    pd.DataFrame(search.cv_results_)
    .loc[:, [
        "param_ridge__alpha",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]]
    .copy()
)
candidate_table["mean_cv_rmse"] = -candidate_table["mean_test_score"]
candidate_table["std_cv_rmse"] = candidate_table["std_test_score"]
```

负号只改变误差方向；标准差乘以负号后数值大小不变，因此直接复制正值即可。

## 6. 冻结后做外部验证

```python
frozen_pipeline = search.best_estimator_
valid_prediction = frozen_pipeline.predict(X_valid)
external_valid_rmse = root_mean_squared_error(y_valid, valid_prediction)
```

若外部验证不如训练内 CV：

1. 如实保留差异；
2. 检查外部验证是否更难或分布不同；
3. 不要删除该结果；
4. 若重新设计候选，应把它记录为下一轮开发，而非继续把同一个验证分数包装成一次独立检查。

## 7. 五个关键断言

```python
assert set(candidate_table["param_ridge__alpha"]) == set(ALPHAS)
assert len(candidate_table) == len(ALPHAS)
assert search.best_params_["ridge__alpha"] in ALPHAS
assert np.isfinite(external_valid_rmse)
assert X_valid.shape[1] == X_train.shape[1]
```

断言检查实现，不证明参数具有普遍最优性。

## 8. 如何写结果

合格表达模板：

> 在固定人工训练区域上，使用带折内填补与标准化的 5 折网格搜索，从预先声明的 5 个 `alpha` 中选择一个。冻结后仅在外部验证集评价一次。该结果演示无泄漏选择程序，不代表测试性能，也不代表 ESOL 或粘合剂任务最佳参数。

实际 `alpha` 与分数必须从已执行 notebook 输出引用。
