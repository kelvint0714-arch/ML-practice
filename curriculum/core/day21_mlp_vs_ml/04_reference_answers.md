# Day 21 参考答案

## A. 协议审计

1. 不一定破坏。不同算法可以使用预先声明的合理预处理，且预处理应放入 Pipeline。
2. 会混入搜索预算差异，不能直接宣称算法家族胜负。
3. 会破坏比较，因为验证样本难度不同。
4. 是必要的公平条件。
5. 会产生选择偏差；应保存全部种子并报告汇总。

## B. 代码示例

```python
rows = []
for name, model in models.items():
    started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
    for split, X_part, y_part in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
    ]:
        prediction = model.predict(X_part)
        rows.append({
            "model": name,
            "split": split,
            "rmse": root_mean_squared_error(y_part, prediction),
            "mae": mean_absolute_error(y_part, prediction),
            "r2": r2_score(y_part, prediction),
            "fit_seconds": fit_seconds,
        })
results = pd.DataFrame(rows)
```

计时器在 `fit()` 之后、任何 `predict()` 之前停止，所以字段名称
`fit_seconds` 与实际口径一致。

## C. 表格解释

> 在这张示例表中，MLP 的验证 RMSE 比 RF 低 0.02，而 MAE 更高 0.08。差异很小且来自一次验证划分，排名可能随种子或样本划分改变。该表若来自 ESOL，只能用于 logS 方法练习，不能推断粘合剂性能。

注意：题目数值是教学假设，不是新实验结果。

## D. 计时题

硬件型号、线程数、缓存状态、后台负载、库版本和首次编译/加载成本都会影响计时。应记录环境并重复测量，跨机器比较尤其要谨慎。

## E. 选择基础模型

Stacking 的收益依赖误差互补性，所以不必机械选择两个单独分数最低的模型。可以预先比较 OOF/验证残差相关性，并把“最强传统模型 + MLP”和“较互补传统模型 + MLP”列为有限候选；仍需无泄漏 OOF、多种子和简单平均对照。

## F. 配置溯源

读取保存配置可以证明 Dummy、Ridge、受限决策树和随机森林沿用 Day 07
冻结配方，避免看过新结果后改动传统模型预算。配置缺失时应抛出清晰错误并
说明无法完成可复现比较；静默使用默认值会让“配方一致”成为错误陈述。
