# Day 10 算法推演：一次只改变 seed

## 1. 建立实验矩阵

先把控制变量写成表：

| 运行 | 训练行 | 验证行 | 树数 | 叶节点最少样本 | seed |
|---:|---|---|---:|---:|---:|
| 1 | 固定 A | 固定 B | 80 | 2 | 0 |
| 2 | 固定 A | 固定 B | 80 | 2 | 1 |
| 3 | 固定 A | 固定 B | 80 | 2 | 2 |
| … | 固定 A | 固定 B | 80 | 2 | … |

每一行只有 seed 不同。若树数也变化，这不再是纯种子稳定性实验。

## 2. 单次运行函数

把“一次训练”封装成函数可以减少漏改：

```python
def run_one_seed(seed, X_train, y_train, X_valid, y_valid):
    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    return {
        "seed": seed,
        "mae": mean_absolute_error(y_valid, prediction),
        "rmse": root_mean_squared_error(y_valid, prediction),
        "r2": r2_score(y_valid, prediction),
    }, prediction
```

教程使用 `n_jobs=1`，减少并行环境差异，也让学习者更容易复查。

## 3. 为什么每次都新建模型

```python
records = []
for seed in SEEDS:
    result, _ = run_one_seed(seed, ...)
    records.append(result)
```

模型在函数内部创建。若把同一个森林对象放在循环外并不断修改参数，容易携带状态或混淆每次运行的边界。

## 4. 汇总但不删除原表

假设四次 RMSE 为：

```text
0.92, 0.88, 0.95, 0.89
```

逐次表是原始证据；汇总表只是视图：

```python
summary = results[["mae", "rmse", "r2"]].agg(
    ["mean", "std", "min", "max"]
)
```

pandas 默认标准差是样本标准差。汇报示例：

> 在固定划分和固定森林参数下，8 个预先声明 seed 的验证 RMSE 为 `mean ± sample std`。该波动只描述模型随机性。

实际数字必须来自已经执行的输出，不能提前写入报告。

## 5. 同 seed 复现检查

选择一个已经在列表中的 seed，再独立运行两次：

```python
_, prediction_a = run_one_seed(42, ...)
_, prediction_b = run_one_seed(42, ...)

max_abs_diff = np.max(np.abs(prediction_a - prediction_b))
assert np.allclose(prediction_a, prediction_b)
```

`np.allclose` 比逐个使用 `==` 更适合浮点结果。若不一致，应先检查数据行顺序、模型参数、版本和并行设置。

## 6. 结果解读顺序

1. 检查运行行数是否等于 seed 数；
2. 检查 seed 是否与预先声明列表完全一致；
3. 检查同 seed 重复是否一致；
4. 查看均值与标准差；
5. 查看 min/max 是否有异常；
6. 最后写能力边界。

不要先找最好 seed 再删掉其他行。

## 7. 延伸实验的正确命名

- 固定 split、多模型 seed：模型随机性；
- 固定模型 seed、多 split seed：划分敏感性；
- 多 split × 多模型 seed：二维重复实验；
- 不同实验批次：实验或批次变异，需真实分组信息。

命名准确比得到更漂亮的数字重要。
