# Day 19 参考答案

## A. 概念题

1. 32 是唯一隐藏层的单元数；逗号使 `(32,)` 成为单元素元组。
2. 隐藏层输出为 `(902, 32)`，最终回归输出通常整理为 `(902,)`。
3. `alpha` 控制 L2 权重惩罚；学习率由 `learning_rate_init` 控制。
4. `fit_transform()` 会让缩放器学习验证集分布，造成预处理泄漏。验证集只能 `transform()`。
5. 可以诊断当前 MLP 是否拟合、是否出现明显训练—验证差距和是否触及迭代上限；不能宣布它优于或劣于某类算法，因为 Day 19 尚未运行同协议基线，也不能改用测试集寻找好看结果。

## B. 代码阅读

第一处：训练与验证在缩放前被合并，验证分布进入了 `fit()`。第二处：模型又用训练和验证标签共同拟合，外部验证彻底失去独立性。`hidden_layer_sizes=32` 在 scikit-learn 中可以被接受，但教学中建议写 `(32,)`，清楚表达元组结构。

正确结构：

```python
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32,), random_state=42),
)
model.fit(X_train, y_train)
valid_pred = model.predict(X_valid)
```

## C. 最小实现

```python
def evaluate(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    rows = []
    for split, X_part, y_part in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
    ]:
        prediction = model.predict(X_part)
        rows.append({
            "split": split,
            "mae": mean_absolute_error(y_part, prediction),
            "rmse": root_mean_squared_error(y_part, prediction),
            "r2": r2_score(y_part, prediction),
        })
    return pd.DataFrame(rows)
```

## D. 诊断题

可检查：

- 训练—验证差距是否表明过拟合；
- 损失曲线末尾是否仍下降以及是否出现收敛警告；
- 数据缩放是否位于 Pipeline；
- 学习率是否导致震荡或过慢；
- 正则化是否过弱；
- 划分是否存在分布差异；
- 多随机种子下现象是否重复。

仅增加轮数可能让训练误差继续下降，却使过拟合更严重。

## E. 证据边界

一种准确写法：

> 在固定的 ESOL scaffold 训练/验证划分上，我们用 1024 维 ECFP 训练了一个 MLP 来预测水溶解度 logS；实际指标只能说明该公开练习流程能运行，尚未使用真实粘合剂数据，也不能支持粘合剂性能结论。

## F. 动手扩展

合理预期是 `(64,)` 参数更多、通常训练更慢，但指标方向不能预先保证。判断时复用相同验证 RMSE、MAE、R² 和种子；要用多划分或多种子结果讨论稳定性。
