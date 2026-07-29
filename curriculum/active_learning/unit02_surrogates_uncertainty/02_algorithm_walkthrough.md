# Unit 02 算法走读：GP 与 RF 集成

## GP 输入、动作、输出

- 输入：`X_labeled.shape == (n, d)`、`y_labeled.shape == (n,)`；
- 动作：根据核函数拟合已标注数据；
- 输出：`gp_mean.shape == (m,)`、`gp_std.shape == (m,)`。

```python
gp.fit(X_labeled, y_labeled)
gp_mean, gp_std = gp.predict(X_pool, return_std=True)
```

`return_std=True` 要求模型额外返回每个候选的标准差。

## RF 集成输入、动作、输出

```python
member_predictions = []
for seed in seeds:
    model = RandomForestRegressor(random_state=seed)
    model.fit(X_labeled, y_labeled)
    member_predictions.append(model.predict(X_pool))

prediction_matrix = np.vstack(member_predictions)
rf_mean = prediction_matrix.mean(axis=0)
rf_std = prediction_matrix.std(axis=0)
```

若有 5 个成员、20 个候选：

- 每次 `predict` 得到 `(20,)`；
- `vstack` 后是 `(5, 20)`；
- `axis=0` 沿模型方向汇总，输出 `(20,)`。

## 为什么不能直接比较 std 大小

GP 标准差和 RF 分歧来源不同，尺度也可能不同。可以分别用于各自模型的排序，但不能看到 `0.8 > 0.3` 就断言 RF 比 GP 更不可靠。

## 离线诊断

本单元两个一次性 query 都固定、campaign 结束后，evaluator 才可以计算：

```python
absolute_error = np.abs(hidden_y_pool - prediction_mean)
```

然后比较高分歧组和低分歧组的平均误差。该步骤用于事后评价不确定性，
不能把整池误差返回策略，也不能继续用它选下一轮。

## 手算

某候选的 4 个 RF 成员预测为 `5.0, 5.4, 4.8, 5.2`：

- 均值：5.1；
- 分歧：四个预测围绕 5.1 的标准差；
- 真实值在 query 前仍未知。
