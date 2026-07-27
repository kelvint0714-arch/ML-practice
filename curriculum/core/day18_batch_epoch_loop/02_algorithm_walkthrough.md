# Day 18 算法推演：写一个无泄漏训练循环

## 1. 准备固定训练与验证

教程先生成一次人工数据并固定划分：

```text
X_train, y_train：允许更新参数
X_valid, y_valid：只评价
```

缩放：

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
```

注意验证只调用 `transform`。

## 2. 循环外创建状态

```python
rng = np.random.default_rng(42)
model = MLPRegressor(
    hidden_layer_sizes=(16,),
    solver="sgd",
    learning_rate_init=0.01,
    random_state=42,
)
```

`rng` 控制每轮样本顺序；`model.random_state` 控制 MLP 内部随机状态。两者用途不同，都应记录。

## 3. 一轮索引

```python
order = rng.permutation(len(X_train_scaled))
```

例如 10 个样本可能得到：

```text
[5, 6, 0, 7, 3, 2, 4, 9, 1, 8]
```

它只改变访问顺序，不改变 X 与 y 对齐。

## 4. 切 batch

```python
for start in range(0, len(order), batch_size):
    batch_ids = order[start:start + batch_size]
    model.partial_fit(
        X_train_scaled[batch_ids],
        y_train[batch_ids],
    )
```

若 `n_train=96`、`batch_size=16`，每 epoch 6 次更新。总更新数：

```text
n_epochs × ceil(n_train / batch_size)
```

## 5. Epoch 结束后评价

```python
train_prediction = model.predict(X_train_scaled)
valid_prediction = model.predict(X_valid_scaled)

history.append({
    "epoch": epoch + 1,
    "updates": updates_so_far,
    "train_rmse": root_mean_squared_error(y_train, train_prediction),
    "valid_rmse": root_mean_squared_error(y_valid, valid_prediction),
})
```

验证预测不更新参数。

## 6. 防止验证进入训练的可审计做法

除了肉眼检查，可记录：

```python
training_row_ids_seen.extend(original_train_ids[batch_ids])
```

然后断言：

```python
assert set(training_row_ids_seen).isdisjoint(set(original_valid_ids))
```

更简单的教程实现使用训练局部索引，并检查每 epoch 每个训练局部索引恰好出现一次。

## 7. 完整性断言

```python
expected_batches = math.ceil(len(X_train) / batch_size)
assert updates_so_far == n_epochs * expected_batches
assert len(history) == n_epochs
assert history["epoch"].tolist() == list(range(1, n_epochs + 1))
assert np.isfinite(history[["train_rmse", "valid_rmse"]]).all().all()
```

每 epoch 还可检查：

```python
assert sorted(order.tolist()) == list(range(len(X_train)))
```

保证没有漏行或重复行。

## 8. 绘图

横轴应明确是 epoch，不是 batch：

```python
plt.plot(history["epoch"], history["train_rmse"], label="train")
plt.plot(history["epoch"], history["valid_rmse"], label="validation")
```

标题写明人工数据、模型和指标。图中观察必须在实际执行后描述。

## 9. 下一步

Day 19 会把 MLP 放到 ESOL ECFP 输入上；Day 20 再学习正则化和早停。迁移时不能把教程人工数据曲线复制过去，必须重新运行并记录真实来源。
