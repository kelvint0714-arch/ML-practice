# Day 19 练习

先独立完成，再查看 `04_reference_answers.md`。

## A. 概念题

1. `hidden_layer_sizes=(32,)` 中，32 和逗号分别表示什么？
2. 写出 `X: (902, 1024)` 经过 32 单元隐藏层后的矩阵形状。
3. 为什么 `alpha=0.001` 不是“学习率为 0.001”？
4. 为什么验证集不能调用 `scaler.fit_transform()`？
5. 只看到一条 MLP 的 train/valid RMSE 时，可以诊断什么、不能宣布什么？

## B. 代码阅读

找出下列代码中的两处问题：

```python
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(np.vstack([X_train, X_valid]))
model = MLPRegressor(hidden_layer_sizes=32)
model.fit(X_all_scaled, np.concatenate([y_train, y_valid]))
```

## C. 最小实现

补全函数，使其返回训练与验证两行结果：

```python
def evaluate(model, X_train, y_train, X_valid, y_valid):
    # 1. fit
    # 2. 对 train/valid predict
    # 3. 返回包含 split、mae、rmse、r2 的 DataFrame
    ...
```

## D. 诊断题

某次运行结果如下：

```text
train RMSE = 0.20
valid RMSE = 2.10
n_iter_ = 120
max_iter = 120
```

请写出至少三个需要检查的方向。不要仅回答“增加 `max_iter`”。

## E. 证据边界

把这句话改写得准确：

> 神经网络已经证明可以准确预测下游任务性能。

要求在改写中包含数据集、目标、划分和限制。

## F. 动手扩展

保持所有其他设置不变，只把隐藏层从 `(32,)` 改为 `(64,)`。运行前先写下：

- 唯一变化项；
- 你预计训练时间如何变化；
- 你将用什么指标判断；
- 为什么一次结果不能证明 32 或 64 普遍更好。
