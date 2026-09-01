# Day 18：Batch、Epoch 与训练循环

> 状态：待学习。代码只提供训练骨架，运行结果需要你自己记录。

## 学习文件导航

按顺序完成以下五个课程文件：

1. [概念讲义](01_concepts.md)
2. [算法推演](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习题](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 用确定性人工数据演示 `partial_fit()` 的 batch/epoch 循环；课程保存的执行输出不等于学习者已完成实验。本人曲线与记录再写入 `learning_outputs/day18_batch_epoch_loop/`。

## 今天为什么学

真实数据通常不会只更新一次参数。
训练过程会把数据分成小批次，重复多轮，让模型逐步调整权重。

今天继续不使用 PyTorch，改用 scikit-learn 的 `MLPRegressor.partial_fit()` 观察显式的 Batch 和 Epoch 循环。

## 前置条件

- 完成 Day 17，知道损失、梯度和学习率的作用；
- 能读懂 `for` 循环和数组切片；
- 知道训练数据可以更新模型，验证数据不能；
- 已经准备好训练与验证数组。

## 今日产出

完成学习后，你应该产生：

1. 一张 Batch、Iteration、Epoch 的关系图；
2. 每个 Epoch 的训练与验证 RMSE 记录；
3. 一条训练曲线；
4. 一段说明验证集没有参与 `partial_fit()` 的文字。

## 核心概念

### 1. Batch

Batch 是一次参数更新看到的一小组样本。
例如 100 个训练样本、批大小 20，一个 Epoch 大约包含 5 个 Batch。

### 2. Iteration

Iteration 通常表示一次参数更新。
不同库对计数细节可能略有差别，因此报告时要写清自己的定义。

### 3. Epoch

一个 Epoch 表示训练过程大致遍历了一次全部训练样本。
更多 Epoch 不一定更好，因为模型可能逐渐过拟合。

### 4. 打乱顺序

每个 Epoch 开始前打乱训练行，避免模型总以完全相同的顺序看到样本。
打乱只改变顺序，不能把训练和验证混在一起。

## 分步骤任务

1. 对训练数据拟合缩放器，再转换训练和验证数据。
2. 创建固定随机种子的 `MLPRegressor`。
3. 决定批大小和 Epoch 数，并先写进记录。
4. 每个 Epoch 生成训练行的新排列。
5. 按批次切片，只把训练批传给 `partial_fit()`。
6. 每个 Epoch 结束后计算训练与验证 RMSE。
7. 保存曲线所需的历史记录。
8. 检查验证数组从未出现在 `partial_fit()` 中。
9. 观察训练和验证曲线是否开始分离。

## 核心代码骨架

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

rng = np.random.default_rng(42)
model = MLPRegressor(
    hidden_layer_sizes=(32,),
    solver="sgd",
    learning_rate_init=0.001,
    random_state=42,
)

batch_size = 32
n_epochs = 20
history = []

for epoch in range(n_epochs):
    order = rng.permutation(len(X_train_scaled))

    for start in range(0, len(order), batch_size):
        batch_ids = order[start:start + batch_size]
        model.partial_fit(
            X_train_scaled[batch_ids],
            y_train[batch_ids],
        )

    train_pred = model.predict(X_train_scaled)
    valid_pred = model.predict(X_valid_scaled)
    history.append({
        "epoch": epoch + 1,
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "valid_rmse": np.sqrt(mean_squared_error(y_valid, valid_pred)),
    })

print(history[-1])
```

## 只解释今天新增的语法

- `rng.permutation(n)` 生成 `0` 到 `n-1` 的随机排列。
- `range(0, len(order), batch_size)` 每次让 `start` 增加一个批大小。
- `order[start:start + batch_size]` 取得当前批次的行号。
- `partial_fit()` 在已有权重基础上继续更新，而不是每次重新建模。
- `history[-1]` 取列表最后一条记录。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 验证数据传给 `partial_fit` | 发生泄漏 | 验证集只用于 `predict` |
| 每批重新创建模型 | 权重不断清零 | 循环外创建一次模型 |
| 在全部数据上拟合缩放器 | 预处理泄漏 | `scaler.fit` 只看训练集 |
| 只看最后一个 Epoch | 看不到过拟合过程 | 保存完整 history |
| 把 Epoch 数当成越多越好 | 可能过拟合 | 同时观察验证曲线 |

## 完成标准

- [ ] 我能区分 Batch、Iteration 和 Epoch；
- [ ] 模型只在循环外创建一次；
- [ ] 每批只含训练样本；
- [ ] 验证集只用于预测和计算指标；
- [ ] 我保存了每个 Epoch 的训练与验证 RMSE；
- [ ] 我能指出曲线分离可能表示什么。

## 自测问题

1. 100 个样本、批大小 32，一个 Epoch 有多少个批次？
2. 为什么每个 Epoch 要打乱训练顺序？
3. `partial_fit()` 与重新创建模型有什么区别？
4. 验证 RMSE 先降后升可能说明什么？
5. 为什么今天仍然不需要 PyTorch？

## 上一天 / 下一天

- 上一天：[Day 17：损失函数与优化](../day17_loss_optimizer/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 19：在 ESOL 上建立 MLP 基线](../day19_mlp_esol/README.md)
