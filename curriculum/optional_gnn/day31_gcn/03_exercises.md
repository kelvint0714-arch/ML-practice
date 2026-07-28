# Day 31.3：独立练习

请先在自己的实验目录完成，再查看参考答案。

## 练习 1：shape 追踪

假设：

```text
N = 34
F = 34
H = 16
C = 4
```

写出下列对象的 shape：

1. `x`
2. `edge_index`
3. 第一层 GCN 输出
4. 第二层 logits
5. `logits.argmax(dim=1)`
6. `train_mask`

## 练习 2：为什么 y 是一维

用自己的话解释：

```python
data.y.shape == (34,)
```

为什么不是 `(34, 1)`？这两个 shape 在索引和损失函数中有什么区别？

## 练习 3：修复数据泄漏

下面代码哪里错了？写出正确版本。

```python
loss = F.cross_entropy(logits, data.y)
```

## 练习 4：检查掩码

编写四个断言，证明：

1. train 与 validation 不重叠；
2. train 与 test 不重叠；
3. validation 与 test 不重叠；
4. 三个掩码覆盖全部节点。

## 练习 5：训练与评价模式

解释下面两段代码为什么不能随意互换：

```python
model.train()
logits = model(data.x, data.edge_index)
```

```python
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
```

## 练习 6：参数量

编写函数统计所有 `requires_grad=True` 的参数数量。然后回答：

- 参数量是否会随着节点数增加而增加？
- 参数量是否会随着隐藏宽度增加而增加？

## 练习 7：两跳信息

画一个五节点链：

```text
0 — 1 — 2 — 3 — 4
```

只考虑图结构，经过两层消息传递后，节点 2 可能接收到哪些节点的信息？为什么这里使用“可能”而不是“必然完整保留”？

## 练习 8：test 封存

列出 Day 31 中：

- 允许对 `test_mask` 做的两项检查；
- 不允许做的两项操作。

## 练习 9：证据边界

写一段不超过 100 字的实验结论，必须同时包含：

- 使用的数据；
- 做了什么；
- validation 的作用；
- 为什么不能外推到粘合剂。

完成后：[查看参考答案](04_reference_answers.md)。
