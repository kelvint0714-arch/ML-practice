# Day 34 练习

先独立作答，不要先打开参考答案。

## 练习 1：任务维度

一个 batch 有 12 张图、共 247 个节点、隐藏宽度 32、类别数 2。写出：

1. GIN 后节点表示 shape；
2. pooling 后图表示 shape；
3. logits shape。

## 练习 2：`batch` 向量

两张图分别有 2 个和 4 个节点。写出它们拼接后的一个合法 `batch` 向量，并解释每个数字。

## 练习 3：找错误

```python
logits = model(batch.x, batch.edge_index, batch.batch)
loss = cross_entropy(logits, batch.x)
```

指出错误，并写出正确的标签对象。

## 练习 4：多数类基线

训练标签为 `[0, 1, 1, 1, 0]`，validation 标签为 `[1, 0, 1, 1]`。

1. 训练集多数类是什么？
2. 对 validation 全预测多数类时 accuracy 是多少？
3. 为什么不能用 test 标签先决定多数类？

## 练习 5：协议边界

判断正误并说明原因：

1. 为了让结果更好，可以看 test accuracy 后再更换池化方式；
2. 固定 60 个 epoch 后报告 validation，不等于最终测试结论；
3. MUTAG 上 GIN 有效就证明它能预测下游任务目标。

## 练习 6：补断言

补全：

```python
assert one_batch.x.shape[0] == __________
assert logits.shape[0] == __________
assert logits.shape[1] == dataset.num_classes
```

## 练习 7：用自己的话解释

不用“高级”“自动提取”这类空话，用 4–6 句话解释：

```text
节点特征 → GINConv → 节点表示 → global_add_pool → 图表示 → logits
```

## 练习 8：结果边界

写出本日 validation 数字至少不能证明的三件事。
