# Day 34 参考答案

完成练习后再核对。

## 练习 1

1. 节点表示：`[247, 32]`；
2. 图表示：`[12, 32]`；
3. logits：`[12, 2]`。

池化把节点行数变成图行数，隐藏宽度保持 32。

## 练习 2

一个合法答案：

```text
[0, 0, 1, 1, 1, 1]
```

前两个节点属于第 0 张图，后四个节点属于第 1 张图。

## 练习 3

`cross_entropy` 的标签应是一张图一个整数类别，而 `batch.x` 是节点特征。正确写法：

```python
loss = cross_entropy(logits, batch.y.view(-1))
```

## 练习 4

1. 多数类是 `1`；
2. validation 中 4 个样本有 3 个是 `1`，accuracy 为 `3/4 = 0.75`；
3. test 必须保留给冻结协议后的最终评价。用 test 决定规则会把最终答案泄漏进开发过程。

## 练习 5

1. 错。根据 test 选择 pooling 是测试集泄漏；
2. 对。它只是固定教学协议下的 validation 结果；
3. 错。MUTAG 与下游任务的样本、图定义、目标和数据来源都不同。

## 练习 6

```python
assert one_batch.x.shape[0] == one_batch.batch.shape[0]
assert logits.shape[0] == one_batch.num_graphs
assert logits.shape[1] == dataset.num_classes
```

## 练习 7

示例：每个节点先带有自己的输入特征。GINConv 把自身表示和邻居表示求和后交给 MLP，得到新的节点表示。两层 GIN 让节点逐步接收更远邻域的信息。`global_add_pool` 根据 `batch` 向量把同一张图的节点表示相加，因此每张图只剩一个向量。线性层把图向量变成类别 logits，最大 logit 对应预测类别。

## 练习 8

可写：

- 不能证明 GIN 普遍优于多数类或其他图模型；
- 不能证明结果在另一随机划分上稳定；
- 不能证明模型能预测真实下游任务性能；
- 不能证明 MUTAG 的节点/边语义适合候选方案；
- 不能把 validation 当作未见测试集性能。
