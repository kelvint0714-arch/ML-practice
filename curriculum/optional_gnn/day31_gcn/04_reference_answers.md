# Day 31.4：参考答案

## 练习 1

| 对象 | shape |
|---|---|
| `x` | `[34, 34]` |
| `edge_index` | `[2, E]`，`E` 由实际存储边条目数决定 |
| 第一层 GCN 输出 | `[34, 16]` |
| 第二层 logits | `[34, 4]` |
| `logits.argmax(dim=1)` | `[34]` |
| `train_mask` | `[34]` |

`edge_index` 的第一维固定是 2，一列保存一个源节点编号和一个目标节点编号。

## 练习 2

`(34,)` 表示有 34 个类别编号的一维向量，第 `i` 项直接对应节点 `i`。`(34, 1)` 是二维列矩阵，多出一个长度为 1 的轴。`cross_entropy` 的节点分类标签通常需要 `[N]`，所以直接使用一维整数标签最自然。

## 练习 3

原代码让 validation/test 节点的标签也参与 loss，破坏了划分边界。正确写法：

```python
loss = F.cross_entropy(
    logits[train_mask],
    data.y[train_mask],
)
```

## 练习 4

```python
assert not torch.any(train_mask & valid_mask)
assert not torch.any(train_mask & test_mask)
assert not torch.any(valid_mask & test_mask)
assert torch.all(train_mask | valid_mask | test_mask)
```

前三条检查互斥，最后一条检查覆盖。

## 练习 5

`model.train()` 启用训练行为，当前模型中的 Dropout 会随机丢弃部分表示；训练代码还需要梯度。`model.eval()` 关闭 Dropout 的随机训练行为，`torch.no_grad()` 不记录反向传播图。评价时需要稳定输出且不更新参数，因此通常同时使用后两者。

## 练习 6

```python
def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
```

- 同一个 GCN 的参数矩阵尺寸由输入宽度、隐藏宽度和输出宽度决定，通常不随当前图的节点数改变；
- 增大隐藏宽度会扩大相邻层权重矩阵，因此参数量通常增加。

## 练习 7

经过两层后，节点 2 可能融合节点 `0、1、2、3、4` 的信息：第一层接收自己与一跳邻居，第二层继续传播到两跳。使用“可能”，因为实际表示还受权重、归一化、激活、Dropout 和训练结果影响，信息可能被压缩或削弱。

## 练习 8

允许：

- 检查 `test_mask.dtype == torch.bool`；
- 检查 test 节点数量以及与其他掩码是否重叠。

不允许：

- 计算 `data.y[test_mask]` 对应的准确率；
- 根据 test 表现改变隐藏层、学习率、epoch 或选择模型。

## 练习 9

参考写法：

> 在 PyG KarateClub 教学图上训练了两层 GCN，并仅用 validation 节点观察开发过程，test 标签保持封存。该结果用于理解节点消息传递与掩码，不包含分子或下游任务数据，不能作为材料性能结论。

答案不必逐字相同，但必须保留数据来源、验证用途与外推边界。

返回：[Day 31 学习入口](README.md)。
