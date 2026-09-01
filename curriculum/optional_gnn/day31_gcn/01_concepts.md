# Day 31.1：GCN、shape、mask 与训练模式

## 1. 今天解决什么问题

KarateClub 数据是一张图：

- 每个成员是一个节点；
- 成员关系是边；
- 每个节点有一个特征向量；
- 每个节点有一个类别标签。

任务是“节点分类”：为图中的每个节点输出一个类别分数。它与“整张分子图输出一个性质值”的图回归不同。

## 2. 先读懂四个核心张量

设：

- `N`：节点数；
- `F`：每个节点的输入特征数；
- `E`：代码中存储的有向边条目数；
- `C`：类别数；
- `H`：隐藏表示宽度。

| 名称 | 典型 shape | 含义 |
|---|---|---|
| `data.x` | `[N, F]` | 每行是一个节点，每列是一个输入特征 |
| `data.edge_index` | `[2, E]` | 每一列是一条 `源节点 → 目标节点` 边 |
| `data.y` | `[N]` | 每个节点对应一个整数类别 |
| `train_mask` | `[N]` | 哪些节点可以用于计算训练 loss |
| `valid_mask` | `[N]` | 哪些节点可用于开发阶段评价 |
| `test_mask` | `[N]` | 哪些节点留到最终一次评价 |

`data.y.shape == (N,)` 右边没有第二个数字，是因为它是一维向量。第 `i` 个位置就是第 `i` 个节点的类别编号。

布尔掩码也是一维：

```python
train_mask = torch.tensor([True, False, True, False])
```

用它索引 `[N, C]` 的类别分数：

```python
train_logits = logits[train_mask]
```

结果只保留掩码为 `True` 的行，列数 `C` 不变。

## 3. 什么是消息传递

普通全连接层只看到当前行的特征。图卷积还要读取邻接关系，使一个节点能够接收邻居的信息。

对节点 `i`，GCN 可以直观理解为：

```text
收集节点 i 自己和邻居的表示
→ 按节点度数做归一化
→ 加权求和
→ 乘以可学习权重
→ 得到节点 i 的新表示
```

简化公式：

```text
H_next = activation(归一化邻接矩阵 × H_current × W)
```

- `H_current`：当前节点表示；
- `W`：模型学习的权重；
- 归一化用于避免高度节点简单地得到过大的数值；
- 实现通常会加入自环，让节点也保留自己的信息。

这不是“把邻居标签传给模型”。模型使用的是节点特征和图连接，训练标签只参与训练节点的 loss。

## 4. 两层 GCN 的 shape

固定：

```text
输入特征数 = F
隐藏宽度 = H = 16
类别数 = C
```

数据流：

```text
x                 [N, F]
GCNConv(F, H)  →  [N, H]
ReLU           →  [N, H]
Dropout        →  [N, H]
GCNConv(H, C)  →  [N, C]  logits
```

行数一直是 `N`，因为任务需要给每个节点一个输出。列数随层定义变化。

两层消息传递通常能融合大约两跳邻域的信息，但真实影响还取决于自环、图结构、非线性和训练结果。

## 5. logits 不是概率

第二层输出 `[N, C]` 的 `logits`：

```text
节点 0: [1.2, -0.3, 0.8, 0.1]
```

它们是未经归一化的类别分数。`torch.nn.functional.cross_entropy` 接收 logits 和整数类别，不需要先手动调用 softmax。

预测类别可以写成：

```python
prediction = logits.argmax(dim=1)
```

`dim=1` 表示在每一行的 `C` 个类别分数中找最大值，结果 shape 是 `[N]`。

## 6. mask 为什么是实验边界

训练：

```python
loss = F.cross_entropy(
    logits[train_mask],
    data.y[train_mask],
)
```

这表示只有训练节点的标签影响梯度。

validation：

- 不调用 `backward()`；
- 用于检查训练过程和比较开发方案；
- 频繁查看会逐渐对 validation 过拟合。

test：

- Day 31–33 不计算 test accuracy；
- 只检查掩码大小、类型、互斥性，不读取 `data.y[test_mask]`；
- 最终协议冻结后才能使用一次。

KarateClub 的节点特征和完整边结构在训练时可见，这是“传导式节点分类”的常见设置。validation 标签只用于开发期评价、不进入训练 loss；test 标签才始终封存。它与未来新图完全不可见的归纳式任务不同。

## 7. `train()`、`eval()` 和 `no_grad()`

### `model.train()`

把模型切换到训练模式。Dropout 会随机丢弃部分表示。

### `model.eval()`

把模型切换到评价模式。Dropout 停止随机丢弃，使评价稳定。

### `torch.no_grad()`

告诉 PyTorch 本段不建立反向传播图，减少评价时的内存和计算。

标准顺序：

```python
model.train()
# forward → loss → backward → optimizer.step

model.eval()
with torch.no_grad():
    valid_logits = model(data.x, data.edge_index)
```

`eval()` 不等于 `no_grad()`：前者改变某些层的行为，后者关闭梯度记录。评价时通常两个都要用。

## 8. 参数量是什么

参数量是模型中需要通过训练学习的标量个数：

```python
parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
```

参数量不是“层数”，也不是“节点数”。模型越大通常表达能力越强，但也可能更容易过拟合、占用更多计算。

Day 32–33 会同时报告参数量，因为相同隐藏宽度并不保证不同图算子拥有相同参数量。

## 9. 今天不能得出的结论

- GCN 在 KarateClub 上有效，不证明它适合分子性质预测；
- validation 分数较高，不等于 test 表现已知；
- 两层 GCN 能运行，不表示越深一定越好；
- 一次随机初始化不能证明算法稳定；
- 教学图的结果不能写成下游任务研究结论。

下一步：[逐行推演一次完整训练](02_algorithm_walkthrough.md)。
