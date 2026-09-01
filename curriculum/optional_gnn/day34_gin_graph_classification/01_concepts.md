# Day 34.1：从节点分类到整图分类

## 1. 今天的输入和输出变了

Day 31–33 在一张 KarateClub 图上为每个节点预测类别：

```text
输入：一张图
输出：每个节点一行 logits，shape = [节点数, 类别数]
```

Day 34 要为每一张图预测一个类别：

```text
输入：很多张大小不同的图
输出：每张图一行 logits，shape = [图数量, 类别数]
```

这叫**图分类**。分子是否具有某种类别标签就是典型的图级任务；连续性质则是图回归。

## 2. MUTAG 在这里扮演什么角色

本日使用公开 [MUTAG 数据](../../../data/public/mutag.md)。官方数据站记录它有 188 张图和 2 个类别。它适合演示小批量图训练，但规模很小，不能用一次随机划分得出“GIN 普遍最好”的结论。

MUTAG 不是下游任务数据。它的图标签、节点标签和实验目标不能直接替代配方、固化工艺或粘接性能。

## 3. 多张不同大小的图怎样组成 batch

普通表格 batch 往往是规则矩阵。图的节点数不同，PyG 不会把每张图补到相同大小，而是：

1. 把多张图的节点特征沿第 0 维拼接；
2. 把边编号加上节点偏移后拼接；
3. 建立 `batch` 向量，记录每个节点属于第几张图。

例如两张图分别有 3、2 个节点：

```text
batch = [0, 0, 0, 1, 1]
```

PyG 返回的是一个 `Batch` 对象，例如 `one_batch`；它的 `one_batch.batch` 才是节点归属向量。`one_batch.batch.shape[0]` 等于当前批次的总节点数，而不是图数；当前图数可用 `one_batch.num_graphs` 或 `one_batch.y.shape[0]` 查看。

## 4. GIN 做什么

GIN 是 Graph Isomorphism Network。一个常见更新写成：

```text
h_i^(k) =
MLP_k(
    (1 + eps_k) * h_i^(k-1)
    + sum_{j in N(i)} h_j^(k-1)
)
```

逐项解释：

- `h_i`：节点 `i` 的表示；
- `N(i)`：节点 `i` 的邻居；
- `sum`：把邻居消息相加；
- `eps`：控制自身表示的权重；
- `MLP`：对“自身＋邻居和”做可学习变换。

本日不要求证明 GIN 的理论表达能力。先能跟踪输入、邻居聚合、节点表示和输出 shape。

## 5. 为什么还需要全局池化

经过 GINConv 后仍然是一行一个节点：

```text
node_hidden.shape = [本批次总节点数, hidden]
```

图标签却是一行一个图，所以要用 `global_add_pool`：

```text
graph_hidden = global_add_pool(node_hidden, batch_vector)
graph_hidden.shape = [本批次图数, hidden]
```

最后的线性层才得到：

```text
logits.shape = [本批次图数, 类别数]
```

如果忘记池化，节点行数与图标签行数无法对应。

## 6. sum、mean、max 并不相同

| 池化 | 直观含义 | 可能特点 |
|---|---|---|
| sum | 所有节点表示相加 | 保留与图大小相关的总量信息 |
| mean | 节点表示取平均 | 更弱化图大小 |
| max | 每一维取最大 | 强调最强响应 |

本日固定使用 sum pooling，避免一边看 validation 一边更换池化方式。正式比较应把池化方式写进预先声明的协议。

## 7. 为什么先保留多数类基线

多数类基线只做一件事：

1. 在训练集统计哪个类别最多；
2. 对所有 validation 图都预测这个类别。

如果 GIN 没有稳定超过这个简单规则，先检查划分、训练和表示，不应直接增加模型复杂度。

多数类只能由训练标签确定；不能先统计 validation 或 test 的多数类。

## 8. 本日的数据边界

- 固定随机划分与随机种子；
- loss 只读取训练图标签；
- validation 用于教学评价；
- test 索引只被保留，不计算或打印 test 指标；
- 固定 epoch，不按历史最好 validation 保存 checkpoint；
- 单次小数据结果不是正式基准或统计显著性证据。

下一步：[算法与数据流推演](02_algorithm_walkthrough.md)。
