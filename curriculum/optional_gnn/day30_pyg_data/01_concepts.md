# Day 30.1：`Data` 字段、shape、dtype 与标签语义

今天不训练神经网络，只学习如何把一张图可靠地交给 PyTorch Geometric。

## 1. `Data` 是容器，不是模型

```python
from torch_geometric.data import Data

graph = Data(x=x, edge_index=edge_index, y=y)
```

这行代码的动作只是把三个已经存在的张量保存为字段：

```text
graph.x
graph.edge_index
graph.y
```

它不会自动：

- 判断特征有没有化学意义；
- 补全无向边的反方向；
- 训练 GNN；
- 计算损失或评价指标；
- 判断 `y` 是节点标签还是整图标签。

这些都要由数据定义和后续代码明确说明。

## 2. 张量与 NumPy 数组

`torch.tensor(...)` 把 Python 列表转换为 PyTorch 张量。

```python
import torch

example = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
```

与 NumPy 数组一样，张量有：

- `shape`：各维度长度；
- `dtype`：元素的数据类型；
- 索引与切片。

张量还可以参与自动求导和在 CPU/GPU 间移动，但今天不使用这些训练功能。

## 3. `x`：节点特征矩阵

Day 29 的节点特征改写成张量：

```python
x = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=torch.float32,
)
```

`x.shape == (4, 2)`：

- 4 行：4 个节点；
- 2 列：每个节点的两个数值特征；
- `x[0]` 是节点 A 的特征；
- `x.dtype == torch.float32`，适合后续神经网络数值计算。

这些数字只是教学特征，不对应真实原子描述符。

## 4. `edge_index`：稀疏边表示

Day 29 的四条无向边：

```text
0—1、1—2、2—3、3—0
```

PyG 常用两行表示方向记录：

```python
edge_index = torch.tensor(
    [
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ],
    dtype=torch.long,
)
```

按列读，而不是按行读：

| 列 | 来源节点 | 目标节点 |
|---:|---:|---:|
| 0 | 0 | 1 |
| 1 | 1 | 0 |
| 2 | 1 | 2 |
| 3 | 2 | 1 |
| 4 | 2 | 3 |
| 5 | 3 | 2 |
| 6 | 3 | 0 |
| 7 | 0 | 3 |

所以 `edge_index.shape == (2, 8)`：

- 第 0 行保存所有来源节点；
- 第 1 行保存所有目标节点；
- 8 列是 8 个方向记录；
- 4 条无向边各有正反两个方向。

## 5. 为什么 `edge_index` 必须是整数

节点编号用于索引。`0.5` 不能表示“第半个节点”，所以 PyG 通常要求：

```text
edge_index.dtype == torch.long
```

在常见系统上，`torch.long` 对应 64 位整数，也会显示为 `torch.int64`。二者是同一种 dtype 的常用名称。

## 6. `y`：先说明任务，再解释 shape

本日设置：

```python
y = torch.tensor([0.75], dtype=torch.float32)
```

我们明确把它定义为“一张图对应一个连续目标”，所以：

```text
y.shape == (1,)
```

表示这张图有一个整图回归标签。它不是四个节点标签。

如果是四节点分类任务，`y` 可能有 4 项；如果是整图多任务回归，`y` 也可能有多个目标。因此只看变量名 `y` 不够，必须同时记录任务、shape 和单位。

## 7. `num_nodes` 从哪里来

创建：

```python
graph = Data(x=x, edge_index=edge_index, y=y)
```

当 `x` 存在时，PyG 可以从 `x.shape[0]` 推断：

```text
graph.num_nodes == 4
```

若没有节点特征，自动推断可能依赖边中出现的最大编号；孤立节点可能因此被遗漏。真实任务中应明确确认节点数，而不是无条件相信自动推断。

## 8. 严格检查清单

### Shape

```python
assert tuple(graph.x.shape) == (4, 2)
assert tuple(graph.edge_index.shape) == (2, 8)
assert tuple(graph.y.shape) == (1,)
assert graph.num_nodes == 4
```

### Dtype

```python
assert graph.x.dtype == torch.float32
assert graph.edge_index.dtype == torch.long
assert graph.y.dtype == torch.float32
```

### 节点编号范围

```python
assert int(graph.edge_index.min()) >= 0
assert int(graph.edge_index.max()) < graph.num_nodes
```

### 无向边双向完整性

```python
directed_pairs = {
    (int(source), int(target))
    for source, target in graph.edge_index.T.tolist()
}

for source, target in directed_pairs:
    assert (target, source) in directed_pairs
```

`.T` 把 `edge_index` 从 `(2, 8)` 转成 `(8, 2)`，这样每一行就恰好是一对端点，方便遍历。

## 9. `edge_attr` 为什么今天没有

真实分子图通常还需要边特征，例如键类型。PyG 可以保存：

```text
graph.edge_attr
```

但 Day 29 的玩具图只表达“是否连接”，没有定义边特征。与其随便编造化学键类型，不如明确保留缺失，并在后续分子图课程再学习。

## 10. 今日证据边界

今天验证的是数据结构是否自洽，不是模型性能。它不能说明：

- PyG 已经正确表示真实分子；
- GNN 已经学到任何规则；
- 玩具标签来自实验；
- GNN 适合当前下游任务任务。

下一步：[把同一张图逐字段装入 `Data`](02_algorithm_walkthrough.md)。
