# Day 30：认识 PyTorch Geometric 的 `Data`

> 可选 GNN 路线。今天只把 Day 29 的同一张四节点图装入 `Data` 容器，不定义模型、不计算 loss、不训练。

## 今天为什么学

PyTorch Geometric（PyG）通常使用 `Data` 对象保存一张图。它把节点特征、边索引和标签放在同一个容器中，但不会自动替你确认 shape、dtype、标签语义或无向边是否完整。

今天的重点是：看到 `Data(x=..., edge_index=..., y=...)` 时，能够逐字段读懂并严格检查。

## 前置条件

- 已完成 [Day 29 图基础](../day29_graph_basics/README.md)；
- 已通过 [GNN 启动条件](../README.md)；
- 能解释数组与张量的 `shape`；
- 能解释四条无向边为什么对应八个方向记录；
- 已在独立 GNN 环境中验证 `torch` 和 `torch_geometric` 可以导入。

## 完整学习包（按顺序）

1. [概念：`Data` 字段、shape、dtype 与标签语义](01_concepts.md)
2. [算法推演：把 Day 29 的图逐字段装入 `Data`](02_algorithm_walkthrough.md)
3. [课程提供的可运行 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

`tutorial.ipynb` 是课程提供的教学示例，不代表你已经完成 Day 30。真正学习时，应把个人副本与自己的字段解释保存到 `learning_outputs/day30_pyg_data/`。

## 环境检查

依赖版本变化较快。开始当天应按 PyTorch 与 PyG 官方说明准备独立环境，然后运行：

```python
import sys
import torch
import torch_geometric

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
```

如果导入失败，就停止执行本日 Notebook，记录具体错误，再按当时的官方安装说明处理。不要把失败的导入注释掉后继续假装教程已通过。

## 今天的目标

完成后应能：

1. 解释 `x`、`edge_index`、`y` 与 `num_nodes`；
2. 解释 `x.shape == (4, 2)`；
3. 解释 `edge_index.shape == (2, 8)`；
4. 说明 `edge_index` 两行分别保存什么；
5. 检查无向边的反方向是否全部存在；
6. 解释为什么 `x` 使用浮点、`edge_index` 使用整数；
7. 明确 `Data` 是数据容器，不是模型。

## 今日产出

以下内容应由你在个人练习目录中完成：

1. 一个四节点 `Data` 对象；
2. 一张字段、shape、dtype、语义检查表；
3. 八个方向记录与四条无向边的对应关系；
4. 一段解释 `y=[0.75]` 为什么是整图回归标签；
5. 一组能从空内核运行的断言。

## 核心概念

| 字段 | 本日 shape | 本日 dtype | 含义 |
|---|---|---|---|
| `x` | `(4, 2)` | `torch.float32` | 四个节点、每个节点两个教学特征 |
| `edge_index` | `(2, 8)` | `torch.long` | 八个方向记录，每列是一对端点 |
| `y` | `(1,)` | `torch.float32` | 一张图的一个连续教学标签 |
| `num_nodes` | 整数 4 | Python 整数 | 图中的节点数 |

今日数据流：

```text
Day 29 的节点特征 + 四条无向边 + 一个整图标签
→ torch.tensor
→ x: (4, 2), float32
→ edge_index: (2, 8), int64
→ y: (1,), float32
→ Data(x=x, edge_index=edge_index, y=y)
→ 检查 num_nodes、shape、dtype、编号范围和反向边
```

## 核心代码骨架

```python
import torch
from torch_geometric.data import Data

x = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=torch.float32,
)

edge_index = torch.tensor(
    [
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ],
    dtype=torch.long,
)

y = torch.tensor([0.75], dtype=torch.float32)
graph = Data(x=x, edge_index=edge_index, y=y)

print(graph)
print("x:", graph.x.shape, graph.x.dtype)
print("edge_index:", graph.edge_index.shape, graph.edge_index.dtype)
print("y:", graph.y.shape, graph.y.dtype)
print("num_nodes:", graph.num_nodes)
```

这里的 `0.75` 只是人为构造的教学标签，不是实验测量值。

## 分步骤任务

1. 在独立 GNN 环境中记录 Python、PyTorch 与 PyG 版本；
2. 用 `torch.float32` 创建节点特征 `x`；
3. 把四条无向边明确展开为八个方向记录；
4. 用 `torch.long` 创建 `(2, 8)` 的 `edge_index`；
5. 创建一个 `(1,)` 的整图连续教学标签；
6. 使用关键字参数创建 `Data`；
7. 检查字段、shape、dtype、节点数与编号范围；
8. 检查每个方向记录都有反方向；
9. 解释为什么结构检查通过仍不等于模型已经训练。

## 常见错误

- 把 `edge_index` 写成 `[边方向数, 2]`；
- 节点编号使用浮点 dtype；
- 无向边只记录一个方向；
- `x` 的行数与节点数不一致；
- 标签有一个值，却说成四个节点各有一个标签；
- 只打印 `Data`，不检查各字段 shape 和 dtype；
- 认为 `Data` 创建完成就等于模型已经训练；
- 把玩具标签解释成真实下游任务性能。

## 完成标准

- [ ] `graph.num_nodes == 4`；
- [ ] `x.shape == (4, 2)` 且 dtype 为 `torch.float32`；
- [ ] `edge_index.shape == (2, 8)` 且 dtype 为 `torch.long`；
- [ ] `y.shape == (1,)` 且 dtype 为 `torch.float32`；
- [ ] 所有节点编号都在 0–3；
- [ ] 每个方向记录都有反方向；
- [ ] 明确本日没有训练模型、没有产生科研性能结论。

## 自测问题

1. 为什么 `edge_index` 的第一维通常是 2？
2. 四条无向边为什么在这里对应八列？
3. `torch.long` 和 `torch.float32` 分别解决什么问题？
4. 仅看到 `y.shape == (1,)`，能否证明它来自真实实验？
5. `Data` 创建成功后，为什么仍要检查语义与数据来源？

## 导航

- 上一天：[Day 29 图基础](../day29_graph_basics/README.md)
- 下一天：[Day 31 GCN](../day31_gcn/README.md)
- 总路线：[可选 GNN 课程](../README.md)
- 学习目录：[返回一步一步学习目录](../../PROGRESS.md)
