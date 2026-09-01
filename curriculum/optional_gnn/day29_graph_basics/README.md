# Day 29：图、节点、边与邻居聚合

> 可选 GNN 路线。只有通过 [GNN 启动条件](../README.md) 后才开始。本日只使用 NumPy 理解图，不安装 PyTorch Geometric，也不训练模型。

## 今天为什么学

普通表格通常把一个样本写成一行固定长度特征。图数据除了数值特征，还必须表达“哪些对象彼此相连”。

如果还分不清节点、边、邻接矩阵和节点特征，就很难理解 GNN 的消息传递。今天先用同一张四节点玩具图把这些对象拆开。

## 前置条件

- 已完成 Day 1–28；
- 能解释二维数组的行、列与 `shape`；
- 能看懂 `for` 循环和列表；
- 已阅读 [GNN 启动条件](../README.md)；
- 知道这个玩具图不是分子数据或下游任务实验数据。

## 完整学习包（按顺序）

1. [概念：图数据的五个基本对象](01_concepts.md)
2. [算法推演：从边列表到邻居消息求和](02_algorithm_walkthrough.md)
3. [课程提供的可运行 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

`tutorial.ipynb` 是课程提供的教学示例，不代表你已经完成 Day 29。真正学习时，应把个人副本和自己的解释保存到 `learning_outputs/day29_graph_basics/`。

## 今天的目标

完成后应能说明：

1. 节点、边、节点特征和整图标签分别是什么；
2. 四个节点的邻接矩阵为什么是 `(4, 4)`；
3. 无向边为什么需要同时记录两个方向；
4. `degree` 怎样从邻接矩阵计算；
5. `adjacency @ x` 为什么等于“把邻居特征加起来”；
6. 为什么这还不是一个可训练的 GNN。

## 今日产出

以下内容应由你在个人练习目录中完成：

1. 一张手画的四节点无向图；
2. 对应的边列表和邻接矩阵；
3. 每个节点的邻居列表与度数；
4. 一次邻居特征求和的纸笔核对；
5. 一段“表格样本与图样本有什么不同”的中文说明。

不要把课程 Notebook 中预存的输出当成个人练习结果。

## 核心概念

| 对象 | 本日 shape 或形式 | 含义 |
|---|---|---|
| 节点列表 | 长度 4 | A、B、C、D 四个对象 |
| 无向边列表 | 4 个端点元组 | 哪两个节点直接相连 |
| 邻接矩阵 | `(4, 4)` | 行节点与列节点是否相连 |
| 节点特征 | `(4, 2)` | 每个节点一行、两个教学特征 |
| 度数 | `(4,)` | 每个节点连接的邻居数量 |
| 邻居消息之和 | `(4, 2)` | 每个节点收到的邻居特征总和 |

今日数据流：

```text
四个节点 + 四条无向边
→ 邻接矩阵 adjacency，shape = (4, 4)
→ 每行求和得到 degree，shape = (4,)
→ adjacency @ x
→ 每个节点收到的邻居特征之和，shape = (4, 2)
```

## 核心代码骨架

```python
import numpy as np

node_names = ["A", "B", "C", "D"]
undirected_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

adjacency = np.zeros((len(node_names), len(node_names)), dtype=int)
for source, target in undirected_edges:
    adjacency[source, target] = 1
    adjacency[target, source] = 1

node_features = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=float,
)

degree = adjacency.sum(axis=1)
neighbor_message_sum = adjacency @ node_features

print("adjacency shape:", adjacency.shape)
print("node_features shape:", node_features.shape)
print("degree:", degree)
print("neighbor_message_sum:\n", neighbor_message_sum)
```

## 分步骤任务

1. 在纸上画出 A、B、C、D 和四条无向边；
2. 不运行代码，先手写 `(4, 4)` 邻接矩阵；
3. 用 NumPy 创建矩阵并同时写入每条边的两个方向；
4. 检查矩阵 shape、对称性与对角线；
5. 写出每个节点的邻居并计算度数；
6. 手算 A 节点收到的邻居特征之和；
7. 用 `adjacency @ node_features` 核对所有节点；
8. 用自己的话说明为什么这还不是经过训练的 GNN。

## 常见错误

- 无向边只写一个方向，邻接矩阵因此不对称；
- 把 `(4, 4)` 邻接矩阵误认为 `(4, 2)` 节点特征；
- 把节点的编号 `0、1、2、3` 当成节点特征；
- 认为矩阵相乘已经包含可学习参数；
- 把邻居之和误说成节点自己的特征；
- 把玩具图结论解释成分子或下游任务研究结论。

## 完成标准

- [ ] 能从边列表独立写出邻接矩阵；
- [ ] 能说出 `axis=1` 为什么是逐行求和；
- [ ] 能手算一个节点的 `degree`；
- [ ] 能手算一个节点收到的邻居消息之和；
- [ ] 能区分节点特征、边结构和整图标签；
- [ ] 明确今天没有训练模型，也没有使用真实下游任务数据。

## 自测问题

1. 为什么四个节点的邻接矩阵不是 `(4, 2)`？
2. 无向边 `(0, 1)` 要修改邻接矩阵哪两个位置？
3. `degree.shape == (4,)` 中右边为什么没有第二个数字？
4. `adjacency @ node_features` 的两个中间维度为什么必须相等？
5. 固定邻居求和与可训练 GNN 的区别是什么？

## 导航

- 上一天：[Day 28 下游任务接入与总报告](../../core/day28_capstone_handoff/README.md)
- 开始本日前必须通过：[GNN 启动条件](../README.md)
- 下一天：[Day 30 PyTorch Geometric Data](../day30_pyg_data/README.md)
- 总路线：[可选 GNN 课程](../README.md)
- 学习目录：[返回一步一步学习目录](../../PROGRESS.md)
