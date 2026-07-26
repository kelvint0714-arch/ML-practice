# Day 34：GIN、图池化与整图预测

## 今天为什么学

前几天预测的是每个节点的类别。分子性质通常是整张分子图的目标，所以需要把所有节点表示汇总成一个图表示，再预测整图标签。

## 前置条件

- 已完成 Day 33；
- 能区分节点分类和图分类；
- 能解释 Batch 中包含多张图。

## 今日产出

- 一个公开小型图分类任务的 GIN；
- 节点表示到图表示的数据流图；
- 图级输出 shape 检查；
- 训练/验证结果记录。

## 核心概念

```text
每张图的节点特征与边
→ 多层 GINConv
→ 每个节点的新表示
→ global pooling
→ 每张图一个向量
→ 线性输出层
→ 图标签或图性质
```

## 核心代码骨架

```python
import torch
from torch_geometric.nn import GINConv, global_add_pool

def make_mlp(in_channels, hidden_channels):
    return torch.nn.Sequential(
        torch.nn.Linear(in_channels, hidden_channels),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_channels, hidden_channels),
    )

class SmallGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = GINConv(make_mlp(in_channels, hidden_channels))
        self.output = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        node_hidden = self.conv(x, edge_index)
        graph_hidden = global_add_pool(node_hidden, batch)
        return self.output(graph_hidden)
```

## 新语法解释

- `Sequential` 把多个层按顺序连接；
- `batch` 表示每个节点属于哪一张图；
- `global_add_pool` 把同一张图的节点向量相加；
- 输出第一维应等于当前 batch 中图的数量。

## 分步骤任务

1. 选择公开小数据，例如 MUTAG；
2. 打印单张图和一个 batch 的字段；
3. 检查 `batch` 向量；
4. 运行前向传播并核对图级输出 shape；
5. 编写训练/验证循环；
6. 与一个简单图特征基线比较；
7. 记录图数量、平均节点数和类别分布。

## 常见错误

- 忘记 pooling，直接把节点输出当图输出；
- 把节点数当 batch 中图的数量；
- 随机划分时没有固定种子；
- 用分类指标解释回归任务，或反过来；
- 认为 GIN 在 MUTAG 上有效就能预测粘合剂。

## 完成标准

- [ ] 我能解释 `batch` 向量；
- [ ] 我能画出节点到整图预测的流程；
- [ ] 输出一行对应一张图；
- [ ] 我保留了简单基线。

## 自测问题

1. 为什么分子性质预测通常需要 pooling？
2. sum、mean、max pooling 会保留相同信息吗？
3. 同一 batch 中的图必须有相同节点数吗？

## 导航

- 上一天：[Day 33 GAT](../day33_gat/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 35 分子图与粘合剂启用评审](../day35_molecular_graph_gate/README.md)
- 总路线：[可选 GNN 课程](../README.md)
