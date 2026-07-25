# Day 33：GAT 与邻居注意力

## 今天为什么学

GCN 和 GraphSAGE 通常按固定规则聚合邻居；GAT 学习不同邻居的权重。注意力权重可以辅助观察模型，但不能自动当作因果机制。

## 前置条件

- 已完成 Day 32；
- 能解释邻居聚合；
- 能区分“模型内部权重”和“实验因果证据”。

## 今日产出

- 一个两层 GAT；
- GCN、GraphSAGE、GAT 公平比较表；
- 对注意力权重解释边界的说明。

## 核心概念

```text
对每条邻居关系计算相关性
→ 在同一节点的邻居之间归一化
→ 按权重聚合邻居
```

多头注意力会并行学习多组权重。`heads` 增加时要特别检查输出维度。

## 核心代码骨架

```python
import torch
from torch.nn import functional as F
from torch_geometric.nn import GATConv

class SmallGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4)
        self.gat2 = GATConv(
            hidden_channels * 4,
            out_channels,
            heads=1,
            concat=False,
        )

    def forward(self, x, edge_index):
        hidden = self.gat1(x, edge_index)
        hidden = F.elu(hidden)
        return self.gat2(hidden, edge_index)
```

## 新语法解释

- 第一层四个头拼接，所以输出维数是 `hidden_channels * 4`；
- `concat=False` 表示最后一层不把多个头继续拼接；
- `F.elu` 是另一种激活函数；
- 多头增加表达能力，也增加参数和计算。

## 分步骤任务

1. 手算第一层输出维数；
2. 先只运行前向传播并检查 shape；
3. 使用 Day 31 相同的数据和掩码；
4. 固定候选 `heads=[1, 2, 4]`；
5. 比较指标、参数量和耗时；
6. 可选：读取一层注意力权重，只描述分布；
7. 明确写出注意力权重不是化学机制证明。

## 常见错误

- 忘记乘以 `heads` 导致下一层 shape 报错；
- 一边增加 heads 一边扩大 hidden size，比较不公平；
- 把高注意力边写成“最重要化学键”；
- 只报告最好一次运行。

## 完成标准

- [ ] 我能算出多头拼接后的维数；
- [ ] 三类模型协议一致；
- [ ] 我能说出注意力解释的限制；
- [ ] 我保存了模型参数和种子。

## 自测问题

1. `heads=4` 为什么不一定带来四倍效果？
2. 注意力权重是输入前就固定的吗？
3. 高注意力是否等于因果重要性？

## 导航

- 上一天：[Day 32 GraphSAGE](../day32_graphsage/README.md)
- 下一天：[Day 34 GIN 与图分类](../day34_gin_graph_classification/README.md)
- 总路线：[可选 GNN 课程](../README.md)
