# Day 32：GraphSAGE 与邻居聚合

## 今天为什么学

GraphSAGE 强调从邻居中聚合信息，并可结合邻居采样处理更大的图。今天只比较模型思路，不追求刷新分数。

## 前置条件

- 已完成 Day 31；
- GCN 代码可以从空内核运行；
- 能解释一跳邻居和两跳邻居。

## 今日产出

- 把 GCN 层替换为 SAGEConv 的实验；
- 相同协议下的 GCN/GraphSAGE 对照表；
- 一段关于 mean 聚合的直观解释。

## 核心概念

GraphSAGE 的简化理解：

```text
当前节点表示
+ 邻居表示的聚合结果
→ 线性变换
→ 新的节点表示
```

它不等于“把所有邻居直接拼成长向量”。常见做法是平均、池化或其他可学习聚合。

## 核心代码骨架

```python
import torch
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

class SmallGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=0.5, training=self.training)
        return self.conv2(hidden, edge_index)
```

## 分步骤任务

1. 复制 Day 31 的数据与评价协议；
2. 只替换模型类，不改变划分和指标；
3. 检查参数量；
4. 使用相同随机种子训练 GCN 和 GraphSAGE；
5. 至少重复三个种子；
6. 保存均值、标准差和训练时间；
7. 写出“当前对照能说明什么、不能说明什么”。

## 新语法与结构

- `SAGEConv` 与 `GCNConv` 接口相似，但内部聚合不同；
- 相同接口不表示算法相同；
- 替换模型时其他实验条件应保持不变；
- 参数量不同会影响公平解释，要一起报告。

## 常见错误

- 换模型同时又改隐藏层和学习率；
- 单次分数略高就宣布胜出；
- 把“邻居采样”误认为随机删边；
- 没有记录训练时间和参数量。

## 完成标准

- [ ] 两个模型使用相同数据和划分；
- [ ] 我能解释 mean 聚合；
- [ ] 我报告了重复实验；
- [ ] 我没有把结果迁移成粘合剂结论。

## 自测问题

1. 为什么聚合函数需要对邻居顺序不敏感？
2. 邻居很多时，采样有什么计算优势？
3. GraphSAGE 一定比 GCN 好吗？

## 导航

- 上一天：[Day 31 GCN](../day31_gcn/README.md)
- 下一天：[Day 33 GAT](../day33_gat/README.md)
- 总路线：[可选 GNN 课程](../README.md)
