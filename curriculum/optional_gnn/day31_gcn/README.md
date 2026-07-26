# Day 31：第一个 GCN 节点分类

## 今天为什么学

GCN 把节点自己的特征与邻居信息结合。今天使用公开的教学图数据理解消息传递，不接入真实粘合剂数据。

## 前置条件

- 已完成 Day 30；
- 能解释 `x` 与 `edge_index`；
- PyTorch/PyG 环境已经验证；
- 能区分训练掩码和测试掩码。

## 今日产出

- 一个两层 GCN 的结构记录；
- 训练 loss 曲线；
- 训练/验证/测试掩码用途说明；
- `notes.md` 中的消息传递复述。

## 今天的目标

1. 看懂 `GCNConv` 的输入和输出；
2. 理解节点嵌入会聚合邻居信息；
3. 知道 `forward` 只定义前向计算；
4. 不把节点分类结果当成分子性质预测。

## 核心概念

GCN 的核心动作是：让每个节点把自己的信息与邻居信息做归一化聚合，再通过可学习权重得到新的节点表示。`GCNConv` 同时需要节点特征 `x` 和连接关系 `edge_index`。

## 核心代码骨架

```python
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class SmallGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        hidden = self.conv1(x, edge_index)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=0.5, training=self.training)
        return self.conv2(hidden, edge_index)
```

## 新语法解释

- `class SmallGCN(...)` 定义模型类型；
- `super().__init__()` 初始化父类；
- `self.conv1` 把一层保存到模型对象；
- `forward` 接收节点特征和边；
- `training=self.training` 让 Dropout 只在训练模式启用。

## 分步骤任务

1. 使用 PyG 官方教学数据，例如 Karate Club；
2. 打印节点数、边数、输入特征数和类别数；
3. 创建两层 GCN；
4. 运行一次前向传播，只检查输出 shape；
5. 再编写训练循环；
6. loss 只能在训练掩码上计算；
7. 验证指标不参与 `backward()`；
8. 保存每个 epoch 的 loss 和验证准确率。

## 必须画出的数据流

```text
节点特征 x + 边 edge_index
→ GCNConv
→ ReLU
→ Dropout
→ GCNConv
→ 每个节点的类别分数
```

## 常见错误

- 忘记把模型切换到 `train()` 或 `eval()`；
- 在所有节点上计算训练 loss；
- 把 `out_channels` 写成节点数；
- 输出 shape 不检查就直接算 loss；
- 把公开社交图结果解释为材料结果。

## 完成标准

- [ ] 我能说出每层输入/输出 shape；
- [ ] 我能解释训练掩码；
- [ ] 我能说明 GCN 如何使用邻居；
- [ ] 我保存了曲线而不是只看最后一个数字。

## 自测问题

1. 没有 `edge_index`，模型还能聚合邻居吗？
2. 两层 GCN 大致能接收几跳邻居的信息？
3. 节点分类和整图回归的输出有什么不同？

## 导航

- 上一天：[Day 30 PyG Data](../day30_pyg_data/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 32 GraphSAGE](../day32_graphsage/README.md)
- 总路线：[可选 GNN 课程](../README.md)
