# Day 32：GraphSAGE 与公平模型比较

## 今天为什么学

今天把 Day 31 的 `GCNConv` 替换为 `SAGEConv`，并在完全相同的数据、掩码、隐藏宽度、优化器、训练轮数和随机种子下比较 GCN 与 GraphSAGE。

> 边界：这是 KarateClub 教学图上的 validation 开发比较。它不能证明某个模型适合分子或下游任务，也不能代替最终 test 评价。

## 前置条件

- 已完成 [Day 31](../day31_gcn/README.md)；
- 能独立解释两层 GCN 的 shape；
- 知道一次随机运行不能代表算法稳定性；
- 继续封存 test 节点标签。

## 学习顺序

1. [概念讲义：GraphSAGE、mean 聚合与公平比较](01_concepts.md)
2. [算法推演：用配对种子比较 GCN 与 GraphSAGE](02_algorithm_walkthrough.md)
3. [教学 Notebook：三种种子的 validation mean/std](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天完成后应该能做到

- 解释 GraphSAGE 如何分别处理“自己”和“邻居聚合”；
- 说明邻居聚合为什么必须对邻居顺序不敏感；
- 在不修改其他条件时替换模型；
- 使用至少三种配对随机种子；
- 报告 validation accuracy 的均值、样本标准差和参数量；
- 区分“相同隐藏宽度”和“相同参数预算”；
- 不保存或比较不稳定的墙钟耗时。

## 今日产出

- 每个模型、每个种子的一行 validation 结果；
- 每个模型的 validation mean/std；
- 参数量对照；
- 一段“当前比较能说明与不能说明什么”的文字。

不把墙钟时间持久化到结果中：短小教学图的耗时非常容易被机器、首次导入、线程和后台任务干扰。

## 核心概念

mean GraphSAGE 分别变换节点自身表示和邻居平均表示：

```text
自己表示 × 自身权重
+ mean(邻居表示) × 邻居权重
→ 新的节点表示
```

聚合函数必须对邻居排列不敏感。今天使用完整邻居，没有启用邻居采样。

## 核心代码骨架

```python
class SmallGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        hidden = F.relu(self.conv1(x, edge_index))
        return self.conv2(hidden, edge_index)
```

## 固定比较协议

| 项目 | GCN | GraphSAGE |
|---|---|---|
| 图、特征、标签 | 相同 KarateClub 图 | 相同 KarateClub 图 |
| train/validation/test 掩码 | 相同 | 相同 |
| 隐藏表示宽度 | 16 | 16 |
| 层数 | 2 | 2 |
| 激活 | ReLU | ReLU |
| 外部 Dropout | 0.5 | 0.5 |
| Adam 参数 | `lr=0.01`, `weight_decay=5e-4` | 相同 |
| epoch | 120 | 120 |
| 种子 | `7, 17, 27` | 相同并配对 |
| 开发指标 | validation accuracy | validation accuracy |
| test | 不访问 | 不访问 |

不同算子的内部参数化不同，所以参数量可能不同。我们如实报告，而不是声称这是“严格等参数”比较。

## 分步骤任务

1. 复用 Day 31 的 KarateClub 和固定掩码；
2. 定义输出 shape 相同的 GCN 与 GraphSAGE；
3. 检查前向 shape 和参数量；
4. 固定 `7、17、27` 三种配对种子；
5. 每次创建全新模型与优化器并训练 120 epoch；
6. 保存全部 validation 结果；
7. 汇总均值、样本标准差和参数量；
8. 写出限制，不查看 test。

## 常见错误

- 替换模型时同时改变隐藏宽度、epoch 或掩码；
- 两个模型使用不同随机种子；
- 只报告最好一次；
- 把相同隐藏宽度称为严格等参数；
- 在小教学图上用一次墙钟时间排名；
- 把传导式教学结果说成已证明归纳泛化。

## 完成标准

- [ ] 两个模型严格复用同一掩码；
- [ ] 使用三种相同种子配对运行；
- [ ] 报告每次结果而不只报告最好一次；
- [ ] 报告均值、样本标准差和参数量；
- [ ] Notebook 中没有 test accuracy；
- [ ] 没有声称 GraphSAGE 一定优于 GCN。

## 自测问题

1. mean 聚合为什么不受邻居输入顺序影响？
2. 邻居采样与随机删边有什么区别？
3. 配对随机种子解决了什么混淆？
4. 均值略高但标准差也很大时怎样表述？
5. 为什么本日不能证明对全新图的归纳能力？

## 导航

- 上一天：[Day 31 GCN](../day31_gcn/README.md)
- 下一天：[Day 33 GAT](../day33_gat/README.md)
- 总路线：[可选 GNN 课程](../README.md)
- 学习进度：[一步一步学习目录](../../PROGRESS.md)
