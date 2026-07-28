# Day 33：GAT、公平比较与注意力边界

## 今天为什么学

今天加入图注意力网络 GAT。在 Day 31–32 完全相同的 KarateClub 开发协议下，用三种配对种子比较 GCN、GraphSAGE 与 GAT，并读取一小段第一层注意力权重用于理解张量结构。

> 注意力权重是模型内部、依赖输入与参数的关联权重，不是因果证据，也不能自动解释为“最重要的化学键”。

## 前置条件

- 已完成 [Day 32](../day32_graphsage/README.md)；
- 能解释邻居聚合和参数量；
- 知道多次 validation 运行仍不是最终 test；
- 能沿着 `[N,F] → [N,H] → [N,C]` 检查 shape。

## 学习顺序

1. [概念讲义：图注意力、多头 shape 与解释边界](01_concepts.md)
2. [算法推演：公平比较并读取有界注意力权重](02_algorithm_walkthrough.md)
3. [教学 Notebook：GCN/SAGE/GAT 三种种子比较](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天完成后应该能做到

- 解释 GAT 如何在一个节点的邻居之间归一化权重；
- 手算多头拼接后的输出宽度；
- 在相同数据、掩码、宽度、epoch 和种子下比较三类模型；
- 报告每次 validation、mean/std 和参数量；
- 从 `GATConv` 读取 `edge_index` 与 `alpha` 的 shape；
- 只展示有限行注意力权重，避免输出淹没 Notebook；
- 明确“注意力不是解释”“关联不是因果”。

## 今日产出

- 三模型 × 三种种子的完整 validation 表；
- 每个模型的 mean/std 与参数量；
- 第一层注意力返回值的 shape；
- 最多 8 行注意力预览；
- 一段关于注意力解释限制的文字。

## 核心概念

GAT 为同一目标节点的不同邻居边计算输入相关的归一化权重。第一层使用两个头：

```text
每头宽度 8 × 2 个头
→ concat 后隐藏总宽度 16
→ 第二层输出 [N,C]
```

权重依赖当前输入、参数、层、头和随机种子，不是固定边属性。

## 核心代码骨架

```python
hidden, (attention_edges, alpha) = gat_layer(
    x,
    edge_index,
    return_attention_weights=True,
)

preview_rows = min(8, attention_edges.shape[1])
bounded_alpha = alpha[:preview_rows]
```

## 固定比较协议

| 项目 | 固定设置 |
|---|---|
| 数据与掩码 | 与 Day 31–32 完全相同 |
| 隐藏表示总宽度 | 16 |
| 层数 | 2 |
| 外部 Dropout | 0.5 |
| Adam | `lr=0.01`, `weight_decay=5e-4` |
| epoch | 120 |
| 配对种子 | `7, 17, 27` |
| 开发指标 | validation accuracy |
| test | 继续封存 |
| GAT 第一层 | `heads=2`，每头宽度 8，拼接后总宽度 16 |
| GAT 第二层 | `heads=1`, `concat=False` |

为了让三类模型的隐藏表示总宽度一致，GAT 第一层使用 `8 × 2 = 16`。不同算子的参数量仍不相同，因此一起报告。

## 分步骤任务

1. 复用 Day 31–32 的图、掩码和训练协议；
2. 定义 GCN、GraphSAGE 与双头 GAT；
3. 检查 GAT 拼接前后 shape；
4. 三个模型都运行 `7、17、27`；
5. 汇总 validation mean/std 与参数量；
6. 预先指定 seed 7 的 GAT 做注意力结构预览；
7. 只显示前 8 条返回边，不按权重排序；
8. 写清注意力不是因果，不读取 test。

## 常见错误

- 忘记 `heads × out_channels` 导致下一层 shape 错误；
- 给 GAT 更多 epoch 后仍称为相同预算；
- 只展示最好种子的 GAT；
- 假设注意力返回边与原始边严格一一对应；
- 把高注意力边称为因果边或重要化学键；
- 用 validation 比较后又提前查看 test 排名。

## 完成标准

- [ ] 三个模型使用同一实验协议；
- [ ] GAT 第一层拼接输出确实是 `[N,16]`；
- [ ] 没有只挑选最好的种子；
- [ ] 注意力输出被限制为最多 8 行；
- [ ] 没有按注意力权重提出因果结论；
- [ ] 没有读取或报告 test accuracy。

## 自测问题

1. `out_channels=8, heads=2, concat=True` 的输出宽度是多少？
2. `alpha.shape == [E_attention,2]` 的两个轴是什么？
3. 返回边数为什么可能大于原始边数？
4. 为什么只预览 8 行且不排序？
5. 注意力权重与因果证据之间缺少哪些环节？

## 导航

- 上一天：[Day 32 GraphSAGE](../day32_graphsage/README.md)
- 下一天：[Day 34 GIN 与图分类](../day34_gin_graph_classification/README.md)
- 总路线：[可选 GNN 课程](../README.md)
- 学习进度：[一步一步学习目录](../../PROGRESS.md)
