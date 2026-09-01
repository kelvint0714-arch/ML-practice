# Day 31：第一个 GCN 节点分类

## 今天为什么学

今天在 PyTorch Geometric 的 `KarateClub` 教学图上完成一个两层 GCN。重点不是追求分数，而是第一次把“图数据 → 消息传递 → 节点类别分数 → 只在训练节点上计算 loss”完整走通。

> 边界：KarateClub 是公开的社交网络教学数据，不是分子图，更不是下游任务实验数据。今天得到的结果只能证明代码和概念能够运行。

## 前置条件

- 已完成 Day 29–30；
- 能说出 `x`、`edge_index`、`y` 和布尔掩码的含义；
- 当前 Python 环境已经安装 `torch` 与 `torch_geometric`；
- 接受本日只使用 validation 观察学习过程，test 标签继续封存。

环境检查：

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

## 学习顺序

严格按下列顺序完成：

1. [概念讲义：GCN、shape、mask 与训练模式](01_concepts.md)
2. [算法推演：两层 GCN 如何完成一次训练](02_algorithm_walkthrough.md)
3. [教学 Notebook：从前向传播到 validation 曲线](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天完成后应该能做到

- 沿着代码说出 `x → hidden → logits` 的 shape；
- 解释一层 GCN 如何把邻居信息聚合到节点；
- 解释为什么 loss 只能用 `train_mask` 中的节点；
- 区分 `model.train()`、`model.eval()` 和 `torch.no_grad()`；
- 统计模型可训练参数量；
- 知道 validation 用于开发观察，而 test 仍不能查看。

## 今日产出

以下内容应由你在个人练习副本中完成：

1. `x → hidden → logits` 的 shape 记录；
2. train/validation/test 掩码互斥检查；
3. 两层 GCN 的参数量；
4. 120 个 epoch 的 train loss 与 validation accuracy 曲线；
5. 一段“为什么 test 仍要封存”的中文说明。

课程 Notebook 的预存输出是教师教学示例，不代表你的个人练习成果。

## 核心概念

```text
x [N,F] + edge_index [2,E]
→ GCNConv(F,16)
→ ReLU
→ Dropout
→ GCNConv(16,C)
→ logits [N,C]
```

loss 只使用 `logits[train_mask]` 与 `y[train_mask]`。validation 只评价，不反向传播；test 标签 Day 31–33 全程封存。

## 核心代码骨架

```python
model.train()
optimizer.zero_grad()
logits = model(data.x, data.edge_index)
loss = F.cross_entropy(
    logits[train_mask],
    data.y[train_mask],
)
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():
    valid_logits = model(data.x, data.edge_index)
```

## 固定实验协议

后续 Day 32–33 也沿用这套协议：

| 项目 | 固定值 |
|---|---|
| 数据 | PyG `KarateClub` |
| 训练掩码 | 数据集自带的 `train_mask` |
| validation/test 划分 | 对其余节点按固定掩码种子随机平分 |
| 掩码种子 | `20260728` |
| 隐藏表示宽度 | `16` |
| 优化器 | Adam，`lr=0.01`，`weight_decay=5e-4` |
| 外部 Dropout | `0.5` |
| 训练轮数 | `120` |
| 开发指标 | validation accuracy |
| test | Day 31–33 全程封存 |

固定协议不代表这是最优超参数。它只是让初学者能进行可解释的公平比较。

## 分步骤任务

1. 加载 KarateClub 并打印 `x`、`edge_index`、`y` 的 shape；
2. 建立固定且互斥的三个布尔掩码；
3. 定义两层 GCN；
4. 先运行一次前向传播并检查 `[N,C]`；
5. 统计可训练参数量；
6. 训练 120 个 epoch，只在训练节点计算 loss；
7. 用评价模式记录 validation accuracy；
8. 画曲线并写清证据边界，不读取 test 标签。

## 常见错误

- 在所有节点上计算交叉熵；
- 把 `out_channels` 误写成节点数；
- 评价时忘记 `model.eval()`；
- 把 `eval()` 误认为自动关闭梯度；
- 查看 test accuracy 后继续调参；
- 把社交网络教学结果解释成材料结论。

## 完成标准

- [ ] Notebook 能从头运行到尾；
- [ ] 所有 shape 检查通过；
- [ ] 三种掩码互斥且覆盖全部节点；
- [ ] loss 只在训练节点上计算；
- [ ] 画出 train loss 与 validation accuracy 曲线；
- [ ] 没有读取或报告 test accuracy；
- [ ] 独立完成练习后才查看答案。

## 自测问题

1. `[N,F]` 中每个轴分别是什么？
2. 为什么 logits 的 shape 是 `[N,C]`？
3. `train_mask` 怎样控制哪些标签进入梯度？
4. `model.eval()` 与 `torch.no_grad()` 有什么不同？
5. 两层消息传递大致能融合几跳邻域？

## 导航

- 上一天：[Day 30 PyG Data](../day30_pyg_data/README.md)
- 下一天：[Day 32 GraphSAGE](../day32_graphsage/README.md)
- 总路线：[可选 GNN 课程](../README.md)
- 学习进度：[一步一步学习目录](../../PROGRESS.md)
