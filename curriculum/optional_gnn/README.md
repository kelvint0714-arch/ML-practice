# Day 29–35：可选 GNN 路线

这条路线用于学习图表示和图神经网络，不是核心课程的强制前置。建议先完成 Day 15–21，理解张量、训练循环和公平比较。

## 环境

```bash
conda create -n gnn python=3.10.20 -y
conda activate gnn
python -m pip install -r requirements-gnn.txt
python -m ipykernel install --user --name gnn --display-name "Python 3 (gnn)"
```

## 路线

| Day | 主题 | 任务卡 |
|---:|---|---|
| 29 | 图、节点、边和邻接关系 | [Day 29](day29_graph_basics/README.md) |
| 30 | PyTorch Geometric `Data` | [Day 30](day30_pyg_data/README.md) |
| 31 | GCN | [Day 31](day31_gcn/README.md) |
| 32 | GraphSAGE | [Day 32](day32_graphsage/README.md) |
| 33 | GAT | [Day 33](day33_gat/README.md) |
| 34 | GIN 与图分类 | [Day 34](day34_gin_graph_classification/README.md) |
| 35 | GNN 数据就绪评审 | [Day 35](day35_molecular_graph_gate/README.md) |

Day 29–34 使用玩具图、KarateClub 和 MUTAG 学习算法。Day 35 不训练新模型，而是检查一个新数据集是否真的适合 GNN。

## GNN 数据就绪条件

| 检查项 | 最低要求 |
|---|---|
| 图对象 | 清楚一张图代表什么 |
| 节点与边 | 特征和关系具有明确含义 |
| 标签 | 目标、单位和评价方式一致 |
| 划分 | 重复、近重复或同组对象不会跨集合泄漏 |
| 样本量 | 足以支持模型复杂度和稳定评价 |
| 简单基线 | Dummy、传统特征模型或 MLP 已按相同协议比较 |
| 输入公平 | 清楚记录每个模型实际获得的信息 |

条件不满足时，No-Go 是合理结论。更复杂的模型不是默认更好的模型。

## 公平比较

如果 GNN 获得了图关系等额外信息，必须明确说明；不能把信息更丰富的模型与缺少关键输入的弱基线直接称为公平比较。

至少保留 Dummy、简单特征模型、MLP（适用时）、GNN、消融和多随机种子结果。

返回 [核心课程](../core/README.md) 或 [唯一学习清单](../PROGRESS.md)。
