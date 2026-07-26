# Day 30：认识 PyTorch Geometric 的 `Data`

> 可选 GNN 路线。依赖版本变化较快，开始当天应核对 PyTorch 与 PyG 官方安装说明，不使用本任务卡中的旧版本猜测。

## 今天为什么学

PyTorch Geometric 通常用一个 `Data` 对象保存图。今天只认识数据结构，不训练 GNN。

## 前置条件

- 已完成 Day 29；
- 已通过 GNN 启动条件；
- 能解释数组的 shape；
- 已在独立环境中验证 `torch` 和 `torch_geometric` 可以导入。

## 环境检查

开始当天运行：

```python
import sys
import torch
import torch_geometric

print(sys.executable)
print(torch.__version__)
print(torch_geometric.__version__)
```

如果导入失败，停止课程并按当时官方文档安装；不要连续尝试来历不明的命令。

## 今日产出

- 一个四节点 `Data` 对象；
- `x`、`edge_index`、`y` 的 shape 表；
- 一段说明：每个张量中的一行或一列代表什么。

## 核心概念

| 字段 | 含义 | 常见 shape |
|---|---|---|
| `x` | 节点特征 | `[节点数, 节点特征数]` |
| `edge_index` | 边的两个端点 | `[2, 边方向数]` |
| `edge_attr` | 边特征，可选 | `[边方向数, 边特征数]` |
| `y` | 节点或整图标签 | 取决于任务 |

## 核心代码骨架

```python
import torch
from torch_geometric.data import Data

x = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
    dtype=torch.float,
)

edge_index = torch.tensor(
    [[0, 1, 1, 2, 2, 3, 3, 0],
     [1, 0, 2, 1, 3, 2, 0, 3]],
    dtype=torch.long,
)

y = torch.tensor([1], dtype=torch.long)
graph = Data(x=x, edge_index=edge_index, y=y)

print(graph)
print(graph.x.shape)
print(graph.edge_index.shape)
print(graph.y.shape)
```

## 新语法解释

- `torch.tensor(...)` 把 Python 数值转换为张量；
- 浮点特征使用 `torch.float`；
- 节点编号使用整数类型 `torch.long`；
- `Data(x=..., edge_index=..., y=...)` 使用关键字参数保存字段；
- `edge_index` 第一行是来源节点，第二行是目标节点。

## 分步骤任务

1. 对照 Day 29 的图检查八个有向记录；
2. 打印三个 shape，并翻译成中文；
3. 增加第三个节点特征，观察 `x.shape`；
4. 删除一条边的两个方向，观察 `edge_index.shape`；
5. 写出 `y=[1]` 当前表示整图标签还是四个节点标签；
6. 用 `graph.num_nodes` 检查节点数。

## 常见错误

- 把 `edge_index` 写成 `[边数, 2]`；
- 节点编号用了浮点类型；
- 无向边只放一个方向；
- 不确认 `y` 属于节点任务还是图任务；
- 在依赖未验证时直接进入训练。

## 完成标准

- [ ] 我能画出 `edge_index` 对应的图；
- [ ] 我能解释 `x.shape == (4, 2)`；
- [ ] 我知道 `Data` 只是容器，不会自动训练；
- [ ] 我记录了实际环境版本。

## 自测问题

1. 四条无向边为什么产生八个方向记录？
2. `edge_index.shape[0]` 通常为什么等于 2？
3. 若每个节点都有标签，`y` 的 shape 可能怎样变化？

## 导航

- 上一天：[Day 29 图基础](../day29_graph_basics/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 31 GCN](../day31_gcn/README.md)
- 总路线：[可选 GNN 课程](../README.md)
