# Day 30.2：把 Day 29 的图逐字段装入 `Data`

这一节的顺序是“检查环境 → 创建张量 → 创建容器 → 逐字段检查”。不要一开始就把所有代码压成一个单元。

## 1. 检查实际运行环境

```python
import sys
import torch
import torch_geometric
from torch_geometric.data import Data

print("Python executable:", sys.executable)
print("torch version:", torch.__version__)
print("torch_geometric version:", torch_geometric.__version__)
```

输入：当前 Python 环境。
动作：导入依赖并读取版本。
输出：解释器路径和实际版本。

这里不写死推荐版本，因为兼容组合会变化。导入失败时应先修复环境，而不是继续运行后续单元。

## 2. 固定随机种子

本日没有随机运算，但仍建立可复现习惯：

```python
SEED = 30
torch.manual_seed(SEED)
```

`torch.manual_seed` 设置 PyTorch 随机数种子。今天的固定输入不会因种子变化，但后续初始化模型时会用到。

## 3. 创建节点特征 `x`

```python
x = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=torch.float32,
)
```

输入：四行 Python 数值列表。
动作：转换为 32 位浮点张量。
输出：`x.shape == (4, 2)`。

立即检查：

```python
assert x.ndim == 2
assert tuple(x.shape) == (4, 2)
assert x.dtype == torch.float32
```

## 4. 从无向边生成双向记录

先保留人容易阅读的无向边：

```python
undirected_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
```

再明确展开正反方向：

```python
directed_edges = []

for source, target in undirected_edges:
    directed_edges.append((source, target))
    directed_edges.append((target, source))
```

此时 `directed_edges` 长度为 8。把“每行一个方向对”的列表转成 PyG 所需的“两行”：

```python
edge_index = torch.tensor(directed_edges, dtype=torch.long).T.contiguous()
```

语法拆解：

- `torch.tensor(directed_edges, ...)` 初始 shape 是 `(8, 2)`；
- `.T` 转置成 `(2, 8)`；
- `.contiguous()` 让转置后的张量在内存布局上连续，便于后续算子使用；
- `dtype=torch.long` 保证节点编号是整数。

## 5. 创建整图回归标签

```python
y = torch.tensor([0.75], dtype=torch.float32)
```

我们事先定义“一张图对应一个连续目标”，所以输出 shape `(1,)` 合理。`0.75` 是教学值，没有实验来源。

## 6. 创建 `Data`

```python
graph = Data(x=x, edge_index=edge_index, y=y)
```

关键字参数的对应关系是：

```text
左侧字段名 x          ← 右侧变量 x
左侧字段名 edge_index ← 右侧变量 edge_index
左侧字段名 y          ← 右侧变量 y
```

`Data` 把引用保存到对象中，因此可以读取：

```python
print(graph.x)
print(graph.edge_index)
print(graph.y)
```

## 7. 建立字段摘要

```python
field_summary = {
    "x": {
        "shape": tuple(graph.x.shape),
        "dtype": str(graph.x.dtype),
        "meaning": "每个节点一行、两个教学特征",
    },
    "edge_index": {
        "shape": tuple(graph.edge_index.shape),
        "dtype": str(graph.edge_index.dtype),
        "meaning": "每列是一条方向记录",
    },
    "y": {
        "shape": tuple(graph.y.shape),
        "dtype": str(graph.y.dtype),
        "meaning": "一张图的一个连续教学标签",
    },
}

for field_name, description in field_summary.items():
    print(field_name, description)
```

输出被限制为三个字段，不会产生难以阅读的大量调试文本。

## 8. 检查 shape、dtype 与节点数

```python
assert tuple(graph.x.shape) == (4, 2)
assert graph.x.dtype == torch.float32

assert tuple(graph.edge_index.shape) == (2, 8)
assert graph.edge_index.dtype == torch.long

assert tuple(graph.y.shape) == (1,)
assert graph.y.dtype == torch.float32

assert graph.num_nodes == 4
```

这些检查分别回答：

```text
有多少节点？
每个节点有多少特征？
保存了多少方向记录？
节点编号能否用于索引？
标签是整图一个连续值吗？
```

## 9. 检查编号范围和无向双向性

```python
assert int(graph.edge_index.min()) >= 0
assert int(graph.edge_index.max()) < graph.num_nodes

directed_pairs = {
    (int(source), int(target))
    for source, target in graph.edge_index.T.tolist()
}

assert len(directed_pairs) == 8

for source, target in directed_pairs:
    assert (target, source) in directed_pairs
```

这会发现：

- 负节点编号；
- 超过节点数的编号；
- 重复方向导致的数量不一致；
- 某条无向边缺少反方向。

## 10. 使用 PyG 自带结构检查

```python
validation_result = graph.validate(raise_on_error=True)
print("PyG Data validation:", validation_result)
```

PyG 的结构检查是额外防线，不能替代对任务语义的人工检查。即使 `validate` 返回成功，`0.75` 是否是真实标签、特征是否合理仍要由数据来源决定。

## 11. 今天为什么不训练

到这里我们只有：

```text
数据容器
```

还没有：

```text
模型、参数、forward、loss、optimizer、训练循环、评价协议
```

把数据检查与模型训练分开，能让初学者知道错误到底来自“图构造”还是“模型代码”。

## 12. 实际学习时的目录蓝图

```text
experiments/day30_pyg_data/
├── README.md
├── day30_pyg_data.ipynb
├── notes.md
└── results/
    └── 可选的字段检查摘要
```

完成推演后再做：[Day 30 练习](03_exercises.md)。
