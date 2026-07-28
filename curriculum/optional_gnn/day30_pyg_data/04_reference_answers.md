# Day 30.4：参考答案

> 本页只用于独立完成练习后的核对。参考代码是一种实现，不是唯一写法。

<details>
<summary>A. 字段解释</summary>

1. `Data` 是保存图字段的数据容器，不是训练模型。
2. `(4, 2)` 表示 4 个节点，每个节点有 2 个特征。
3. `(2, 8)` 表示第一行存来源、第二行存目标，共有 8 个方向记录。
4. 每一列把同一条方向记录的来源和目标对齐，所以要按列读。
5. 本日任务事先定义为“一张图对应一个连续目标”，因此一个值是整图标签；shape 本身不能脱离任务定义解释。
6. 当 `x` 存在时，通常可从 `x.shape[0]` 推断节点数。

</details>

<details>
<summary>B. Dtype 判断</summary>

1. 连续节点特征：`torch.float32`，用于数值计算；
2. 节点编号：`torch.long`，因为编号用于整数索引；
3. 连续回归标签：`torch.float32`，因为目标可以包含小数。

</details>

<details>
<summary>C. 构造另一张四节点图</summary>

```python
import torch
from torch_geometric.data import Data

x = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=torch.float32,
)

undirected_edges = [(0, 1), (0, 2), (0, 3)]
directed_edges = []

for source, target in undirected_edges:
    directed_edges.append((source, target))
    directed_edges.append((target, source))

edge_index = torch.tensor(
    directed_edges,
    dtype=torch.long,
).T.contiguous()

y = torch.tensor([0.25], dtype=torch.float32)
star_graph = Data(x=x, edge_index=edge_index, y=y)

assert tuple(star_graph.edge_index.shape) == (2, 6)
assert star_graph.num_nodes == 4

pairs = {
    (int(source), int(target))
    for source, target in star_graph.edge_index.T.tolist()
}
for source, target in pairs:
    assert (target, source) in pairs
```

</details>

<details>
<summary>D. 编写检查函数</summary>

一种合格实现：

```python
def check_toy_graph(graph, expected_num_nodes, expected_num_features):
    required_fields = ("x", "edge_index", "y")
    for field_name in required_fields:
        if getattr(graph, field_name, None) is None:
            raise ValueError(f"缺少字段：{field_name}")

    if tuple(graph.x.shape) != (
        expected_num_nodes,
        expected_num_features,
    ):
        raise ValueError("x shape 不符合预期")
    if graph.x.dtype != torch.float32:
        raise TypeError("x 必须为 torch.float32")

    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("edge_index shape 必须是 (2, 方向记录数)")
    if graph.edge_index.dtype != torch.long:
        raise TypeError("edge_index 必须为 torch.long")

    if tuple(graph.y.shape) != (1,):
        raise ValueError("本练习要求一个整图标签")
    if graph.y.dtype != torch.float32:
        raise TypeError("连续标签必须为 torch.float32")

    if graph.num_nodes != expected_num_nodes:
        raise ValueError("num_nodes 不符合预期")
    if graph.edge_index.numel() == 0:
        raise ValueError("本练习不接受空边集合")
    if int(graph.edge_index.min()) < 0:
        raise ValueError("节点编号不能为负")
    if int(graph.edge_index.max()) >= expected_num_nodes:
        raise ValueError("节点编号超出范围")

    pairs = [
        (int(source), int(target))
        for source, target in graph.edge_index.T.tolist()
    ]
    if len(pairs) != len(set(pairs)):
        raise ValueError("存在重复方向记录")

    pair_set = set(pairs)
    for source, target in pair_set:
        if source == target:
            raise ValueError("本练习不接受自环")
        if (target, source) not in pair_set:
            raise ValueError("无向边缺少反方向")

    return True

assert check_toy_graph(star_graph, 4, 2)
```

</details>

<details>
<summary>E. 标签语义</summary>

1. 不充分。数值 1.0 可能是回归目标，也可能是某种编码，必须查看任务定义和 dtype。
2. 不充分。四项可能是四个节点标签，也可能是一张图的四个任务，必须知道维度定义。
3. 错。shape 只说明一个维度长度，不能证明数据来源。
4. 对。任务定义、shape、dtype、单位和数据来源共同决定标签怎样解释。

</details>

<details>
<summary>F. Debug 题</summary>

问题包括：

1. 连续节点特征通常应为 `torch.float32`，示例却用了 `torch.long`；
2. 节点编号必须是整数索引，示例却把 `edge_index` 设为浮点；
3. 示例的 `edge_index.shape` 是 `(4, 2)`，而 PyG 要求第一维为 2；
4. 四条无向边只各写了一次，没有明确包含反方向；
5. 创建 `Data` 后还应先检查字段、shape、dtype、编号范围与双向性；
6. 本日目标是理解数据容器，不应跳过验证马上训练。

修复的核心形式：

```python
x = x.to(torch.float32)
edge_index = torch.tensor(
    [
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ],
    dtype=torch.long,
)
```

</details>

<details>
<summary>G. 证据边界示例</summary>

```text
本日输入是人为构造的节点特征、边和整图标签。
Data 把这些张量保存到同一个图对象中。
Data 没有训练模型，也没有确认特征具有化学意义。
y=0.75 是教学值，不是实验测量值。
结构检查通过能证明 shape、dtype、编号范围和双向边符合本练习约定。
它不能证明图代表真实分子，也不能证明 GNN 有预测能力。
进入 Day 31 前还应确认依赖环境正常，并能解释所有字段。
```

</details>

完成核对后，回到 [Day 30 任务卡](README.md)，用自己的话填写个人笔记。
