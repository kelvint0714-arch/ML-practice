# Day 29.2：从边列表到邻居消息求和

这一节按“对象 → 编号 → 边 → 矩阵 → 检查 → 聚合”的顺序推演。

## 1. 固定输入

节点名称：

```python
node_names = ["A", "B", "C", "D"]
```

输入类型是 Python 列表，长度为 4。列表的位置就是本日使用的节点编号：

| 列表位置 | 节点名称 |
|---:|---|
| 0 | A |
| 1 | B |
| 2 | C |
| 3 | D |

无向边只先写一次：

```python
undirected_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
```

`(0, 1)` 是含两个整数的元组。外层列表包含四个元组，因此这里表示四条无向边。

## 2. 创建空邻接矩阵

```python
import numpy as np

num_nodes = len(node_names)
adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
```

逐项解释：

- `len(node_names)` 返回整数 4；
- `(num_nodes, num_nodes)` 是 shape 元组 `(4, 4)`；
- `np.zeros(...)` 创建全零数组；
- `dtype=int` 表示元素保存为整数；
- 此时还没有把任何连接写进矩阵。

## 3. 把每条无向边写入两个位置

```python
for source, target in undirected_edges:
    adjacency[source, target] = 1
    adjacency[target, source] = 1
```

第一次循环：

```text
source = 0
target = 1
```

执行后：

```text
adjacency[0, 1] = 1
adjacency[1, 0] = 1
```

四次循环结束后，矩阵应为：

```text
[[0, 1, 0, 1],
 [1, 0, 1, 0],
 [0, 1, 0, 1],
 [1, 0, 1, 0]]
```

## 4. 先检查结构，再做后续计算

```python
assert adjacency.shape == (4, 4)
assert np.array_equal(adjacency, adjacency.T)
assert np.all(np.diag(adjacency) == 0)
```

含义：

- `.shape` 检查矩阵大小；
- `.T` 是转置，行列互换；
- 无向图的邻接矩阵应与转置相同；
- `np.diag(...)` 取对角线；
- 本日没有自环，所以对角线应全为 0。

断言不输出文字通常表示检查通过；失败时会抛出 `AssertionError`。

## 5. 从一行读出邻居

A 的编号是 0。它对应邻接矩阵第 0 行：

```text
[0, 1, 0, 1]
```

值为 1 的列编号是 1 和 3，所以 A 的邻居是 B 和 D。

NumPy 写法：

```python
neighbor_indices_of_a = np.flatnonzero(adjacency[0])
neighbor_names_of_a = [node_names[index] for index in neighbor_indices_of_a]
```

输入、动作、输出：

```text
输入：第 0 行 [0, 1, 0, 1]
动作：找出非零元素的位置
输出：数组 [1, 3]，再映射成 ["B", "D"]
```

## 6. 计算每个节点的度数

```python
degree = adjacency.sum(axis=1)
```

第 0 行的计算：

```text
0 + 1 + 0 + 1 = 2
```

所有行一起得到：

```text
[2, 2, 2, 2]
```

shape 从 `(4, 4)` 变为 `(4,)`。右侧空着不是缺失，而是“一维数组只记录一个维度长度”。

## 7. 加入节点特征

```python
node_features = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=float,
)
```

`node_features.shape == (4, 2)`：

- 第 0 行是 A 的两个特征；
- 第 1 行是 B 的两个特征；
- 以此类推。

特征值只是教学数字，不对应真实原子属性。

## 8. 手算 A 收到的邻居消息

A 的邻居编号为 1 和 3：

```python
manual_message_for_a = node_features[1] + node_features[3]
```

纸笔计算：

```text
[0.0, 1.0] + [0.5, 0.5] = [0.5, 1.5]
```

输出 shape 是 `(2,)`，因为它是 A 的一行、两个特征。

## 9. 一次计算所有节点

```python
neighbor_message_sum = adjacency @ node_features
```

矩阵乘法的 shape 规则：

```text
(4, 4) @ (4, 2) → (4, 2)
```

中间两个 4 必须相等：邻接矩阵的列在枚举节点，节点特征矩阵的行也在枚举节点。

输出第 0 行应该等于手算结果：

```python
assert np.allclose(neighbor_message_sum[0], manual_message_for_a)
```

## 10. 完整的最小函数

```python
def build_undirected_adjacency(number_of_nodes, edges):
    matrix = np.zeros((number_of_nodes, number_of_nodes), dtype=int)

    for source, target in edges:
        if source == target:
            raise ValueError("本练习不接受自环")
        if not (0 <= source < number_of_nodes):
            raise ValueError("source 超出节点编号范围")
        if not (0 <= target < number_of_nodes):
            raise ValueError("target 超出节点编号范围")

        matrix[source, target] = 1
        matrix[target, source] = 1

    return matrix
```

函数的输入、动作、输出：

```text
输入：节点数和无向边列表
动作：检查端点并把每条边写入正反两个方向
输出：整数邻接矩阵
```

## 11. 实际学习时的目录蓝图

开始 Day 29 时再创建个人目录：

```text
learning_outputs/day29_graph_basics/
├── README.md
├── day29_graph_basics.ipynb
├── notes.md
└── results/
    └── 可选的个人检查结果
```

课程教程留在 `curriculum/`；你的尝试与解释留在 `learning_outputs/`。完成推演后再做：[Day 29 练习](03_exercises.md)。
