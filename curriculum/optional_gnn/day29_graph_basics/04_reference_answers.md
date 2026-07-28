# Day 29.4：参考答案

> 本页只用于独立完成练习后的核对。不要把参考答案复制成自己的学习记录。

<details>
<summary>A. 名词对应</summary>

1. A：节点；
2. A—B：边；
3. `[1.0, 0.0]`：节点特征；
4. 整张图的目标值：整图标签；
5. 原子元素类型：节点特征；
6. 单键或双键类型：边特征。

</details>

<details>
<summary>B. 手画与邻接矩阵</summary>

编号：

```text
A=0，B=1，C=2，D=3
```

边列表的一种写法：

```python
edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
```

邻接矩阵：

```text
[[0, 1, 1, 0],
 [1, 0, 1, 0],
 [1, 1, 0, 1],
 [0, 0, 1, 0]]
```

邻居和度数：

| 节点 | 邻居 | 度数 |
|---|---|---:|
| A | B、C | 2 |
| B | A、C | 2 |
| C | A、B、D | 3 |
| D | C | 1 |

矩阵关于主对角线对称，符合无向图表示。

</details>

<details>
<summary>C. 独立实现</summary>

一种合格实现：

```python
import numpy as np

def build_undirected_adjacency(number_of_nodes, edges):
    if number_of_nodes <= 0:
        raise ValueError("number_of_nodes 必须为正整数")

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

exercise_edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
exercise_adjacency = build_undirected_adjacency(4, exercise_edges)

assert exercise_adjacency.shape == (4, 4)
assert np.issubdtype(exercise_adjacency.dtype, np.integer)
assert np.array_equal(exercise_adjacency, exercise_adjacency.T)
assert np.all(np.diag(exercise_adjacency) == 0)
```

</details>

<details>
<summary>D. 手算消息求和</summary>

`node_features.shape == (4, 2)`。

A 的邻居是 B 和 C：

```text
[0, 1] + [1, 1] = [1, 2]
```

C 的邻居是 A、B 和 D：

```text
[1, 0] + [0, 1] + [2, 0] = [3, 1]
```

代码核对：

```python
node_features = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ]
)

messages = exercise_adjacency @ node_features

assert messages.shape == (4, 2)
assert np.allclose(messages[0], [1.0, 2.0])
assert np.allclose(messages[2], [3.0, 1.0])
```

</details>

<details>
<summary>E. Shape 解释</summary>

1. 邻接矩阵第一个 4 是四个来源节点，第二个 4 是四个目标节点；
2. 节点特征第一个 4 是节点数，第二个 2 是每个节点的特征数；
3. `degree` 每个节点只有一个度数，所以是长度为 4 的一维数组 `(4,)`；
4. 矩阵乘法要求左矩阵列数等于右矩阵行数，两者都是节点数 4；
5. 输出仍然每个节点一行，所以保留 4 行，每行包含聚合后的两个特征。

</details>

<details>
<summary>F. Debug 题</summary>

第一段只写了 `adjacency[0, 1] = 1`，没有写反方向 `adjacency[1, 0] = 1`，所以矩阵不对称，断言失败。

第二段中：

```text
adjacency.shape = (4, 4)
node_features.shape = (3, 2)
```

矩阵乘法中间维度 4 与 3 不相等，因此无法相乘；而且图有 4 个节点，特征矩阵却只提供了 3 行。

</details>

<details>
<summary>G. 证据边界示例</summary>

```text
今天的数据是人为构造的四节点玩具图。
节点编号只是数组索引，不是节点特征。
邻接矩阵表达节点之间是否连接。
它没有表达真实原子性质、键类型、配方比例或工艺条件。
邻居求和是固定计算，不是经过训练的模型。
今天的结果能支持我理解图表示、shape、度数与消息求和。
它不能支持 GNN 性能结论或粘合剂性质预测结论。
```

</details>

完成核对后，回到 [Day 29 任务卡](README.md)，用自己的话填写个人笔记。
