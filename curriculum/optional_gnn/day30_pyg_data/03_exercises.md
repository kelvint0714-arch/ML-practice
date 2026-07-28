# Day 30.3：PyG `Data` 练习

先关闭 [参考答案](04_reference_answers.md)。先写出期望 shape 和 dtype，再运行代码。

## A. 字段解释

用完整中文句子回答：

1. `Data` 是数据容器还是训练模型？
2. `x.shape == (4, 2)` 中两个数字分别表示什么？
3. `edge_index.shape == (2, 8)` 中两个数字分别表示什么？
4. `edge_index` 为什么按列读？
5. `y.shape == (1,)` 在本日为什么表示整图标签？
6. `graph.num_nodes` 在有 `x` 时通常从哪里推断？

## B. Dtype 判断

给下列字段选择本日合适的 dtype，并说明理由：

1. 连续节点特征；
2. 节点编号；
3. 连续整图回归标签。

候选：

```text
torch.float32
torch.long
```

## C. 构造另一张四节点图

仍使用 Day 29 的四个节点特征，但把无向边改为：

```text
0—1、0—2、0—3
```

完成：

1. 写出三个无向边元组；
2. 展开成六个方向记录；
3. 创建 `edge_index`；
4. 检查 `edge_index.shape == (2, 6)`；
5. 创建 `Data`；
6. 检查 `num_nodes == 4`；
7. 检查每个方向都有反方向。

## D. 编写检查函数

自己完成：

```python
def check_toy_graph(graph, expected_num_nodes, expected_num_features):
    # 检查 x、edge_index、y 是否存在
    # 检查 shape
    # 检查 dtype
    # 检查节点编号范围
    # 检查反方向
    # 检查通过后返回 True
    pass
```

本练习约定：

- `y` 是一个整图连续标签；
- 不接受自环；
- 不接受重复方向记录；
- 图必须按无向边的双向形式保存。

## E. 标签语义

判断下面说法是否充分，并解释：

1. `y = torch.tensor([1.0])`，所以一定是二分类；
2. `y.shape == (4,)`，所以一定是整图四任务回归；
3. `y.shape == (1,)`，所以一定来自实验；
4. 只有同时知道任务定义、shape、dtype 和单位，才能正确解释标签。

## F. Debug 题

下面代码至少有哪些问题？

```python
x = torch.tensor(
    [[1, 0], [0, 1], [1, 1], [0, 0]],
    dtype=torch.long,
)

edge_index = torch.tensor(
    [[0, 1], [1, 2], [2, 3], [3, 0]],
    dtype=torch.float32,
)

y = torch.tensor([0.75], dtype=torch.float32)
graph = Data(x=x, edge_index=edge_index, y=y)
```

至少从以下角度检查：

- `x` dtype；
- `edge_index` dtype；
- `edge_index` shape；
- 无向边方向；
- 是否应该马上训练。

## G. 证据边界

用自己的话填写：

```text
本日输入是什么：
Data 做了什么：
Data 没有做什么：
y 的来源是什么：
结构检查通过能证明什么：
结构检查通过不能证明什么：
进入 Day 31 前我还应确认什么：
```

## 验收清单

- [ ] 能独立创建一个小型 `Data`；
- [ ] 能解释 `x`、`edge_index`、`y`；
- [ ] 能检查 shape、dtype 和节点编号范围；
- [ ] 能检查无向边双向性；
- [ ] 能区分标签形状与标签语义；
- [ ] 没有训练模型，也没有声称得到性能结果。

全部完成后再打开：[折叠参考答案](04_reference_answers.md)。
