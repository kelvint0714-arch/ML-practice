# Day 33.3：独立练习

## 练习 1：多头 shape

写出以下层的输出 shape：

```python
GATConv(
    in_channels=34,
    out_channels=8,
    heads=4,
    concat=True,
)
```

输入 `x.shape == [34,34]`。

## 练习 2：修复 shape 错误

第一层输出是 `[N,32]`，但第二层写成：

```python
GATConv(8, 4, heads=1, concat=False)
```

为什么会报错？第二层的 `in_channels` 应该是多少？

## 练习 3：concat

分别解释：

- `heads=4, out_channels=8, concat=True`
- `heads=4, out_channels=8, concat=False`

输出宽度的区别。

## 练习 4：注意力返回值

假设：

```text
attention_edges.shape == [2, 190]
alpha.shape == [190, 2]
```

解释每个轴的意义。为什么 190 可能大于原始边数？

## 练习 5：有界输出

写代码只显示注意力的前 8 条边，不排序。解释为什么“不排序”在本练习中是有意的。

## 练习 6：找出过度解释

下面结论有什么问题？

> 边 3→7 的注意力最高，因此这条关系导致节点 7 属于类别 2。

至少指出三项逻辑缺口。

## 练习 7：公平比较

为了让 GAT 分数更高，有人把它训练 500 epoch，而 GCN/SAGE 仍训练 120 epoch。这样得到的结果回答了什么？没有回答什么？

## 练习 8：参数量与结果

如果 GAT 的参数量最多、validation mean 也最高，可以直接说“注意力机制带来了提升”吗？请写一个更谨慎的表述。

## 练习 9：test 封存

Day 33 已经比较了三个模型，为什么仍不能查看 test 排名？什么时候才可以查看？

完成后：[查看参考答案](04_reference_answers.md)。
