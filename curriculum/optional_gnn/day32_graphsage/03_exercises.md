# Day 32.3：独立练习

## 练习 1：手算 mean 聚合

节点 0 的三个邻居表示为：

```text
[2, 0]
[1, 4]
[3, 2]
```

计算 mean 聚合结果。如果交换三行顺序，结果是否变化？

## 练习 2：解释两个权重

用自己的话解释：为什么 GraphSAGE 可以对“节点自身表示”和“邻居平均表示”使用不同的可学习变换？

## 练习 3：shape

给定 `x.shape == [34, 34]`、`hidden_channels=16`、`classes=4`，写出两层 GraphSAGE 每一步的 shape。

## 练习 4：找出不公平比较

下面比较有哪些问题？

| 条件 | GCN | GraphSAGE |
|---|---:|---:|
| hidden | 16 | 64 |
| epoch | 80 | 300 |
| seed | 7 | 27 |
| validation mask | A | B |

至少指出四项，并写出修复方案。

## 练习 5：配对种子

写出循环骨架，使两个模型都使用 `7、17、27`，且每次都重新初始化模型。

## 练习 6：解释 mean/std

模型 A 的 validation accuracy：

```text
0.60, 0.80, 0.70
```

模型 B：

```text
0.69, 0.70, 0.71
```

不要求精确手算标准差。仅根据均值和波动，写出谨慎结论。

## 练习 7：参数量

为什么“相同隐藏宽度”不等于“相同参数量”？参数量不同是否意味着比较完全无效？

## 练习 8：归纳能力边界

当前训练使用完整 KarateClub 图但遮住 validation 标签。它为什么不能直接证明 GraphSAGE 能对一张全新的图泛化？

## 练习 9：test 纪律

如果 GraphSAGE validation 均值低于 GCN，你是否可以查看 test accuracy，看看它会不会反超？解释原因。

完成后：[查看参考答案](04_reference_answers.md)。
