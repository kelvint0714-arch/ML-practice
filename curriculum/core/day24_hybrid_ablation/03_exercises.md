# Day 24 练习

## A. 区分调参与消融

分别判断：

1. 同时尝试 20 个 `alpha` 找最小 RMSE；
2. 保持参数不变，比较有/无 MLP；
3. 保持基础模型不变，比较 `passthrough=False/True`；
4. 换新划分同时换模型。

哪些是有效消融？为什么？

再解释：把 `stack_tree_only` 直接赋值为 `tree_only` 的预测，为什么不是有效消融？

## B. 计算题

基准 RMSE 为 1.70，三个变体分别为 1.75、1.68、1.70。计算 `delta_vs_baseline` 并解释正负号。

## C. 代码题

写函数：

```python
def ablation_table(y_true, predictions, baseline_name):
    ...
```

要求输出 `variant, rmse, mae, r2, delta_vs_baseline`，并在基准名称不存在时抛出清晰错误。

## D. 设计题

为正则化（弱/强）和早停（关/开）设计 2×2 消融。列出四行和每个主要对比。

## E. 决策题

复杂 stack 平均改善 0.01 RMSE，但标准差为 0.08，耗时为简单平均的 10 倍。写一段不超过 100 字的决策说明。

## F. 粘合剂边界

真实表格中希望比较“配方”和“配方+固化条件”。列出开始该消融前必须让化学组确认的四件事。

## G. 结构审计

写出 `stack_tree_only` 和 `stack_tree_mlp` 的一级模型列表。除新增 MLP 外，
哪些设置（外部划分、scaffold splits、二层 Ridge、指标）必须完全一致？
