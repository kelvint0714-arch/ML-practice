# Day 24 参考答案

## A. 区分调参与消融

1. 是调参，不是机制消融；
2. 是消融；
3. 是消融；
4. 不是有效消融，因为多个因素同时变化。

把 `stack_tree_only` 直接设为树预测也不是有效消融：它删除了二层拟合机制，
无法与完整 stack 做“只移除 MLP”的控制变量比较。正确方案仍需用树的 OOF
预测训练 Ridge 二层。

## B. 计算题

- 1.75：`+0.05`，比基准差；
- 1.68：`-0.02`，比基准好；
- 1.70：`0`，与基准相同。

这些是题目示例数值，不是实验结果。

## C. 代码题

```python
def ablation_table(y_true, predictions, baseline_name):
    if baseline_name not in predictions:
        raise KeyError(f"Unknown baseline: {baseline_name}")
    baseline_rmse = root_mean_squared_error(
        y_true, predictions[baseline_name]
    )
    rows = []
    for name, pred in predictions.items():
        rmse = root_mean_squared_error(y_true, pred)
        rows.append({
            "variant": name,
            "rmse": rmse,
            "mae": mean_absolute_error(y_true, pred),
            "r2": r2_score(y_true, pred),
            "delta_vs_baseline": rmse - baseline_rmse,
        })
    return pd.DataFrame(rows)
```

## D. 2×2 消融

四行是：弱正则/关、弱正则/开、强正则/关、强正则/开。固定早停比较弱/强可观察正则效应；固定正则比较关/开可观察早停效应；仍需承认交互作用。

## E. 决策题

> 当前平均改善 0.01，小于跨种子标准差 0.08，且训练成本增至 10 倍，尚无稳定收益证据；优先保留简单平均，并保存完整负结果。

## F. 下游任务边界

确认字段含义与单位、固化条件在预测时是否可获得、缺失原因、配方/批次分组、测试标准一致性以及数据使用权限；任选四项并具体说明均可。

## G. 结构审计

`stack_tree_only` 的一级列表是 `[("tree", frozen_tree)]`；
`stack_tree_mlp` 是 `[("tree", frozen_tree), ("mlp", frozen_mlp)]`。
两者必须共享外部 train/valid、同一 Bemis–Murcko GroupKFold split 列表、
Ridge 二层、`passthrough=False` 和同一指标；这样差值才主要对应新增 MLP。
