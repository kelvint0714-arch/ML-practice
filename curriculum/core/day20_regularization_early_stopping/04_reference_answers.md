# Day 20 参考答案

## A. 判断题

1. 错。过强正则化可能欠拟合。
2. 错。内部早停集来自传给 `fit()` 的外部训练集；外部验证不应进入。
3. 错。它是内部训练损失历史。
4. 错。缺失表示不适用，应保存 `None`/空值并解释。
5. 错。它表示触及上限，需要结合曲线、警告和泛化指标判断。
6. 错。`MLPRegressor` 回归任务的 `validation_scores_` 是逐轮内部 `R²`，不是 RMSE。

## B. 数据边界

```text
外部 train（唯一传给 fit 的数据）
├── 内部更新集：计算梯度
└── 内部早停集：决定停止时间

外部 valid：比较预先固定候选
test：方案冻结前不使用
```

在教程的原生 sklearn Pipeline 中，插补器和缩放器先对完整外部 train
拟合，MLP 才在变换后的矩阵中切内部早停集。所以外部边界保持安全，
但内部早停集的预处理没有严格隔离；严谨嵌套选择需显式切分并只在
subtrain 拟合预处理器。

## C. 代码题

```python
def summarize_mlp(pipeline):
    mlp = pipeline.named_steps["mlpregressor"]
    return {
        "n_iter": mlp.n_iter_,
        "loss_points": len(mlp.loss_curve_),
        "last_loss": mlp.loss_curve_[-1],
        "best_internal_r2": getattr(
            mlp, "best_validation_score_", None
        ),
    }
```

## D. 诊断题

B 的验证 RMSE 较低且训练—验证差距较小，但它的训练误差明显更高，可能有欠拟合。两行单次结果不足以评价随机性；应在相同预算下做预先声明的多种子或交叉验证，并同时考虑误差、方差和耗时。

## E. 2×2 实验

| `alpha` | 早停关闭 | 早停开启 |
|---:|---|---|
| 0.00001 | A | B |
| 0.001 | C | D |

其余设置必须一致。A↔C、B↔D 主要比较正则强度；A↔B、C↔D 主要比较早停。

## F. 结论改写

> 在当前固定 ESOL 划分和参数下，开启内部早停的候选于某轮停止，并得到实际运行的训练/验证指标。该现象与减少继续拟合相符，但一次结果不能证明早停普遍解决过拟合，仍需多种子稳定性检查。
