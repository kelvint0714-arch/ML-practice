# Day 20 练习

## A. 判断题

判断并解释：

1. `alpha` 越大，验证 RMSE 一定越低。
2. 开启早停后，外部验证集会参与每轮训练。
3. `loss_curve_` 是外部验证 RMSE。
4. `best_validation_score_` 缺失时应填 0。
5. 达到 `max_iter` 必然说明模型完全不可用。
6. 回归任务的 `validation_scores_` 是逐轮内部 RMSE。

## B. 数据边界

请画出外部 train、内部更新集、内部早停集、外部 valid 和 test 的关系，
并标出 sklearn Pipeline 的插补器/缩放器究竟在哪一层 `fit()`。

## C. 代码题

写一个 `summarize_mlp(pipeline)` 函数，返回：

- `n_iter`；
- `loss_points`；
- `last_loss`；
- `best_internal_r2`，属性不存在时返回 `None`。

## D. 诊断题

方案 A 的 train/valid RMSE 为 0.5/1.8；方案 B 为 1.4/1.5。哪一个更好？请给出一个不能只看当前两行就回答的原因，并写出下一步验证。

## E. 消融思考

当前两个候选同时改变了 `alpha` 和 `early_stopping`。若要分别估计两个因素的影响，请列出一个 2×2 实验矩阵。

## F. 结论改写

把“早停解决了过拟合”改成一段符合当前证据的表述。
