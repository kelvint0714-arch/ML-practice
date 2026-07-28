# Day 12 练习参考答案

## A. 概念题

1. `alpha` 是训练前设定的超参数；`coef_` 是 `fit()` 从训练数据学到的模型参数。
2. 使用测试分数选择会让测试集参与开发，最终分数不再是独立评估。
3. `5 × 5 = 25` 次折内拟合，加 1 次最佳候选在完整训练区域上的 refit，共至少 26 次。
4. 不是。它是最佳候选的训练内交叉验证评分。
5. 这样每折只在该折训练部分学习填补值和缩放统计量，防止折内验证信息泄漏。
6. `-0.7` 更大，因此更好；对应普通 RMSE 分别是 `0.7` 和 `1.1`。

## B. 流程题

7. 合格图：

```text
X_train/y_train → 内部 K 折选 alpha
                         ↓ 冻结
X_valid/y_valid → 一次外部开发检查
                         ↓ 完整方案冻结
X_test/y_test   → 按协议一次最终评价
```

8. 预注册示例可使用 README 中的五个对数尺度候选、5 折、seed 42、RMSE、填补+标准化，并明确测试集不读取。
9. 折分数同时参与候选选择；从多个有噪声估计中取最好者，会吸收选择带来的乐观偏差。
10. 内层选择超参数，外层在未参与内层选择的样本上估计选择程序的性能。

## C. 代码题

11. 检查：

```python
assert len(candidate_table) == 3
assert set(candidate_table["param_ridge__alpha"]) == {0.001, 0.1, 10.0}
```

12. 示例：

```python
candidate_table["mean_train_rmse"] = -raw["mean_train_score"]
candidate_table["generalization_gap"] = (
    candidate_table["mean_cv_rmse"] - candidate_table["mean_train_rmse"]
)
```

13. `refit=False` 时搜索不会建立在完整训练区域上重拟合的最佳 Pipeline，因此不能使用 `best_estimator_` 直接预测。
14. Pipeline 嵌套参数必须通过两个下划线寻址。错误名不是合法参数。
15. 示例：

```python
fold0_rmse = -pd.DataFrame(search.cv_results_)["split0_test_score"]
```

## D. 边界题

16. 不能把同一外部验证集直接并入搜索后仍称它为外部验证。若重设计流程，要建立新的证据边界并诚实记录迭代。
17. 应说明外部验证集已被反复用于开发，最后分数是多轮选择后的开发结果，不再是一次独立检查。
18. 可改为：

> 在当前人工训练数据、固定五折和预先声明候选中，该 `alpha` 获得最低平均 CV RMSE；它不保证在候选范围外、其他划分或其他任务上最优。

19. 研究者已经依据其信息形成认识，后续方案可能受影响；信息暴露无法通过改名撤销。
