# Day 11 练习：Pipeline 与泄漏

## A. 判断题（说明理由）

1. 先把全部 `X` 标准化，再只用训练行拟合 Ridge，不算泄漏。
2. `pipeline.predict(X_valid)` 会重新学习验证集均值。
3. 随机森林不要求标准化，所以它永远不会发生预处理泄漏。
4. 将填补器放进 Pipeline 能保证每个交叉验证折分别学习填充值。
5. Pipeline 会自动判断哪些列在现实预测时不可获得。

## B. 流程题

6. 画出“先全表填补与缩放、后划分”的错误流程，在泄漏位置标红或加星号。
7. 画出正确 Pipeline 的 `fit` 数据流与 `predict` 数据流。
8. 列出 `SimpleImputer`、`StandardScaler`、`Ridge` 各自在训练时保存的状态。
9. 为什么验证特征分布参与缩放，即使没有使用 `y_valid`，也违反独立验证边界？

## C. 代码题

10. 在 notebook 的人工数据中新增一列，并插入两个 `NaN`，确认 Pipeline 仍能运行。
11. 打印：

```python
pipeline.named_steps["imputer"].statistics_
pipeline.named_steps["scaler"].mean_
pipeline.named_steps["ridge"].coef_
```

分别解释数组长度。
12. 写断言证明缩放器均值等于“训练数据经训练填补器转换后”的列均值。
13. 故意对验证集调用一个新的 `StandardScaler().fit_transform(X_valid)`，解释为什么得到相同形状仍不代表流程正确。
14. 用 `cross_validate` 对完整 Pipeline 做 3 折验证，确认原始 `pipeline` 对象本身仍未被 `cross_validate` 拟合。

## D. 研究边界题

15. 某列是实验完成后才测得的“最终破坏模式”。即使把它放进 Pipeline，作为预测输入是否合理？
16. 验证 RMSE 在错误流程中比正确流程更差，是否能说明错误流程没有泄漏？
17. 写三句话解释为什么测试集今天仍不应使用。
18. 列出一个真实粘合剂表格中可能需要按组划分、而不是只靠 Pipeline 解决的问题。

## 提交检查

- [ ] 预处理和模型在同一条 Pipeline 内；
- [ ] 只对训练数据调用 `.fit()`；
- [ ] 验证数据只进入 `.predict()` 和指标计算；
- [ ] 有内部统计量复核；
- [ ] 知道 Pipeline 不负责数据划分和特征可获得性；
- [ ] 没有根据错误流程分数高低判断泄漏是否存在。
