# Day 26 练习

## A. 区域题

解释已标注集、候选池和固定外部对照集为什么必须分开。候选被揭示后，它属于哪个区域？为什么本课程的 ESOL valid 不能再称为严格未见 test？

## B. 顺序题

将下列步骤排序：

- 更新已标注集；
- 计算候选分歧；
- 揭示被选标签；
- 拟合集成；
- 固定并保存 query；
- 在独立集评价。

## C. 泄漏审计

指出问题：

```python
pool_error = np.abs(y_pool - model.predict(X_pool))
query = np.argsort(pool_error)[-10:]
```

如果只在模拟结束后用 `pool_error` 评价采集策略，是否可以？说明边界。

## D. 随机对照

列出主动策略与随机策略必须保持相同的五项条件。

## E. 代码题

写一个确定性函数 `select_top_disagreement(prediction_matrix, candidate_ids, batch_size)`，返回按分歧从高到低的候选 ID，并检查预算合法。

## F. 真实项目交接

设计一张 8 列候选交接表，至少包含候选编号、预测、分歧、配方/工艺、可行性、禁止原因和模型版本。

## G. 结论边界

解释为什么在 ESOL 池模拟中改善，仍不能说“主动学习已为粘合剂项目节省实验”。
