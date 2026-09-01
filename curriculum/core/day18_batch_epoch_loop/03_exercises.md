# Day 18 练习：Batch、Epoch 与训练循环

## A. 计算题

1. 100 个训练样本、batch size 32，一个 epoch 有几个 batch？最后一批几行？
2. 96 个训练样本、batch size 16、20 个 epoch，共调用多少次 `partial_fit()`？
3. 10 个样本、batch size 大于 10，一个 epoch 有几个更新？
4. batch size 为 1 与全批训练在每个 epoch 的更新次数上有何区别？

## B. 概念题

5. 为什么模型必须在循环外创建？
6. 为什么每个 epoch 要打乱训练行？
7. `X_valid_scaled = scaler.fit_transform(X_valid)` 有什么问题？
8. 验证 RMSE 先降后升可能提示什么？能否仅凭曲线证明原因？
9. 根据最低验证 RMSE 选择 epoch 是否属于模型选择？
10. `partial_fit()` 与重新调用一个新模型的 `fit()` 有何区别？

## C. 代码题

11. 把 batch size 改为 20，计算并断言每 epoch 批次数。
12. 在每个 epoch 内建立 `seen_counts`，证明每个训练样本恰好出现一次。
13. 为 history 加入 `epoch_seconds`，说明它为何只能作为当前环境粗略记录。
14. 保存每个 epoch 的训练—验证差距 `valid_rmse - train_rmse`。
15. 故意删除 `rng.permutation`，使用固定顺序；保持其他条件不变并比较，但不要仅凭一次结果宣布优劣。
16. 故意在批循环内新建模型，观察 loss/指标行为后解释为什么学习无法累积。

## D. 边界题

17. 为什么验证集可以每个 epoch 预测，却不能进入 `partial_fit()`？
18. 保存“最佳验证 epoch”后，还能把该验证分数称为严格测试分数吗？
19. 改写错误结论：

> 训练 RMSE 一直下降，所以 epoch 越多越好。

20. 写出把本日循环迁移到 ESOL 时必须重新记录的五项内容。

## 提交检查

- [ ] 模型和 RNG 在循环外创建；
- [ ] 每轮训练索引完整且无重复；
- [ ] 验证数据不进入缩放器 fit 或 `partial_fit`；
- [ ] history 每 epoch 一行；
- [ ] 更新次数断言正确；
- [ ] 曲线标题与轴标签明确；
- [ ] 没有把人工曲线写成 ESOL 或下游任务结果。
