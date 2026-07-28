# Day 18 练习参考答案

## A. 计算题

1. `ceil(100/32)=4` 个 batch，最后一批 4 行。
2. 每 epoch `96/16=6` 次，共 `6×20=120` 次。
3. 1 次更新。
4. batch size 1 时每 epoch 更新 `n` 次；全批时每 epoch更新 1 次。

## B. 概念题

5. 循环内重建会清空已经学习的权重，batch 之间无法累积。
6. 减少固定顺序造成的系统性更新模式，同时保持 X/y 对齐；固定 RNG 后仍可复现。
7. 验证分布会生成另一套缩放参数，且验证信息参与拟合；应使用训练缩放器 `.transform(X_valid)`。
8. 可能提示过拟合，也可能来自随机波动、学习率或验证集较小；需更多检查，不能单图定因。
9. 是。验证信息决定保留哪一轮。
10. `partial_fit` 在当前权重上继续更新；新模型的 `fit` 从新初始化状态开始。

## C. 代码题

11. `expected_batches = math.ceil(len(X_train) / 20)`。
12. 示例：

```python
seen_counts = np.zeros(len(X_train), dtype=int)
for batch_ids in epoch_batches:
    seen_counts[batch_ids] += 1
assert np.all(seen_counts == 1)
```

13. 增加墙钟计时即可，但小数据耗时受系统负载、硬件和首次调用开销影响。
14. 示例：

```python
record = {
    "generalization_gap": valid_rmse - train_rmse,
}
```

它只是描述性差值，不自动证明过拟合。
15. 一次差异同时受优化随机性影响，应重复固定协议或仅作为机制观察。
16. 每批新模型只学当前批，上一批状态丢失，epoch 历史不再代表连续训练。

## D. 边界题

17. 验证预测只测量当前固定参数；进入 `partial_fit` 会让参数利用验证标签更新，破坏独立性。
18. 不能。它是参与 epoch 选择的开发验证分数。
19. 可改为：

> 训练 RMSE 下降只说明模型更贴合训练数据；epoch 是否继续需要结合预先声明规则和验证曲线，更多轮可能过拟合。

20. 数据与缓存版本、ECFP 配置、固定 train/valid 边界、缩放策略、模型参数、batch size、epoch 数、两个随机种子、软件版本与完整曲线，任取至少五项。
