# Day 20：MLP 正则化与早停

> 状态：待学习。候选设置需要实际运行后才能比较，本文不预设赢家。

## 今天为什么学

MLP 可以不断降低训练误差，但验证误差可能先下降后上升。
这说明模型逐渐记住训练细节，却没有提高对新样本的泛化能力。

今天比较较弱正则化与“更强正则化＋内部早停”两个预先声明的设置。
仍然只使用 NumPy 和 scikit-learn，不要求 PyTorch。

## 前置条件

- 完成 Day 19 的简单 MLP 基线；
- 能从训练与验证差距识别过拟合迹象；
- 知道 `alpha` 和学习率不是同一个参数；
- 明白外部验证集不能用于每一步更新。

## 今日产出

完成学习后，你应该产生：

1. 两个预先固定的 MLP 设置；
2. 训练与外部验证 RMSE 对照；
3. 实际迭代次数、损失曲线和早停记录；
4. 一段解释正则化与早停差别的文字；
5. Day 21 公平比较所需的冻结方案。

## 核心概念

### 1. L2 正则化

MLPRegressor 的 `alpha` 会惩罚过大的权重。
更大的 `alpha` 通常约束更强，但过强也可能造成欠拟合。

### 2. 早停

`early_stopping=True` 会从训练数据内部再留出一小部分，用于监控训练。
当内部验证分数连续若干轮没有改善时，训练提前停止。

### 3. 两个验证集不要混淆

- 内部早停验证：由 MLPRegressor 从训练集内部划出；
- 外部课程验证：用于比较冻结后的完整方案；
- 测试集：今天仍不使用。

### 4. 早停不是自动保证最优

早停结果仍受随机种子、内部划分、学习率和样本量影响。
需要重复实验后才能讨论稳定性。

## 分步骤任务

1. 在运行前写下两个候选设置，不看结果临时增加第三个。
2. 保持相同隐藏层、学习率、数据和随机种子。
3. 让候选只在 `alpha` 和早停设置上有明确差异。
4. 为每个候选建立独立 Pipeline。
5. 只在训练数据上调用 `fit()`。
6. 计算训练与外部验证 RMSE。
7. 记录 `n_iter_` 和 `loss_curve_` 长度。
8. 对早停模型记录内部最佳验证分数。
9. 判断是否存在过拟合或欠拟合迹象。
10. 冻结进入下一天比较的设置，不查看测试集。

## 核心代码骨架

```python
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
candidates = {
    "weak_regularization": MLPRegressor(
        hidden_layer_sizes=(64,), alpha=0.00001, learning_rate_init=0.001,
        max_iter=500, early_stopping=False, random_state=42),
    "regularized_early_stop": MLPRegressor(
        hidden_layer_sizes=(64,), alpha=0.001, learning_rate_init=0.001,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42),
}
rows = []
for name, mlp in candidates.items():
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        mlp,
    )
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    valid_pred = pipeline.predict(X_valid)
    trained_mlp = pipeline.named_steps["mlpregressor"]
    rows.append({
        "model": name,
        "train_rmse": root_mean_squared_error(y_train, train_pred),
        "valid_rmse": root_mean_squared_error(y_valid, valid_pred),
        "n_iter": trained_mlp.n_iter_,
        "loss_points": len(trained_mlp.loss_curve_),
        "internal_best_score": getattr(
            trained_mlp, "best_validation_score_", None
        ),
    })
for row in rows:
    print(row)
```

## 只解释今天新增的语法

- `early_stopping=True` 开启训练集内部的早停验证。
- `validation_fraction=0.1` 表示从训练数据内部留出 10%。
- `n_iter_no_change=20` 表示连续若干轮未改善后停止。
- `loss_curve_` 保存每轮训练损失，不是外部验证 RMSE。
- `getattr(object, name, default)` 在属性不存在时返回默认值，避免直接报错。
- `None` 表示该项不适用或没有数值，不应伪造为零。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 用外部验证集逐 Epoch 更新 | 外部验证被训练过程使用 | 只用内部早停划分 |
| 同时改变很多参数 | 无法解释差异来源 | 做清晰的控制变量 |
| 认为更大 `alpha` 必然更好 | 可能欠拟合 | 查看训练与验证差距 |
| 把 `loss_curve_` 当验证曲线 | 指标含义错误 | 单独记录外部验证 |
| 选择后继续反复试测试集 | 最终证据污染 | 测试集继续封存 |

## 完成标准

- [ ] 两个候选在运行前已经写明；
- [ ] 除正则化和早停外，主要设置保持一致；
- [ ] 所有 `fit()` 只接收训练数据；
- [ ] 我记录了训练/外部验证 RMSE 和迭代次数；
- [ ] 我能区分训练损失、内部早停验证和外部验证；
- [ ] 我冻结了下一天使用的 MLP 方案，但没有宣称它普遍最好。

## 自测问题

1. `alpha` 增大时模型一定更好吗？
2. 早停内部验证数据来自哪里？
3. `loss_curve_` 为什么不能当作外部验证 RMSE？
4. 训练和验证 RMSE 都很差可能是什么情况？
5. 为什么 Day 20 仍然不能查看测试集？

## 上一天 / 下一天

- 上一天：[Day 19：在 ESOL 上建立 MLP 基线](../day19_mlp_esol/README.md)
- 下一天：[Day 21：传统模型与 MLP 公平比较](../day21_mlp_vs_ml/README.md)
