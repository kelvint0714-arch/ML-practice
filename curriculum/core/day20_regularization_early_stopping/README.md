# Day 20：MLP 正则化与早停

> 状态：待学习。候选设置需要实际运行后才能比较，本文不预设赢家。

## 本日完整学习包

1. [概念精讲](01_concepts.md)
2. [算法走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 是课程附带材料，不等于学习者实验已经完成；个人运行证据应在开始学习后另存到 `learning_outputs/`。

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

`validation_scores_` 和 `best_validation_score_` 在回归任务中记录的是内部留出集
`R²`，不是 RMSE。还要注意 sklearn 的实现边界：Pipeline 会先在全部
`X_train` 上拟合插补器和缩放器，然后 `MLPRegressor` 才在变换后的数据中
划出内部早停集。因此外部 valid/test 没有进入训练，但这个内部早停分数
并不满足“预处理器也完全没见过内部留出集”的严格隔离。

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
8. 对早停模型记录内部最佳 `R²`，并披露预处理先见过全部 `X_train` 的限制。
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
        hidden_layer_sizes=(32,), alpha=0.00001, learning_rate_init=0.001,
        max_iter=300, early_stopping=False, random_state=42),
    "regularized_early_stop": MLPRegressor(
        hidden_layer_sizes=(32,), alpha=0.001, learning_rate_init=0.001,
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42),
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
        "best_internal_r2": getattr(
            trained_mlp, "best_validation_score_", None
        ),
    })
for row in rows:
    print(row)
```

## 只解释今天新增的语法

- `early_stopping=True` 开启训练集内部的早停验证。
- `validation_fraction=0.1` 表示从训练数据内部留出 10%。
- `n_iter_no_change=10` 表示连续若干轮内部 `R²` 未充分改善后停止。
- `loss_curve_` 保存每轮训练损失，不是外部验证 RMSE。
- `validation_scores_` 保存内部早停留出集每轮的 `R²`，也不是 RMSE。
- `getattr(object, name, default)` 在属性不存在时返回默认值，避免直接报错。
- `None` 表示该项不适用或没有数值，不应伪造为零。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 用外部验证集逐 Epoch 更新 | 外部验证被训练过程使用 | 只用内部早停划分 |
| 同时改变很多参数 | 无法解释差异来源 | 做清晰的控制变量 |
| 认为更大 `alpha` 必然更好 | 可能欠拟合 | 查看训练与验证差距 |
| 把 `loss_curve_` 当验证曲线 | 指标含义错误 | 单独记录外部验证 |
| 把内部 `validation_scores_` 当 RMSE | 指标和方向都读反 | 明确它是回归 `R²` |
| 声称原生 Pipeline 内部早停完全隔离 | 忽略预处理先拟合全部 `X_train` | 披露限制；严格实验显式切 subtrain |
| 选择后继续反复试测试集 | 最终证据污染 | 测试集继续封存 |

## 完成标准

- [ ] 两个候选在运行前已经写明；
- [ ] 除正则化和早停外，主要设置保持一致；
- [ ] 所有 `fit()` 只接收训练数据；
- [ ] 我记录了训练/外部验证 RMSE 和迭代次数；
- [ ] 我能区分训练损失、内部早停验证和外部验证；
- [ ] 我知道内部 `validation_scores_` 是 `R²`，并能复述 sklearn Pipeline 的预处理边界；
- [ ] 我冻结了下一天使用的 MLP 方案，但没有宣称它普遍最好。

## 自测问题

1. `alpha` 增大时模型一定更好吗？
2. 早停内部验证数据来自哪里？
3. `loss_curve_` 为什么不能当作外部验证 RMSE？
4. 为什么原生 Pipeline 的内部早停集没有做到预处理完全隔离？
5. 为什么 Day 20 仍然不能查看测试集？

## 上一天 / 下一天

- 上一天：[Day 19：在 ESOL 上建立 MLP 基线](../day19_mlp_esol/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 21：传统模型与 MLP 公平比较](../day21_mlp_vs_ml/README.md)
