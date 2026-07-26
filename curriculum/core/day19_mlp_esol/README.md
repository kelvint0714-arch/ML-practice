# Day 19：在 ESOL 上建立 MLP 基线

> 状态：待学习。本文不包含已验证的 MLP 分数，也不代表粘合剂实验结果。

## 今天为什么学

前四天分别学习了形状、前向计算、损失和训练循环。
今天把这些概念放回 ESOL，使用 scikit-learn 的 `MLPRegressor` 建立第一个神经网络回归基线。

重点不是让 MLP 一定超过随机森林，而是让它和传统模型使用相同输入、划分和指标。

## 前置条件

- 能解释隐藏层、ReLU、损失和 Epoch；
- 知道 MLP 对特征尺度比较敏感；
- 会用 Pipeline 防止标准化泄漏；
- 已经从 ESOL 流程得到 `X_train`、`y_train`、`X_valid`、`y_valid`。

## 今日产出

完成学习后，你应该产生：

1. 一条可复现的 MLP Pipeline；
2. 训练与验证 MAE、RMSE、R²；
3. MLP 参数与迭代次数记录；
4. 与 Dummy 和一个传统模型相同口径的对照表；
5. 一段准确的结论边界。

## 核心概念

### 1. 为什么继续使用 ECFP

Day 19 的目的是比较算法，不是同时更换数据表示。
因此 MLP 暂时使用与 ESOL 参考基线 E01 相同的 1024 维 ECFP。

### 2. 隐藏层结构

`hidden_layer_sizes=(64,)` 表示一个隐藏层，其中有 64 个单元。
它不是 64 层，也不是 64 个输入特征。

### 3. MLP 也需要基线

神经网络结构更复杂，不代表小数据上一定更好。
比较中必须保留 Dummy、Ridge 或随机森林，且不能因为结果不好就改变比较协议。

### 4. 今天暂不早停

为了先建立一个最简单的 MLP 基线，代码暂时关闭 `early_stopping`。
正则化和早停在 Day 20 单独研究。

## 分步骤任务

1. 复核 ESOL 标签仍是原始 logS。
2. 复核训练与验证使用固定的同一划分。
3. 建立填补、标准化、MLP 三步 Pipeline。
4. 在运行前记录全部 MLP 参数。
5. 只对训练数据调用 `fit()`。
6. 计算训练与验证预测。
7. 用 Day 1 相同的 MAE、RMSE、R²。
8. 保存实际迭代次数与是否收敛的提示。
9. 与传统模型结果并排，不改变旧模型数据。
10. 明确 ESOL 不是粘合剂数据。

## 核心代码骨架

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlp_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=400,
        early_stopping=False,
        random_state=42,
    )),
])

mlp_pipeline.fit(X_train, y_train)

def metrics(y_true, prediction):
    return {
        "mae": mean_absolute_error(y_true, prediction),
        "rmse": np.sqrt(mean_squared_error(y_true, prediction)),
        "r2": r2_score(y_true, prediction),
    }

train_pred = mlp_pipeline.predict(X_train)
valid_pred = mlp_pipeline.predict(X_valid)
mlp = mlp_pipeline.named_steps["mlp"]

print("train:", metrics(y_train, train_pred))
print("valid:", metrics(y_valid, valid_pred))
print("iterations:", mlp.n_iter_)
```

## 只解释今天新增的语法

- `(64,)` 是只有一项的元组，末尾逗号不能省略。
- `activation="relu"` 指定隐藏层激活函数。
- `solver="adam"` 选择 MLPRegressor 内部的优化算法。
- `alpha` 是 L2 正则化强度，不是学习率。
- `max_iter` 是最多迭代次数，不保证一定运行到该数值。
- `n_iter_` 末尾下划线表示这是 `fit()` 后生成的属性。

## 常见错误

| 错误 | 后果 | 处理 |
|---|---|---|
| 不标准化直接训练 MLP | 优化可能困难 | 保留训练内 Pipeline |
| 把 `(64,)` 写成 `64` 后误解结构 | 参数含义不清 | 明确隐藏层元组 |
| 看到收敛警告就删除警告 | 掩盖训练问题 | 记录参数和完整提示 |
| MLP 分数低就换划分 | 比较失去公平性 | 固定同一协议 |
| 把 ESOL 写成粘合剂强度 | 结论越界 | 明确任务是 logS |

## 完成标准

- [ ] MLP 与传统模型使用相同 ESOL 输入和划分；
- [ ] 标准化器只在训练数据上拟合；
- [ ] 我保存了训练与验证三项指标；
- [ ] 我记录了全部参数、随机种子和迭代次数；
- [ ] 我没有查看测试集来调 MLP；
- [ ] 我没有声称 MLP 一定优于传统模型。

## 自测问题

1. `hidden_layer_sizes=(64,)` 表示什么？
2. `alpha` 与 `learning_rate_init` 有何区别？
3. 为什么 MLP 需要与 Dummy 比较？
4. `n_iter_ == max_iter` 时应检查什么？
5. 为什么 ESOL MLP 结果不能直接用于推荐粘合剂配方？

## 上一天 / 下一天

- 上一天：[Day 18：Batch、Epoch 与训练循环](../day18_batch_epoch_loop/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 20：正则化与早停](../day20_regularization_early_stopping/README.md)
