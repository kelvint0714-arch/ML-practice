# Day 6.2：从一次残差修正到逐阶段 RMSE

配套可运行版本见 [tutorial.ipynb](tutorial.ipynb)。

## 1. 纸笔完成第一轮

给定：

```text
y_true = [1, 2, 6]
初始预测 = [3, 3, 3]
第一棵树修正 = [-1.5, -1.5, 3.0]
learning_rate = 0.1
```

残差：

```text
[-2, -1, 3]
```

更新后：

```text
[3,3,3] + 0.1×[-1.5,-1.5,3.0]
= [2.85, 2.85, 3.30]
```

新的残差：

```text
[1,2,6] - [2.85,2.85,3.30]
= [-1.85, -0.85, 2.70]
```

第三个样本仍然预测偏低，后续树需要继续给它正修正。

## 2. 固定人工训练/验证数据

```python
import numpy as np

X_train = np.array([
    [0.0], [1.0], [2.0], [3.0], [4.0], [5.0],
])
y_train = np.array([0.2, 1.1, 1.9, 3.2, 3.9, 5.1])

X_valid = np.array([[1.5], [3.5], [5.5]])
y_valid = np.array([1.4, 3.6, 5.4])

assert X_train.shape == (6, 1)
assert y_train.shape == (6,)
```

## 3. 训练固定基线

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=2,
    random_state=42,
)
model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
valid_prediction = model.predict(X_valid)

train_rmse = float(np.sqrt(mean_squared_error(
    y_train,
    train_prediction,
)))
valid_mae = float(mean_absolute_error(y_valid, valid_prediction))
valid_rmse = float(np.sqrt(mean_squared_error(
    y_valid,
    valid_prediction,
)))
valid_r2 = float(r2_score(y_valid, valid_prediction))

print("train RMSE:", train_rmse)
print("valid MAE:", valid_mae)
print("valid RMSE:", valid_rmse)
print("valid R²:", valid_r2)
```

这是固定基线，不是参数搜索。

## 4. 验证模型确实包含顺序树

```python
print("estimators_ shape:", model.estimators_.shape)
assert model.estimators_.shape == (100, 1)
```

对于单目标回归，`estimators_` 第一维对应提升阶段。这里有 100 个阶段。

## 5. 取得逐阶段预测

```python
staged_train_predictions = list(model.staged_predict(X_train))
staged_valid_predictions = list(model.staged_predict(X_valid))

assert len(staged_train_predictions) == model.n_estimators
assert len(staged_valid_predictions) == model.n_estimators
assert np.allclose(staged_valid_predictions[-1], valid_prediction)
```

最后一个阶段应等于普通 `predict` 的最终结果。

## 6. 建立学习过程表

```python
import pandas as pd

stage_rows = []

for stage_number, (train_stage_pred, valid_stage_pred) in enumerate(
    zip(staged_train_predictions, staged_valid_predictions),
    start=1,
):
    stage_rows.append({
        "stage": stage_number,
        "train_rmse": float(np.sqrt(mean_squared_error(
            y_train,
            train_stage_pred,
        ))),
        "valid_rmse": float(np.sqrt(mean_squared_error(
            y_valid,
            valid_stage_pred,
        ))),
    })

stage_table = pd.DataFrame(stage_rows)
selected_stages = stage_table[
    stage_table["stage"].isin([1, 5, 10, 25, 50, 100])
]
print(selected_stages.to_string(index=False))
```

不打印全部 100 行，避免输出淹没重点。

## 7. 检查训练误差的方向

```python
train_changes = np.diff(stage_table["train_rmse"].to_numpy())
print("largest train RMSE increase:", float(train_changes.max()))

assert stage_table["train_rmse"].iloc[-1] < stage_table["train_rmse"].iloc[0]
```

这里检查第一阶段到最后阶段总体下降，不把严格逐步单调当作所有配置的硬规则。

## 8. 可选：只改变学习率

```python
learning_rates = [0.03, 0.1, 0.3]
comparison_rows = []

for learning_rate in learning_rates:
    candidate = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=100,
        max_depth=2,
        random_state=42,
    )
    candidate.fit(X_train, y_train)
    candidate_valid_prediction = candidate.predict(X_valid)

    comparison_rows.append({
        "learning_rate": learning_rate,
        "n_estimators": candidate.n_estimators,
        "valid_rmse": float(np.sqrt(mean_squared_error(
            y_valid,
            candidate_valid_prediction,
        ))),
    })

learning_rate_table = pd.DataFrame(comparison_rows)
print(learning_rate_table.to_string(index=False))
```

候选值在运行前固定，并且没有同时改变树数或深度。

## 9. 解释顺序

写结果时先回答机制：

```text
初始模型提供起点；
每个新阶段学习当前组合模型留下的错误方向；
learning_rate 缩放单阶段贡献；
最终预测是初始值与所有阶段修正之和。
```

再写本次实际观察，最后注明玩具数据边界。

## 10. 教程与个人运行分开

真正学习时再由教程副本创建：

```text
learning_outputs/day06_gradient_boosting/
├── README.md
├── day06_gradient_boosting.ipynb
├── notes.md
└── results/
    ├── staged_metrics.csv
    └── optional_learning_rate.csv
```

若没有做可选学习率观察，不生成第二个 CSV。不要把 [tutorial.ipynb](tutorial.ipynb) 的预存输出当成自己的结果。

下一步：[Day 6 练习](03_exercises.md)。
