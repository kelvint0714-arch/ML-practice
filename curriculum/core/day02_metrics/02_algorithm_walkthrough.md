# Day 2.2：从四个预测逐步算到指标函数

这一节按“纸笔 → 表格 → 函数 → 检查”的顺序完成。第一次学习时，不要先复制最后的完整代码。

## 1. 固定输入

真实值和预测值：

```text
y_true = [-3.0, -2.0, -1.0,  0.0]
y_pred = [-2.5, -2.4, -0.2, -0.1]
```

先检查：

- 两列各有 4 项；
- 第 1 个预测对应第 1 个真实值；
- 它们都表示同一个目标、同一个单位。

## 2. 逐行建立误差表

按照 `error = y_pred - y_true`：

| 样本 | `y_true` | `y_pred` | `error` | `absolute_error` | `squared_error` |
|---:|---:|---:|---:|---:|---:|
| 1 | -3.0 | -2.5 | 0.5 | 0.5 | 0.25 |
| 2 | -2.0 | -2.4 | -0.4 | 0.4 | 0.16 |
| 3 | -1.0 | -0.2 | 0.8 | 0.8 | 0.64 |
| 4 | 0.0 | -0.1 | -0.1 | 0.1 | 0.01 |

先口头解释两行：

- 第 1 行误差为正，表示预测偏高 0.5；
- 第 2 行误差为负，表示预测偏低 0.4。

## 3. 手算 MAE

```text
绝对误差总和 = 0.5 + 0.4 + 0.8 + 0.1 = 1.8
MAE = 1.8 / 4 = 0.45
```

注意分母是样本数 4，不是特征数，也不是误差总和。

## 4. 手算 RMSE

```text
平方误差总和 = 0.25 + 0.16 + 0.64 + 0.01 = 1.06
MSE = 1.06 / 4 = 0.265
RMSE = sqrt(0.265)
```

先保留 `sqrt(0.265)`，再用计算器或 NumPy求值。不要用已经四舍五入的中间数继续计算。

## 5. 推演 R²

这批真实值的平均值：

```text
y_true_mean = (-3 - 2 - 1 + 0) / 4 = -1.5
```

均值参考的平方误差：

| `y_true` | 与 -1.5 的差 | 平方 |
|---:|---:|---:|
| -3.0 | -1.5 | 2.25 |
| -2.0 | -0.5 | 0.25 |
| -1.0 | 0.5 | 0.25 |
| 0.0 | 1.5 | 2.25 |

所以：

```text
当前模型平方误差总和 = 1.06
均值参考平方误差总和 = 5.00
R² = 1 - 1.06 / 5.00
```

结果应在 0 和 1 之间，说明这组预测在平方误差意义下优于数学均值参考。

## 6. 把每一步翻译成 pandas

```python
import numpy as np
import pandas as pd

y_true = np.array([-3.0, -2.0, -1.0, 0.0])
y_pred = np.array([-2.5, -2.4, -0.2, -0.1])

error_table = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred,
})
error_table["error"] = error_table["y_pred"] - error_table["y_true"]
error_table["absolute_error"] = error_table["error"].abs()
error_table["squared_error"] = error_table["error"] ** 2

print(error_table)
```

输入、动作、输出：

```text
输入：两个同长度的一维数组
动作：逐项相减、取绝对值、平方
输出：每个样本一行的可检查误差表
```

## 7. 先写输入检查，再写统一函数

指标函数不应悄悄接受明显不合理的输入。

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(actual, predicted):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)

    if actual.shape != predicted.shape:
        raise ValueError("actual 与 predicted 必须有相同 shape")
    if actual.size < 2:
        raise ValueError("至少需要两个样本")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("输入不能包含 NaN 或无穷")

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }

scores = regression_metrics(y_true, y_pred)
print(scores)
```

这里的 `.reshape(-1)` 把 `(4,)` 和 `(4, 1)` 都整理成单目标回归常用的一维 `(4,)`。

## 8. 用断言核对手算

```python
manual_mae = error_table["absolute_error"].mean()
manual_rmse = np.sqrt(error_table["squared_error"].mean())

y_mean = y_true.mean()
model_squared_error_sum = error_table["squared_error"].sum()
mean_baseline_squared_error_sum = ((y_true - y_mean) ** 2).sum()
manual_r2 = 1 - (
    model_squared_error_sum / mean_baseline_squared_error_sum
)

assert np.isclose(manual_mae, scores["mae"])
assert np.isclose(manual_rmse, scores["rmse"])
assert np.isclose(manual_r2, scores["r2"])
```

断言没有输出表示检查通过；`AssertionError` 才表示不一致。

## 9. 制造一个负 R²

只为了理解合法范围，可以尝试：

```python
bad_prediction = np.array([3.0, 2.0, 1.0, 0.0])
bad_scores = regression_metrics(y_true, bad_prediction)
print(bad_scores)
assert bad_scores["r2"] < 0
```

这不是程序故障，而是这组预测比均值参考差。

## 10. 实际学习时的实验目录蓝图

只有你亲自开始 Day 2 时，再创建：

```text
learning_outputs/day02_metrics/
├── README.md
├── day02_metrics.ipynb
├── notes.md
└── results/
    ├── error_table.csv
    └── metrics.json
```

文件职责：

- `README.md`：问题、输入、运行方法、指标定义和证据边界；
- Notebook：按本页顺序运行，必须能从空内核执行；
- `notes.md`：用自己的话回答“为什么 RMSE 通常不小于 MAE”和“负 R² 是什么”；
- CSV：每个样本一行，不手工改数字；
- JSON：由指标函数输出，不预先填写。

推荐 Notebook 顺序：

```text
Goal → Setup → Error Table → Manual Checks → Metric Function → Negative R² Check → Save
```

完成推演后再做：[Day 2 练习](03_exercises.md)。
