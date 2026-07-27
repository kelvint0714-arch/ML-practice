# Day 2.4：参考答案

> 本页只用于做完练习后的核对。先直接抄答案，会失去发现自己究竟卡在“符号、算术还是解释”哪一步的机会。

<details>
<summary>A. 方向与单位</summary>

1. 错。RMSE 是带目标单位的误差，不是百分比；越小越好。
2. 对。绝对误差的平均为 0 时，每个绝对误差都必须为 0。
3. 对。平方后再开平方，RMSE 回到原目标单位。
4. 错。R² 可以为负；负数表示比相应均值参考更差。
5. 对。对同一组有限误差，均方根不小于绝对值平均。
6. 错。例如误差 `[-2, 2]` 的平均为 0，但 MAE 为 2。

</details>

<details>
<summary>B. 纸笔误差表</summary>

| 样本 | `y_true` | `y_pred` | error | absolute | squared |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 1 | 1 | 1 |
| 2 | 2 | 2 | 0 | 0 | 0 |
| 3 | 4 | 1 | -3 | 3 | 9 |
| 4 | 7 | 9 | 2 | 2 | 4 |

```text
MAE = (1 + 0 + 3 + 2) / 4 = 1.5
MSE = (1 + 0 + 9 + 4) / 4 = 3.5
RMSE = sqrt(3.5) ≈ 1.871
```

第 3 个样本的平方误差 9 最大。平均误差：

```text
(1 + 0 - 3 + 2) / 4 = 0
```

它既不表示整体偏高，也不表示整体偏低；正负误差正好抵消，但模型仍然有明显误差。

</details>

<details>
<summary>C. 手算 R²</summary>

```text
y_true 平均值 = (1 + 2 + 4 + 7) / 4 = 3.5
当前预测平方误差总和 = 14
均值参考平方误差总和
= (1-3.5)² + (2-3.5)² + (4-3.5)² + (7-3.5)²
= 6.25 + 2.25 + 0.25 + 12.25
= 21
R² = 1 - 14 / 21 = 1/3 ≈ 0.333
```

解释示例：在这四个样本上，当前预测的平方误差小于数学均值参考，因此 R² 为正；这个小例子不能外推到其他数据。

</details>

<details>
<summary>D. Python 最小实现</summary>

一种合格实现：

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

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

y_true = np.array([1.0, 2.0, 4.0, 7.0])
y_pred = np.array([2.0, 2.0, 1.0, 9.0])
scores = regression_metrics(y_true, y_pred)

assert isinstance(scores["mae"], float)
assert np.isclose(scores["mae"], 1.5)

try:
    regression_metrics([1, 2, 3], [1, 2])
except ValueError:
    pass
else:
    raise AssertionError("长度不一致时应抛出 ValueError")

try:
    regression_metrics([1, np.nan], [1, 2])
except ValueError:
    pass
else:
    raise AssertionError("NaN 输入时应抛出 ValueError")
```

</details>

<details>
<summary>E. MAE 相同但错误结构不同</summary>

两个模型的 MAE 都是 1。

```text
模型 A RMSE = 1
模型 B RMSE = sqrt((0 + 0 + 0 + 16) / 4) = 2
```

模型 B 有一个集中的大错误，因此 RMSE 更大。如果极端错误代价很高，应重点关注 RMSE，同时查看逐样本误差，不能只靠一个汇总数字。四个玩具样本不足以得出真实项目的普遍结论。

</details>

<details>
<summary>F. Debug 题</summary>

至少有这些问题：

1. `y_true` 有 3 项，`y_pred` 有 2 项，无法逐项对应；
2. `mean_squared_error` 默认返回 MSE，代码没有开平方，变量名 `rmse` 不准确；
3. 输出标签写成 `accuracy`，但回归误差不是准确率；
4. 代码片段没有展示 NumPy 和指标函数的导入；
5. 没有检查有限值和 shape。

修复核心：

```python
import numpy as np
from sklearn.metrics import mean_squared_error

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.2, 2.1, 2.8])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)
```

</details>

完成核对后，不要把参考答案复制到学习记录。回到 [Day 2 任务卡](README.md)，用自己的话完成“结果解释”。
