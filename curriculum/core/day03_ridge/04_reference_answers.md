# Day 3.4：参考答案

> 先完成练习再展开。参考答案用于定位错误，不是可以复制成个人练习记录的运行证据。

<details>
<summary>A. 手算线性预测</summary>

```text
预测
= -0.5 + 2.0×3.0 + (-1.0)×4.0 + 0.25×8.0
= -0.5 + 6 - 4 + 2
= 3.5
```

第二个特征贡献 `-1.0×4.0=-4.0`。若该特征从 4 变为 5，贡献再减少 1，因此预测从 3.5 变成 2.5。

</details>

<details>
<summary>B. shape 题</summary>

```text
X_train.shape = (80, 12)
y_train.shape = (80,)
model.coef_.shape = (12,)
model.predict(X_train).shape = (80,)
```

`y_train` 是包含 80 个目标值的一维数组。`(80,)` 中的逗号是 Python 单元素元组语法，不表示右侧丢失数据；若形状是 `(80, 1)`，那才是二维 80 行 1 列。

</details>

<details>
<summary>C. 参数与超参数</summary>

| 项目 | 分类 |
|---|---|
| `coef_` | 训练后学习的参数 |
| `intercept_` | 训练后学习的参数 |
| `alpha` | 训练前设置的超参数 |
| validation RMSE | 评价结果 |
| `fit_intercept` | 训练前设置的超参数 |

</details>

<details>
<summary>D. 预测趋势</summary>

1. 通常可能：惩罚项权重增大，系数通常更受压缩。
2. 不一定：过小和过大都可能不合适，验证误差不保证单调下降。
3. 通常可能：更强约束减少了模型贴合训练数据的自由度。
4. 概念错误：改变 Ridge 的 `alpha` 不会改变算法家族。
5. 通常可能：权重被压得过小，训练和验证都可能变差。

</details>

<details>
<summary>E. 候选比较函数</summary>

一种实现：

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def compare_ridge_alphas(
    X_train,
    y_train,
    X_valid,
    y_valid,
    alphas,
):
    records = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        train_prediction = model.predict(X_train)
        valid_prediction = model.predict(X_valid)

        records.append({
            "alpha": float(alpha),
            "coefficient_norm": float(np.linalg.norm(model.coef_)),
            "train_rmse": float(np.sqrt(mean_squared_error(
                y_train,
                train_prediction,
            ))),
            "valid_rmse": float(np.sqrt(mean_squared_error(
                y_valid,
                valid_prediction,
            ))),
        })

    result = pd.DataFrame(records)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("结果含有非有限值")
    return result
```

函数没有访问或接收 test，因此不会用 test 选择候选。

</details>

<details>
<summary>F. 找错题</summary>

1. `fit` 之前读取 `coef_` 会触发尚未拟合错误；
2. 代码用 validation 调用 `fit`，让验证集参与训练；
3. 随后却在 train 上评价，train/validation 职责被颠倒；
4. `mean_squared_error` 返回的是 MSE，不是 RMSE；
5. 没有保留真正未参与训练的候选比较数据。

正确骨架应是：

```python
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
valid_prediction = model.predict(X_valid)
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_prediction))
```

</details>

<details>
<summary>G. 结果解释示例</summary>

随着 `alpha` 从 0.01 增至 100，系数范数从 8.2 降到 0.2，符合更强 L2 惩罚压缩权重的直觉。`alpha=0.01` 的训练误差很小而验证误差明显更大，提示弱正则化下可能存在过拟合，但不能仅凭这一表证明唯一原因。`alpha=100` 的训练和验证误差都较高，说明模型可能受限过强而欠拟合。在固定的假想数据与划分之外，这张表不能证明某个 `alpha` 对 ESOL 或下游任务任务普遍最好。

</details>

核对后回到 [Day 3 任务卡](README.md)，重新用自己的语言解释 Ridge。
