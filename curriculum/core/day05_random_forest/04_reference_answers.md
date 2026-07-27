# Day 5.4：参考答案

> 展开前先完成练习。这里的代码和解释只用于核对，不是个人运行证据。

<details>
<summary>A. Bootstrap 判断</summary>

1. 可能：长度为 4，编号合法，重复允许；
2. 可能：有放回抽样也可能恰好每个编号一次；
3. 不符合本题：只抽了 3 次，而题设要求 4 次；
4. 不可能：原训练编号中没有 4；
5. 可能：有放回抽样允许四次都抽到编号 3。

</details>

<details>
<summary>B. 平均预测</summary>

```text
样本 A：(1+2+3+2)/4 = 2
样本 B：(4+3+2+3)/4 = 3
样本 C：(7+8+9+8)/4 = 8
```

森林预测为 `[2, 3, 8]`。

</details>

<details>
<summary>C. 参数配对</summary>

| 参数 | 作用编号 |
|---|---:|
| `n_estimators` | 4 |
| `max_depth` | 5 |
| `max_features` | 2 |
| `bootstrap` | 1 |
| `random_state` | 3 |

</details>

<details>
<summary>D. 反例解释</summary>

1. 完全相同的树对同一样本给出相同预测，平均后数值不变，共同错误不会抵消。
2. 如果每棵树都没有有效信号，或都受同一数据偏差影响，有差异也不能保证准确。
3. 更多树通常让平均更稳定，但收益可能已饱和，同时不会消除偏差、泄漏或错误划分。
4. 随机森林仍可能过拟合，特别是在小样本、噪声、深树或反复调参等条件下。

</details>

<details>
<summary>E. 逐树平均检查</summary>

```python
tree_prediction_matrix = np.vstack([
    tree.predict(X_valid)
    for tree in forest.estimators_
])

assert tree_prediction_matrix.shape == (
    len(forest.estimators_),
    len(X_valid),
)

manual_forest_prediction = tree_prediction_matrix.mean(axis=0)
sklearn_forest_prediction = forest.predict(X_valid)

assert np.allclose(
    manual_forest_prediction,
    sklearn_forest_prediction,
)
```

`axis=0` 表示对“不同树”这一行方向求平均，为每个验证样本保留一个预测。

</details>

<details>
<summary>F. 公平对照实现</summary>

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

def compare_tree_and_forest(X_train, y_train, X_valid, y_valid):
    models = {
        "one_tree": DecisionTreeRegressor(
            max_depth=3,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42,
            n_jobs=1,
        ),
    }
    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        for split, X_split, y_split in [
            ("train", X_train, y_train),
            ("valid", X_valid, y_valid),
        ]:
            prediction = model.predict(X_split)
            rows.append({
                "model": model_name,
                "split": split,
                "mae": float(mean_absolute_error(y_split, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(
                    y_split,
                    prediction,
                ))),
                "r2": float(r2_score(y_split, prediction)),
            })

    return pd.DataFrame(rows)
```

</details>

<details>
<summary>G. 假想结果解释</summary>

单树的训练 RMSE 更低，说明它对训练数据拟合更强。随机森林在这一次固定 validation 上 RMSE 更低，因此当前验证表现较好。不能据此声称随机森林对所有数据、所有划分或真实粘合剂任务永远更好，也不能把 validation 数字称为最终 test 性能。

</details>

<details>
<summary>H. 不公平比较</summary>

- 单树和森林使用的训练样本数量不同；
- 单树在 validation 评价，森林在 test 评价；
- 两个指标面对不同样本，不能直接归因于模型；
- 代码暴露 test 并可能将其用于开发比较；
- 未显示两个模型的其他参数与随机性是否固定。

修复时让两个模型使用同一 `X_train/y_train` 拟合，并在同一 `X_valid/y_valid` 上用同一函数评价；test 保持不用。

</details>

核对后返回 [Day 5 任务卡](README.md)，自己画一次 Bagging 流程。
