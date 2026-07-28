# Day 4.4：参考答案

> 参考答案不能替代自己的树规则图、实际运行表或结果解释。完成练习后再逐项展开。

<details>
<summary>A. 叶节点预测</summary>

叶预测：

```text
(2 + 3 + 7) / 3 = 4
```

平方误差：

```text
(2-4)² = 4
(3-4)² = 1
(7-4)² = 9
总和 = 14
```

若 `[2, 3]` 在一个叶，预测 2.5，训练平方误差为 `0.25+0.25=0.5`；`[7]` 单独成叶，误差为 0；总和降为 0.5。

但这只说明训练误差降低。新切分可能是在记住噪声，验证样本未必受益。

</details>

<details>
<summary>B. 候选阈值</summary>

不切分时总均值为 2：

```text
SSE = 4 + 4 + 4 + 4 = 16
```

`x <= 0.5`：

- 左叶 `[0]`，SSE 为 0；
- 右叶 `[0,4,4]`，均值为 `8/3`；
- 右叶 SSE 为 `64/9 + 16/9 + 16/9 = 96/9 ≈ 10.667`。

`x <= 1.5`：

- 左叶 `[0,0]`，预测 0，SSE 为 0；
- 右叶 `[4,4]`，预测 4，SSE 为 0。

因此第二个候选对这份训练数据更好。

</details>

<details>
<summary>C. 术语</summary>

1. 根节点；
2. 叶节点；
3. `max_depth`；
4. `min_samples_leaf`；
5. 过拟合。

</details>

<details>
<summary>D. 读结果表</summary>

深度 10 的训练 R² 为 1、验证 R² 为负，训练—验证差距最大，过拟合信号最强。深度 1 的 train 与 valid 都较低，可能欠拟合。

深度 3 只是在这一次验证表中表现相对较好，不能证明它在其他划分或数据上全局最佳。正式比较时至少固定数据划分、特征、目标、指标、其他树参数和随机性协议。

</details>

<details>
<summary>E. 控制变量函数</summary>

一种实现：

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

def compare_tree_depths(
    X_train,
    y_train,
    X_valid,
    y_valid,
    depths,
):
    records = []

    for depth in depths:
        model = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        train_prediction = model.predict(X_train)
        valid_prediction = model.predict(X_valid)

        train_r2 = float(r2_score(y_train, train_prediction))
        valid_r2 = float(r2_score(y_valid, valid_prediction))

        records.append({
            "max_depth": depth,
            "actual_tree_depth": model.get_depth(),
            "leaf_count": model.get_n_leaves(),
            "train_rmse": float(np.sqrt(mean_squared_error(
                y_train,
                train_prediction,
            ))),
            "valid_rmse": float(np.sqrt(mean_squared_error(
                y_valid,
                valid_prediction,
            ))),
            "train_r2": train_r2,
            "valid_r2": valid_r2,
            "r2_gap": train_r2 - valid_r2,
        })

    return pd.DataFrame(records)
```

</details>

<details>
<summary>F. 泄漏与不公平比较</summary>

主要问题：

1. 每个深度使用不同 `random_state` 重新划分，候选没有在同一验证样本上比较；
2. `DecisionTreeRegressor` 没有固定自身 `random_state`，复现协议不完整；
3. 在循环中重复划分使“树深影响”和“数据划分影响”混在一起。

修复：先在循环外固定一次划分，再在循环内只改变 `max_depth`，并固定模型随机种子。

</details>

<details>
<summary>G. 外推</summary>

普通回归树不会沿一条学到的斜率继续外推。`x=100` 会沿已有阈值走到最外侧的某个叶节点，输出该叶训练标签的均值。

因此报告时应检查验证/应用样本是否超出训练特征范围，并说明树模型的外推局限。

</details>

核对后返回 [Day 4 任务卡](README.md)，用自己的话复述“树怎样预测”。
