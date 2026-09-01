# Day 4.2：从手工切分到树深控制实验

本页使用一个故意加入离群点的人工数据集。配套可运行版本见 [tutorial.ipynb](tutorial.ipynb)。

## 1. 先看正常关系和异常点

```python
import numpy as np

X_train = np.arange(12, dtype=float).reshape(-1, 1)
y_train = X_train.reshape(-1).copy()
y_train[5] = 20.0

X_valid = (np.arange(11, dtype=float) + 0.5).reshape(-1, 1)
y_valid = X_valid.reshape(-1)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("outlier sample:", X_train[5, 0], y_train[5])
```

除 `x=5` 外，训练关系为 `y=x`。验证集保持正常关系，没有复制该异常标签。

## 2. 先手工理解一个树桩

深度 1 的树只有一次主要切分。它最多形成两个叶区间，每个区间只能给一个均值预测，因此不可能精细跟随整条 `y=x` 关系。

预判：

```text
深度很小 → train 与 valid 都可能不够好
深度增加 → train 通常改善
深度过大 → 可能围绕离群点形成狭窄规则，valid 恶化
```

先写下预判，再运行。

## 3. 定义统一指标

```python
from sklearn.metrics import mean_squared_error, r2_score

def regression_metrics(actual, predicted):
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }
```

## 4. 固定候选深度

```python
depth_values = [1, 2, 3, None]
```

除了 `max_depth`，数据、指标和 `random_state` 都保持不变。

## 5. 训练并记录同一行的 train/valid

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

records = []

for depth in depth_values:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    train_scores = regression_metrics(y_train, train_prediction)
    valid_scores = regression_metrics(y_valid, valid_prediction)

    records.append({
        "max_depth": "None" if depth is None else str(depth),
        "actual_tree_depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        "train_rmse": train_scores["rmse"],
        "valid_rmse": valid_scores["rmse"],
        "train_r2": train_scores["r2"],
        "valid_r2": valid_scores["r2"],
        "r2_gap": train_scores["r2"] - valid_scores["r2"],
    })

results = pd.DataFrame(records)
print(results.to_string(index=False))
```

为什么额外记录：

- `actual_tree_depth`：实际长了多少层；
- `leaf_count`：模型划分了多少个最终区域；
- `r2_gap`：训练与验证表现差距的一个信号。

## 6. 检查不限深树是否记住训练样本

```python
unrestricted_model = DecisionTreeRegressor(
    max_depth=None,
    random_state=42,
)
unrestricted_model.fit(X_train, y_train)
unrestricted_train_prediction = unrestricted_model.predict(X_train)

assert np.allclose(unrestricted_train_prediction, y_train)
print("unrestricted tree memorized training labels")
```

这条断言在本玩具数据上成立。它证明模型能记住训练标签，不证明它能泛化。

## 7. 定位离群点附近的预测

```python
near_outlier = np.array([[4.5], [5.0], [5.5]])

for depth in [1, 2, 3, None]:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(near_outlier)
    print("depth:", depth, "predictions:", predictions)
```

观察深树是否给 `x=5.0` 一个非常局部的高预测，以及相邻验证位置如何受切分边界影响。

## 8. 可选：打印规则

```python
from sklearn.tree import export_text

small_tree = DecisionTreeRegressor(
    max_depth=2,
    random_state=42,
)
small_tree.fit(X_train, y_train)
print(export_text(small_tree, feature_names=["x"]))
```

逐行读取：

```text
|--- x <= 某阈值
|   |--- ...
|   |--- value: [叶节点预测]
```

## 9. 写结论的四层结构

```text
设置：所有候选只改变 max_depth。
观察：随着允许深度增加，实际深度/叶数和训练拟合发生怎样变化。
诊断：训练很好而验证明显较差，是过拟合信号。
边界：离群点是人为设置；不能把最小 validation RMSE 宣布为正式最优参数。
```

## 10. 教程与个人练习分开

[tutorial.ipynb](tutorial.ipynb) 是供应的教程，不是你的实验结果。实际完成 Day 4 时再创建：

```text
learning_outputs/day04_decision_tree/
├── README.md
├── day04_decision_tree.ipynb
├── notes.md
└── results/
    └── depth_comparison.csv
```

个人 Notebook 必须包含：

- 运行前对四个深度的预判；
- 自己运行得到的完整对照表；
- 至少一条围绕离群点的规则解释；
- “能支持/不能支持”的结论边界；
- 不调用 test。

下一步：[Day 4 练习](03_exercises.md)。
