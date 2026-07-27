# Day 5.2：从一次 Bootstrap 到单树/森林公平对照

配套可运行版本见 [tutorial.ipynb](tutorial.ipynb)。先完成前两节的纸笔预测，再运行模型。

## 1. 手工模拟一次 Bootstrap

原编号：

```text
[0, 1, 2, 3, 4]
```

假设抽到：

```text
[1, 1, 4, 0, 4]
```

回答：

- 样本 1 和 4 各出现两次；
- 样本 2 和 3 没有出现；
- 这仍然是 5 次抽样；
- 它是有放回抽样。

用 NumPy 固定复现一个例子：

```python
import numpy as np

rng = np.random.default_rng(42)
original_indices = np.arange(5)
bootstrap_indices = rng.choice(
    original_indices,
    size=len(original_indices),
    replace=True,
)

print("original:", original_indices)
print("bootstrap:", bootstrap_indices)
```

`replace=True` 是允许重复的关键。

## 2. 手算森林平均

三棵树对一个样本预测：

```text
[2.0, 2.5, 3.0]
```

森林回归预测：

```text
(2.0 + 2.5 + 3.0) / 3 = 2.5
```

平均预测不是从真实答案算出的；预测阶段模型不能看 `y_valid`。

## 3. 固定人工数据

```python
X_train = np.array([
    [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0],
    [4.0, 0.0], [5.0, 1.0], [6.0, 0.0], [7.0, 1.0],
    [8.0, 0.0], [9.0, 1.0],
])
y_train = np.array([
    0.2, 1.1, 1.9, 3.2, 3.9,
    5.1, 5.8, 7.2, 7.9, 9.1,
])

X_valid = np.array([
    [1.5, 0.0], [3.5, 1.0], [5.5, 0.0], [7.5, 1.0],
])
y_valid = np.array([1.4, 3.6, 5.4, 7.6])

assert X_train.shape == (10, 2)
assert y_train.shape == (10,)
```

第一列近似决定连续目标，第二列是 0/1 人工特征。它不是分子指纹。

## 4. 公平定义两个模型

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

models = {
    "one_tree": DecisionTreeRegressor(
        max_depth=3,
        random_state=42,
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        max_features=1,
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    ),
}
```

公平之处：

- 相同 train/valid；
- 相同最大深度；
- 相同评价函数；
- 固定随机种子；
- 两者每行结果使用同一列含义。

`n_jobs=1` 让教学运行更可预测；它不改变算法问题。

## 5. 同一循环训练与评价

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }

records = []

for model_name, model in models.items():
    model.fit(X_train, y_train)

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
    ]:
        prediction = model.predict(X_split)
        scores = regression_metrics(y_split, prediction)
        records.append({
            "model": model_name,
            "split": split_name,
            **scores,
        })

results = pd.DataFrame(records)
print(results.to_string(index=False))
```

`**scores` 把字典中的 `mae`、`rmse`、`r2` 展开到当前记录中。

## 6. 查看森林中树的差异

```python
forest = models["random_forest"]
first_valid_sample = X_valid[[0]]

tree_predictions = np.array([
    tree.predict(first_valid_sample)[0]
    for tree in forest.estimators_
])

print("first 10 tree predictions:", tree_predictions[:10])
print("tree prediction mean:", tree_predictions.mean())
print("forest prediction:", forest.predict(first_valid_sample)[0])

assert np.isclose(
    tree_predictions.mean(),
    forest.predict(first_valid_sample)[0],
)
```

这一步直接验证：sklearn 的回归森林输出等于内部树预测的平均值。

## 7. 检查差异是否真的存在

```python
unique_tree_predictions = np.unique(
    np.round(tree_predictions, decimals=8)
)
print("number of unique tree predictions:", len(unique_tree_predictions))
assert len(unique_tree_predictions) > 1
```

在这份数据和配置下，内部树对该样本不完全相同。不要把这条具体断言盲目复制到任意小数据；如果所有树碰巧预测相同，它也不必然意味着代码错误。

## 8. 可选观察树数量

核心任务完成后，固定其他条件，只改变：

```python
tree_counts = [10, 50, 100, 300]
```

记录 validation RMSE 和运行时间，但不要使用 test，也不要看到结果后无止境追加候选。

## 9. 结果解释模板

```text
对照：单树与森林使用同一份人工数据和指标。
观察：写出本次实际运行的 train/valid 差异。
机制：Bootstrap 与随机特征使内部树产生差异，回归预测再取平均。
边界：一次玩具数据结果不能证明随机森林永远优于单树。
```

## 10. 供应教程与个人证据

[tutorial.ipynb](tutorial.ipynb) 是课程材料。真正学习时再创建：

```text
experiments/day05_random_forest/
├── README.md
├── day05_random_forest.ipynb
├── notes.md
└── results/
    ├── model_comparison.csv
    └── optional_tree_count.csv
```

`optional_tree_count.csv` 只有做了可选实验才生成，不能留一份预填数字的假结果。

下一步：[Day 5 练习](03_exercises.md)。
