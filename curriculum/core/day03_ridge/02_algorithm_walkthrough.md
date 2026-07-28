# Day 3.2：从一次线性预测到 Ridge 对照表

本页先做一次手算，再运行一个固定模型，最后只改变 `alpha`。配套的可运行版本见 [tutorial.ipynb](tutorial.ipynb)。

## 1. 先手算预测

已知：

```text
截距 = 0.2
权重 = 0.9
输入 = 4.0
```

代入：

```text
预测 = 0.2 + 0.9 × 4.0 = 3.8
```

训练过程的目标是找出截距和权重；预测过程只是把新输入代进去。

## 2. 固定玩具数据

```python
import numpy as np

X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([0.2, 1.1, 1.9, 3.2, 3.9, 5.1])

X_valid = np.array([[1.5], [3.5], [5.5]])
y_valid = np.array([1.4, 3.6, 5.4])

assert X_train.shape == (6, 1)
assert y_train.shape == (6,)
assert X_valid.shape == (3, 1)
assert y_valid.shape == (3,)
```

每行代表一个人工样本，唯一一列是输入特征。它不是 ESOL 或粘合剂数据。

## 3. 只训练 `alpha=1.0`

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

fixed_model = Ridge(alpha=1.0)
fixed_model.fit(X_train, y_train)

valid_prediction = fixed_model.predict(X_valid)
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_prediction))

print("coefficient:", float(fixed_model.coef_[0]))
print("intercept:", float(fixed_model.intercept_))
print("valid predictions:", valid_prediction)
print("valid RMSE:", float(valid_rmse))
```

逐行对应：

```text
Ridge(...)              → 创建未训练模型
.fit(X_train, y_train)  → 学习权重和截距
.predict(X_valid)       → 对验证输入形成预测
mean_squared_error      → 与验证答案比较
```

## 4. 手工复算第一个模型预测

```python
first_input = X_valid[0, 0]
manual_prediction = (
    fixed_model.intercept_
    + fixed_model.coef_[0] * first_input
)

assert np.isclose(manual_prediction, valid_prediction[0])
print("manual first prediction:", float(manual_prediction))
```

这一步证明 `predict` 在单特征 Ridge 中确实执行“截距＋权重×输入”。

## 5. 预先固定候选值

在看对照结果之前写下：

```python
alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]
```

这次是算法行为观察，不是正式调参。所有候选使用相同数据和指标。

## 6. 统一评价函数

```python
def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))
```

## 7. 只改变 `alpha`

```python
import pandas as pd

records = []

for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    records.append({
        "alpha": alpha,
        "coefficient": float(model.coef_[0]),
        "absolute_coefficient": float(abs(model.coef_[0])),
        "intercept": float(model.intercept_),
        "train_rmse": rmse(y_train, train_prediction),
        "valid_rmse": rmse(y_valid, valid_prediction),
    })

results = pd.DataFrame(records)
print(results.to_string(index=False))
```

先观察趋势，不急着按最小验证误差宣布“最佳”：

- `alpha` 增大时，绝对系数是否整体变小；
- 训练 RMSE 是否倾向增大；
- 极大 `alpha` 是否使预测更接近一个近似常数；
- validation 的变化是否与 train 完全同步。

## 8. 添加行为断言

对于这份单特征、同尺度玩具数据，可以检查候选序列两端：

```python
small_alpha_weight = results.loc[
    results["alpha"] == 0.01, "absolute_coefficient"
].iloc[0]
large_alpha_weight = results.loc[
    results["alpha"] == 100.0, "absolute_coefficient"
].iloc[0]

assert large_alpha_weight < small_alpha_weight
assert np.isfinite(results.select_dtypes("number").to_numpy()).all()
```

这条断言只核对本练习预期行为，不是“所有数据上每一项都严格单调”的普遍定理。

## 9. 怎样解释结果

合格表述结构：

```text
观察：alpha 从较小值增大到很大值时，系数被明显压缩。
机制：Ridge 对权重平方施加的惩罚变强。
结果：过强限制会让模型难以贴合训练关系，可能表现为欠拟合。
边界：这是人工单特征数据上的行为观察，不是正式模型选择。
```

不要只写“`alpha=某值` 最好”。

## 10. 教程 Notebook 与学习证据

[tutorial.ipynb](tutorial.ipynb) 是仓库提供的可运行教学伴侣，不代表你已经完成实验。亲自学习时再创建：

```text
experiments/day03_ridge/
├── README.md
├── day03_ridge.ipynb
├── notes.md
└── results/
    └── alpha_sensitivity.csv
```

- Notebook 中先写自己的预测，再运行；
- CSV 必须由实际 `results.to_csv(...)` 产生；
- `notes.md` 记录你看到的系数和误差趋势；
- 不复制教程输出冒充个人运行。

下一步：[Day 3 练习](03_exercises.md)。
