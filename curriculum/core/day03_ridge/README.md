# Day 3：线性回归与 Ridge

## 今天为什么学

Ridge 是你学习的第一个真正会根据输入 `X` 改变预测值的算法。

今天先理解线性模型怎样形成预测，再用很小的人工数据观察正则化强度。暂时不学习标准化和 Pipeline，它们会在 Day 11 单独处理。

## 前置条件

- 已完成 [Day 2 回归评价指标](../day02_metrics/README.md)；
- 能解释 `X`、`y`、`fit` 和 `predict`；
- 知道 MAE、RMSE 越小越好，R² 越大通常越好；
- 能区分训练集与验证集；
- 今天不打开 ESOL 完整 Notebook。

## 完整学习包（按顺序）

1. [概念：线性预测与 L2 正则化](01_concepts.md)
2. [算法推演：固定模型与 alpha 对照](02_algorithm_walkthrough.md)
3. [课程提供的可运行 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

`tutorial.ipynb` 是供应的人工数据教学示例，不是你的模型实验。个人运行、副本、笔记和 `alpha_sensitivity.csv` 应在实际学习时保存到 `experiments/day03_ridge/`。

## 今日产出

今天只要求两个核心产物：

1. 一张“输入×权重＋截距＝预测”的手算示意；
2. 一张不同 `alpha` 下训练 RMSE、验证 RMSE 和系数的结果表。

本页中的数据是为了理解算法而制作的人工小数据，不是真实粘合剂数据。

## 核心概念

### 1. 线性模型怎样预测

只有一个输入特征时，线性模型可以写成：

```text
预测值 = 截距 + 权重 × 输入值
```

多个特征时：

```text
预测值 = 截距
       + 权重1 × 特征1
       + 权重2 × 特征2
       + ...
```

训练的主要任务，就是根据训练样本确定截距和每个权重。

例如：

```text
截距 = 0.5
权重 = 2.0
输入 = 3.0

预测 = 0.5 + 2.0 × 3.0 = 6.5
```

### 2. 普通线性回归的问题

当特征很多、样本较少或特征互相相关时，模型可能得到非常大的系数，并过度依赖训练数据中的偶然变化。

### 3. Ridge 做了什么

Ridge 在“预测误差”之外，增加一项对大系数的惩罚：

```text
要尽量减小：

预测误差
+
alpha × 所有权重平方之和
```

因此 Ridge 也叫带 L2 正则化的线性回归。

### 4. `alpha` 的方向

| `alpha` | 直观含义 | 可能结果 |
|---:|---|---|
| 很小 | 很少限制权重 | 更接近普通线性回归 |
| 适中 | 适当压缩权重 | 可能提高泛化稳定性 |
| 很大 | 强烈压缩权重 | 可能过于简单而欠拟合 |

`alpha` 不是准确率，也不是神经网络的学习率。

## 分步骤任务

### 第一步：先手算一个预测

假设：

```text
截距 = 0.2
权重 = 0.9
输入 = 4
```

先算出预测值，再继续。

### 第二步：画出算法流程

用自己的话补全：

```text
训练 X 和训练 y
→ Ridge 学习 ______ 和 ______
→ 把验证 X 代入
→ 得到 ______
→ 与验证 y 比较
→ 计算 ______
```

### 第三步：运行一个固定 Ridge

先只使用 `alpha=1.0`。找到 `.fit()`、`.predict()`、`.coef_` 和 `.intercept_`。

### 第四步：再比较固定候选值

在看结果前写下：

```python
[0.01, 0.1, 1.0, 10.0, 100.0]
```

今天的扫描只用于观察算法行为，不把它表述为正式超参数调优。

### 第五步：解释欠拟合

观察 `alpha` 很大时，系数是否被明显压小，以及训练、验证 RMSE 怎样变化。

## 核心代码：最小跟做

先在一个代码单元中只运行固定的 `alpha=1.0`：

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([0.2, 1.1, 1.9, 3.2, 3.9, 5.1])

X_valid = np.array([[1.5], [3.5], [5.5]])
y_valid = np.array([1.4, 3.6, 5.4])

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
valid_prediction = model.predict(X_valid)
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_prediction))

print("weight:", model.coef_[0])
print("intercept:", model.intercept_)
print("valid RMSE:", valid_rmse)
```

能指出创建、训练、预测和评价四步后，才在同一个 Notebook 的下一个单元运行候选循环：

```python
import pandas as pd

alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]
records = []

for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_prediction))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_prediction))

    records.append({
        "alpha": alpha,
        "coefficient": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "train_rmse": float(train_rmse),
        "valid_rmse": float(valid_rmse),
    })

results = pd.DataFrame(records)
print(results)
```

新增语法只关注：

- `model.coef_`：训练后学到的权重；
- `model.intercept_`：训练后学到的截距；
- 名称末尾的 `_` 通常表示这个属性要在 `fit` 后才存在；
- `records.append(...)`：把每次候选实验的结果加入列表。

## 常见错误

- 还没调用 `fit` 就读取 `coef_`；
- 把 `alpha` 当成准确率或学习率；
- 认为 `alpha` 越大一定越好；
- 只看训练 RMSE；
- 同时改变数据和 `alpha`，导致无法公平比较；
- 把人工小数据结果写成 ESOL 或粘合剂研究结论；
- 今天提前加入 StandardScaler、Pipeline 或正式调参；
- 看见一条近似直线就认为所有真实化学关系都是线性的。

## 完成标准

- 能写出“截距＋权重×输入”的预测形式；
- 能解释训练主要在学习哪些数值；
- 能解释 L2 正则化为什么限制大系数；
- 能说明 `alpha` 变大的大致作用；
- 能从代码中找到创建、训练、预测和评价；
- 结果表包含全部事先确定的 `alpha`；
- 能指出过大 `alpha` 可能产生欠拟合；
- 没有把玩具数据结果当成研究结论。

## 自测问题

1. Ridge 属于线性模型还是树模型？
2. `fit` 以后，模型主要学到了什么？
3. Ridge 比普通线性回归多了什么约束？
4. `alpha` 增大时，系数通常受到怎样的影响？
5. 系数越接近零是否一定代表模型更好？
6. 为什么要同时查看训练和验证误差？
7. `model.coef_` 为什么要在 `fit` 之后读取？
8. 今天为什么暂时不讲标准化和 Pipeline？

## 导航

- 上一天：[Day 2 回归评价指标](../day02_metrics/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 4 决策树](../day04_decision_tree/README.md)
