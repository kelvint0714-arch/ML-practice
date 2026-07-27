# Day 6：梯度提升树

## 今天为什么学

随机森林让许多树相对独立地训练，最后平均预测。梯度提升采用另一种思路：

> 先做一个简单预测，后面的树依次修正当前仍然存在的错误。

今天只学习这种“顺序修错”的算法直觉，并运行一个固定基线。学习率扫描是可选观察，早停和更复杂实现以后再学。

## 前置条件

- 已完成 [Day 5 随机森林](../day05_random_forest/README.md)；
- 能解释单棵树和 Bagging；
- 知道 MAE、RMSE 和 R² 的方向；
- 能区分训练表现与验证表现；
- 今天继续使用人工小数据。

## 完整学习包（按顺序）

1. [概念：顺序修错与学习率](01_concepts.md)
2. [算法推演：残差与逐阶段预测](02_algorithm_walkthrough.md)
3. [课程提供的可运行 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

课程 Tutorial 提供可复查示例输出，但不代表学习者已运行。个人 `staged_metrics.csv`、Notebook 和解释只在实际学习时保存到 `experiments/day06_gradient_boosting/`。

## 今日产出

1. 一张 Bagging 与 Boosting 的对比图；
2. 一个固定 `GradientBoostingRegressor` 结果；
3. 一段说明“后面的树在修正什么”的中文解释；
4. 可选的学习率对照表。

## 核心概念

### 1. 残差

残差可以先直观理解为：

```text
真实值 - 当前预测值
```

例如：

```text
真实值 = 5
当前预测 = 3
残差 = 2
```

说明当前模型还少预测了2。

### 2. Boosting 的顺序

```text
初始简单预测
→ 第1棵树学习当前错误
→ 更新总预测
→ 第2棵树学习剩余错误
→ 再更新
→ 重复
```

这些树不是彼此独立的。后面的树依赖前面模型留下的错误。

### 3. 学习率

`learning_rate` 控制每棵新树对总预测的贡献。

- 学习率小：每次修正更谨慎，通常需要更多树；
- 学习率大：每次修正更强，也可能更容易过拟合；
- 学习率不是准确率。

### 4. `n_estimators`

在这里，`n_estimators` 表示顺序加入多少棵弱树。

它和随机森林都出现“树的数量”，但训练关系不同：

| 随机森林 | 梯度提升 |
|---|---|
| 树相对独立 | 树按顺序依赖前面的错误 |
| 最后平均预测 | 每一步逐渐修正总预测 |
| 主要属于 Bagging | 属于 Boosting |

### 5. 为什么今天不用 XGBoost

XGBoost 也是梯度提升家族的一种实现，但包含更多工程功能和参数。

今天使用 sklearn 自带的 `GradientBoostingRegressor`，先理解算法主干，不增加安装与版本负担。

## 分步骤任务

### 第一步：不用代码画流程

画出：

```text
第一次预测 → 剩余错误 → 新树修正 → 新的剩余错误
```

### 第二步：比较 Bagging 与 Boosting

至少写出“树是否独立”和“怎样汇总预测”两个区别。

### 第三步：运行固定基线

只使用：

```text
learning_rate = 0.1
n_estimators = 100
max_depth = 2
```

先理解固定模型，再考虑改变参数。

### 可选：只改变学习率

核心验收完成后，可以固定其他设置并比较：

```python
[0.03, 0.1, 0.3]
```

不要同时改变学习率和树数量，否则不容易判断差异来自哪里。

## 核心代码：最小跟做

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([0.2, 1.1, 1.9, 3.2, 3.9, 5.1])
X_valid = np.array([[1.5], [3.5], [5.5]])
y_valid = np.array([1.4, 3.6, 5.4])

model = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=2,
    random_state=42,
)

model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
valid_prediction = model.predict(X_valid)

train_rmse = np.sqrt(mean_squared_error(y_train, train_prediction))
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_prediction))
valid_r2 = r2_score(y_valid, valid_prediction)

print("train RMSE:", train_rmse)
print("valid RMSE:", valid_rmse)
print("valid R²:", valid_r2)
```

今天新增语法很少：

- `GradientBoostingRegressor(...)`：创建梯度提升回归模型；
- `learning_rate=0.1`：每一步修正的缩放比例；
- `n_estimators=100`：顺序加入100棵弱树；
- 其余仍然是相同的 `fit → predict → metric` 骨架。

## 常见错误

- 把 Boosting 说成许多独立树简单平均；
- 把学习率当成准确率；
- 认为树更多一定更好；
- 同时修改多个主要参数；
- 使用最终测试集选择学习率；
- 把 sklearn 的梯度提升直接说成 XGBoost；
- 只看训练误差；
- 把人工数据结果当成 ESOL 或粘合剂结论。

## 完成标准

- 能解释残差的直观含义；
- 能口述“后面的树修正前面错误”；
- 能说出 Bagging 与 Boosting 的两个区别；
- 能解释 `learning_rate`；
- 能解释 `n_estimators`；
- 能从代码中找到创建、训练、预测和评价；
- 固定基线成功运行；
- 没有为了追求分数提前使用测试集或复杂库。

## 自测问题

1. Boosting 中后面的树主要学习什么？
2. 梯度提升的树是否相互独立？
3. 学习率变小后，通常为什么可能需要更多树？
4. `n_estimators=100` 与随机森林中的同名参数含义完全一样吗？
5. Bagging 与 Boosting 怎样汇总模型？
6. 为什么今天先不用 XGBoost？
7. 固定其他条件、只改变学习率属于什么实验原则？
8. 验证结果好是否已经等于最终测试结论？

## 导航

- 上一天：[Day 5 随机森林](../day05_random_forest/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 7 完整 ESOL 基线](../day07_integrated_baseline/README.md)
