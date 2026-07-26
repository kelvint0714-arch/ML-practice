# Day 5：从单棵树到随机森林

## 今天为什么学

单棵决策树容易因为训练样本稍有变化就形成不同的切分，也容易过拟合。

随机森林的核心思想是：

> 训练许多不完全相同的树，再汇总它们的预测。

今天先理解为什么“许多有差异的树取平均”可能更稳定，再运行固定的一棵树与一个森林。树数量扫描只是可选观察，不是今天的核心。

## 前置条件

- 已完成 [Day 4 决策树](../day04_decision_tree/README.md)；
- 能解释分裂、叶节点、树深和过拟合；
- 知道训练 RMSE 与验证 RMSE 的作用不同；
- 能读懂一个简单 `for` 循环；
- 今天继续使用人工小数据。

## 今日产出

今天只要求：

1. 一张“单棵树 vs 固定随机森林”的两行结果表；
2. 一段对 Bootstrap、随机特征和预测平均的中文解释；
3. 一句说明当前小实验能证明什么、不能证明什么。

## 核心概念

### 1. Bootstrap

假设原训练样本编号是：

```text
[1, 2, 3, 4, 5]
```

一棵树可能抽到：

```text
[1, 1, 3, 4, 5]
```

另一棵树可能抽到：

```text
[2, 2, 3, 4, 4]
```

这种允许重复抽取的方式叫有放回抽样。不同树因此会看到不同的训练样本组合。

### 2. 随机特征

每次寻找切分时，一棵树通常只查看随机选择的部分特征，而不是始终查看全部特征。

这进一步增加树与树之间的差异。

### 3. Bagging

随机森林属于 Bagging 思路：

```text
分别训练多棵有差异的树
→ 每棵树产生一个预测
→ 回归任务中把预测取平均
```

例如三棵树预测：

```text
2.0、2.5、3.0
```

森林最终预测：

```text
(2.0 + 2.5 + 3.0) ÷ 3 = 2.5
```

### 4. 为什么平均可能更稳定

如果不同树犯下的错误不完全相同，取平均可以抵消一部分偶然偏差。

如果所有树都完全一样，它们会犯相同错误，重复很多次也没有意义。因此随机森林有意制造树之间的差异。

### 5. `n_estimators`

`n_estimators` 表示森林中的树数量。

- 它不是树深；
- 它不是样本数；
- 它不是神经网络训练轮数；
- 树更多通常计算更慢，不保证验证性能持续明显提升。

## 分步骤任务

### 第一步：先画流程

不要看代码，先画：

```text
训练数据
├── 抽样与随机特征 → 树1 ┐
├── 抽样与随机特征 → 树2 ├→ 预测取平均
└── 抽样与随机特征 → 树3 ┘
```

### 第二步：固定公平对照

比较：

- 一棵限制深度的决策树；
- 一个固定100棵树的随机森林。

两者必须使用同一训练集、同一验证集和同一指标。

### 第三步：运行最小代码

只观察两行结果，不立即扫描许多参数。

### 第四步：解释结果

如果森林更好，说明它在这份小数据的当前验证样本上更好；不能写成随机森林永远优于单棵树。

### 可选：改变树数量

核心验收完成后，可以比较：

```python
[10, 50, 100, 300]
```

观察结果是否逐渐稳定。这个可选实验不阻挡进入 Day 6。

## 核心代码：最小跟做

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

X_train = np.array([
    [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0],
    [4.0, 0.0], [5.0, 1.0], [6.0, 0.0], [7.0, 1.0],
    [8.0, 0.0], [9.0, 1.0],
])
y_train = np.array([0.2, 1.1, 1.9, 3.2, 3.9, 5.1, 5.8, 7.2, 7.9, 9.1])

X_valid = np.array([
    [1.5, 0.0], [3.5, 1.0], [5.5, 0.0], [7.5, 1.0],
])
y_valid = np.array([1.4, 3.6, 5.4, 7.6])

models = {
    "one_tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        max_features=1,
        bootstrap=True,
        random_state=42,
    ),
}

records = []

for model_name, model in models.items():
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    records.append({
        "model": model_name,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_prediction))),
        "valid_rmse": float(np.sqrt(mean_squared_error(y_valid, valid_prediction))),
    })

results = pd.DataFrame(records)
print(results)
```

今天新增语法：

- `models = {...}`：用字典保存模型名称和模型对象；
- `models.items()`：每次同时取出一个名称和对应模型；
- `max_features=1`：这份数据有两列特征，每次切分只随机考虑其中一列；
- `bootstrap=True`：每棵树使用有放回抽样得到的训练样本。

## 常见错误

- 把 `n_estimators` 当成树深；
- 认为森林中的所有树都完全相同；
- 认为树越多性能一定越好；
- 对单树和森林使用不同数据或不同指标；
- 只报告表现更好的模型，隐藏对照；
- 把特征重要性直接说成化学因果关系；
- 使用最终测试集挑选树数量；
- 把人工小数据结果当成真实材料结论。

## 完成标准

- 能解释 Bootstrap 和有放回抽样；
- 能说明随机特征为什么让树产生差异；
- 能口述 Bagging 流程；
- 能解释森林回归怎样汇总预测；
- 能说明 `n_estimators=100` 的含义；
- 得到固定单树与固定森林的两行对照；
- 能说明树更多通常需要更多计算，但今天不要求写计时代码；
- 结论没有超出当前小数据。

## 自测问题

1. 随机森林为什么不直接复制100棵完全相同的树？
2. Bootstrap 是否允许同一个训练样本被重复抽到？
3. 回归随机森林怎样汇总多棵树的预测？
4. `n_estimators=100` 表示什么？
5. 树数量增加是否保证验证 RMSE 降低？
6. 为什么单树与森林必须使用同一批数据？
7. 随机森林与一棵很深的决策树有什么根本区别？
8. 特征重要性是否自动证明化学因果关系？

## 导航

- 上一天：[Day 4 决策树](../day04_decision_tree/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 6 梯度提升](../day06_gradient_boosting/README.md)
