# Day 4：决策树与过拟合

## 今天为什么学

决策树能把特征空间不断切分，并在叶节点给出预测。它容易理解，也容易把训练数据记得过于精细。

今天先用人工小数据观察树怎样切分，再通过控制变量实验真正看懂：

> 为什么训练成绩非常好，验证成绩却可能很差。

## 前置条件

- 已完成 [Day 3 Ridge](../day03_ridge/README.md)；
- 能调用 `fit`、`predict` 和评价函数；
- 知道 train 与 validation 的作用不同；
- 知道验证 RMSE 越小通常越好；
- 今天继续使用人工小数据，不需要先运行完整 ESOL Notebook。

## 完整学习包（按顺序）

1. [概念：切分、叶节点与过拟合](01_concepts.md)
2. [算法推演：只改变树深](02_algorithm_walkthrough.md)
3. [课程提供的可运行 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

课程 Tutorial 的预存输出只证明示例可运行。你的预判、实际运行表和结论应在学习当天保存到 `learning_outputs/day04_decision_tree/`，不能直接复制课程结果冒充个人证据。

## 今日产出

今天应完成：

1. 一组只改变 `max_depth` 的决策树实验；
2. 每个配置的 train 与 validation 指标放在同一行；
3. 一列训练—验证 R² 差距；
4. 一段过拟合判断；
5. 一段对树深如何改变欠拟合/过拟合的解释，而不是宣布“最佳参数”。

本页提供任务与代码骨架，不表示这些实验已经运行。

学习顺序仍然是：先画出切分规则，口头解释叶节点，再运行最小代码。不要先从参数表开始背。

今天比较深度只是观察算法行为，不属于正式调参或最终模型选择。正确的调参边界会在 Day 12 学习。

## 核心概念

### 1. 分裂

决策树每次选择一个特征和切分规则，把样本分到不同分支。

### 2. 叶节点

样本到达叶节点后，回归树通常使用该叶节点训练样本的目标均值作为预测。

### 3. `max_depth`

`max_depth` 限制树最多向下生长多少层。

- 深度小：模型简单，可能欠拟合；
- 深度很大：模型复杂，可能过拟合；
- `None`：不主动设置最大深度限制。

### 4. `min_samples_leaf`

它规定一个叶节点至少需要多少训练样本。

值增大通常会让叶节点更稳健，但也可能让模型过于简单。

### 5. 控制变量

今天先固定其他参数，只改变 `max_depth`。

不要同时改变深度、叶节点样本数和随机种子，否则无法判断变化来自哪里。

## 分步骤任务

### 第一步：写下实验问题

示例：

```text
在同一份人工小数据上，树深增加会怎样改变训练和验证性能？
```

### 第二步：提前固定观察值

在看结果前写下：

```python
depth_values = [2, 5, 10, None]
```

不要看到结果后不断追加只为了得到更好数字。

### 第三步：循环训练

每棵树使用同一训练数据、同一验证数据、同一评价函数和同一随机种子。

### 第四步：整理对照表

每个深度记录一行，同时放入 train、valid 和两者的 R² 差距。

### 第五步：判断过拟合

同时查看训练和验证，不要只按训练 R² 排名。

## 核心代码骨架

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

X_train = np.arange(12, dtype=float).reshape(-1, 1)
y_train = X_train.reshape(-1).copy()
y_train[5] = 20.0

X_valid = (np.arange(11, dtype=float) + 0.5).reshape(-1, 1)
y_valid = X_valid.reshape(-1)

def regression_metrics(actual, predicted):
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }

depth_values = [1, 2, 3, None]
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
        "max_depth": depth,
        "train_rmse": train_scores["rmse"],
        "valid_rmse": valid_scores["rmse"],
        "train_r2": train_scores["r2"],
        "valid_r2": valid_scores["r2"],
        "r2_gap": train_scores["r2"] - valid_scores["r2"],
    })

results = pd.DataFrame(records)
print(results)
```

今天新增语法：

- `for depth in depth_values`：依次把候选深度交给同一实验骨架；
- `records.append(...)`：把一次结果追加到列表；
- `None`：表示没有显式设置该上限，不是数字零。
- `y_train[5] = 20.0`：故意给训练数据加入一个离群点，用来观察复杂树怎样记忆异常；
- `r2_gap`：训练 R² 减去验证 R²，差距过大是过拟合信号之一。

这段代码是自包含的：从第一行开始运行，不依赖 ESOL Notebook 中已经存在的隐藏变量。

这份人工数据的正常关系接近 `y=x`，但训练集中故意放入一个离群点。深树可以把这个点记住，训练 R² 接近1；验证数据保持正常关系，因此深树的验证 R² 会明显下降。这是为了展示机制，不代表真实数据一定以同样方式过拟合。

## 常见错误

- 只记录验证成绩，无法观察过拟合；
- 只记录训练成绩并宣布最深的树最好；
- 在不同深度使用不同数据划分；
- 一次同时改多个超参数；
- 忘记固定 `random_state`；
- 把 `max_depth=None` 解释成深度为零；
- 根据一次验证结果声称找到了全局最佳模型；
- 把人工小数据结果直接写成 ESOL 或下游任务结论。

## 完成标准

- 能解释分裂和叶节点；
- 能说明 `max_depth` 如何控制复杂度；
- 能说明 `min_samples_leaf` 的直观作用；
- 结果表同时包含 train 和 valid；
- 能指出哪类结果表现出过拟合；
- 能区分过拟合和程序报错；
- 实验中一次只改变一个主要变量；
- 没有使用 test 观察或选择树深；
- 能用自己的话写三句结果分析。

## 自测问题

1. 回归树的叶节点通常输出什么？
2. 树越深是否一定越好？
3. `max_depth=None` 表示什么？
4. 为什么要同时查看训练和验证指标？
5. 训练 R² 很高、验证 R² 很低通常说明什么？
6. `min_samples_leaf` 增大后，模型通常更复杂还是更简单？
7. 为什么候选参数应尽量在看结果前确定？
8. 控制变量实验中哪些设置必须保持一致？

## 导航

- 上一天：[Day 3 Ridge](../day03_ridge/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 5 随机森林](../day05_random_forest/README.md)
