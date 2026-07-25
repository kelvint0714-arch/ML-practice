# Day 4：决策树与过拟合

## 今天为什么学

决策树能把特征空间不断切分，并在叶节点给出预测。它容易理解，也容易把训练数据记得过于精细。

Day 1 已经出现一棵不限深树和一棵受限制树。今天要通过控制变量实验真正看懂：

> 为什么训练成绩非常好，验证成绩却可能很差。

## 前置条件

- 已完成 [Day 3 评价指标](../day03_metrics/README.md)；
- 能调用 `fit`、`predict` 和评价函数；
- 知道 train 与 validation 的作用不同；
- 知道验证 RMSE 越小通常越好；
- 已能得到 `X_train`、`y_train`、`X_valid`、`y_valid`。

## 今日产出

今天应完成：

1. 一组只改变 `max_depth` 的决策树实验；
2. 每个配置的 train 与 validation 指标；
3. 一列训练—验证 R² 差距；
4. 一段过拟合判断；
5. 一个你能够解释的参数选择，而不是只报最好数字。

本页提供任务与代码骨架，不表示这些实验已经运行。

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
在同一 ESOL 划分上，树深增加会怎样改变训练和验证性能？
```

### 第二步：提前固定候选值

在看结果前写下：

```python
depth_values = [2, 5, 10, None]
```

不要看到结果后不断追加只为了得到更好数字。

### 第三步：循环训练

每棵树使用同一训练数据、同一验证数据、同一评价函数和同一随机种子。

### 第四步：整理长表

每个深度记录两行：

- train；
- valid。

### 第五步：判断过拟合

同时查看训练和验证，不要只按训练 R² 排名。

## 核心代码骨架

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

depth_values = [2, 5, 10, None]
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
        "split": "train",
        **train_scores,
    })
    records.append({
        "max_depth": depth,
        "split": "valid",
        **valid_scores,
    })

results = pd.DataFrame(records)
print(results)
```

今天新增语法：

- `for depth in depth_values`：依次把候选深度交给同一实验骨架；
- `records.append(...)`：把一次结果追加到列表；
- `**train_scores`：把指标字典展开进当前结果字典；
- `None`：表示没有显式设置该上限，不是数字零。

## 常见错误

- 只记录验证成绩，无法观察过拟合；
- 只记录训练成绩并选择最深的树；
- 在不同深度使用不同数据划分；
- 一次同时改多个超参数；
- 忘记固定 `random_state`；
- 把 `max_depth=None` 解释成深度为零；
- 根据一次验证结果声称找到了全局最佳模型；
- 把 ESOL 树模型结果直接写成粘合剂项目结论。

## 完成标准

- 能解释分裂和叶节点；
- 能说明 `max_depth` 如何控制复杂度；
- 能说明 `min_samples_leaf` 的直观作用；
- 结果表同时包含 train 和 valid；
- 能指出哪类结果表现出过拟合；
- 能区分过拟合和程序报错；
- 实验中一次只改变一个主要变量；
- 没有使用 test 选择树深；
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

- 上一天：[Day 3 评价指标](../day03_metrics/README.md)
- 下一天：[Day 5 随机森林](../day05_random_forest/README.md)
