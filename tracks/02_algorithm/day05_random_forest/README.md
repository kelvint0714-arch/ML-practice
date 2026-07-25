# Day 5：从单棵树到随机森林

## 今天为什么学

单棵决策树容易受到训练样本变化影响，也容易过拟合。随机森林训练许多带有随机性的树，再把它们的预测取平均。

今天的重点不是背参数，而是理解：

> 多个不完全相同的模型共同预测，为什么可能比一棵树更稳定。

## 前置条件

- 已完成 [Day 4 决策树](../day04_decision_tree/README.md)；
- 能解释树深和过拟合；
- 能使用统一的 `regression_metrics`；
- 能读懂一个 `for` 循环；
- 已准备相同的 train 和 validation 数组。

## 今日产出

今天应完成：

1. 一个固定配置的随机森林；
2. 一组只改变树数量的实验；
3. 随机森林与单棵树的公平比较；
4. 训练时间和验证指标记录；
5. 一段关于稳定性与成本的分析。

任务完成前，不应预先写“随机森林一定最好”。

## 核心概念

### 1. Bootstrap

每棵树可以从原训练集进行有放回抽样。

因此不同树看到的训练样本组合不完全相同。

### 2. 随机特征子集

树在每次分裂时不一定查看全部特征，而是只查看随机选择的一部分。

这会增加树与树之间的差异。

### 3. Bagging

Bagging 的直观过程是：

```text
重复抽样
→ 分别训练多个模型
→ 汇总预测
```

回归随机森林通常对多棵树的预测取平均。

### 4. `n_estimators`

`n_estimators` 表示森林中的树数量。

增加树数量通常会提高稳定性，但训练时间和内存开销也会增加。

它不是树深，也不是神经网络训练轮数。

## 分步骤任务

### 第一步：固定单棵树参考

使用 Day 4 中事先确定的受限制决策树作为参考。

不要重新选择一个只对今天结果最有利的树。

### 第二步：固定森林基本参数

除 `n_estimators` 外，先固定 `min_samples_leaf`、`max_features`、`random_state` 和 `n_jobs`。

### 第三步：改变树数量

建议候选值：

```python
[10, 50, 100, 300]
```

### 第四步：记录耗时

树更多可能只带来很小性能变化，却需要更多计算，因此除了指标，也记录 `fit_seconds`。

### 第五步：写结果边界

今天只能比较当前固定验证集上的候选配置。

不能证明随机森林在任何材料数据上都优于决策树。

## 核心代码骨架

```python
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

tree_counts = [10, 50, 100, 300]
records = []

for count in tree_counts:
    model = RandomForestRegressor(
        n_estimators=count,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start_time

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    train_scores = regression_metrics(y_train, train_prediction)
    valid_scores = regression_metrics(y_valid, valid_prediction)

    records.append({
        "n_estimators": count,
        "fit_seconds": fit_seconds,
        "train_r2": train_scores["r2"],
        "valid_rmse": valid_scores["rmse"],
        "valid_r2": valid_scores["r2"],
    })

results = pd.DataFrame(records)
print(results)
```

今天新增语法：

- `time.perf_counter()`：读取适合测量短时间间隔的时钟；
- `结束时间 - 开始时间`：得到拟合耗时；
- `n_jobs=-1`：允许 sklearn 使用可用 CPU 核心；
- `"sqrt"`：每次分裂考虑总特征数平方根数量的候选特征。

## 常见错误

- 把 `n_estimators` 当成树深；
- 每个树数量同时使用不同随机种子；
- 树更多后只看性能，不记录耗时；
- 认为一棵树和一棵森林树的设置完全相同；
- 把特征重要性直接说成确定的化学因果关系；
- 只记录最好的配置，删除其他候选结果；
- 使用 test 挑选树数量；
- 因一次结果更好就声称算法普遍优越。

## 完成标准

- 能解释 Bootstrap 和 Bagging；
- 能说明为什么不同树需要保持差异；
- 能解释 `n_estimators`；
- 能解释 `max_features` 的直观作用；
- 结果包含训练、验证和耗时信息；
- 一次实验只主要改变树数量；
- 与单棵树比较时数据和指标一致；
- 没有读取 test 进行选择；
- 能写出性能与计算成本的权衡。

## 自测问题

1. 随机森林如何产生不同的树？
2. 为什么最终要汇总多棵树的预测？
3. `n_estimators=300` 表示什么？
4. 树数量增加是否一定明显提高验证性能？
5. `max_features="sqrt"` 为什么会增加树之间差异？
6. 为什么需要记录训练时间？
7. 随机森林特征重要性是否等于化学因果关系？
8. 怎样才算与单棵树进行了公平比较？

## 导航

- 上一天：[Day 4 决策树](../day04_decision_tree/README.md)
- 下一天：[Day 6 Ridge 与缩放](../day06_ridge_scaling/README.md)
