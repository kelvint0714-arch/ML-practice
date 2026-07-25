# Day 7：梯度提升树

## 今天为什么学

随机森林让许多树相对独立地训练，然后平均预测。梯度提升采用另一种思路：后面的弱模型逐步修正前面模型留下的错误。

今天使用 scikit-learn 自带的 `HistGradientBoostingRegressor`，因此不需要安装新软件包。

## 前置条件

- 已完成 [Day 6 Ridge 与缩放](../day06_ridge_scaling/README.md)；
- 能解释随机森林的 Bagging；
- 知道一次只改变一个主要变量；
- 会读取训练和验证 RMSE；
- 已准备相同的 ESOL train 和 validation 数组。

## 今日产出

今天应完成：

1. 一个固定的直方图梯度提升基线；
2. 一组只改变 `learning_rate` 的候选；
3. 每个候选的迭代次数和验证指标；
4. 与随机森林训练方式的文字比较；
5. 对 XGBoost 是否需要现在安装的判断。

本日不声称 HistGradientBoosting 与 XGBoost 完全相同。

## 核心概念

### 1. Boosting

Boosting 按顺序增加弱模型。

新模型关注当前组合模型仍未处理好的误差。

模型之间因此不是相互独立的。

### 2. 学习率

`learning_rate` 控制每一步新模型对总预测的贡献。

- 学习率小：每一步更谨慎，通常需要更多迭代；
- 学习率大：更新更快，也可能过早过拟合；
- 学习率不是模型准确率。

### 3. `max_iter`

在 `HistGradientBoostingRegressor` 中，`max_iter` 控制最多进行多少轮提升。

它与随机森林的 `n_estimators` 都和树数量有关，但训练关系不同。

### 4. Early Stopping

早停在模型长时间没有改善时停止继续增加迭代。

早停使用的数据仍属于模型开发过程，不能使用最终 test。

### 5. 与 XGBoost 的关系

两者都属于梯度提升树家族，但实现、参数和功能并不完全相同。

今天的必做任务只使用 sklearn。

XGBoost 留作可选扩展，安装和锁定版本后才能进入正式比较。

## 分步骤任务

### 第一步：固定候选学习率

例如：

```python
[0.03, 0.05, 0.1, 0.2]
```

### 第二步：固定其他设置

保持 `max_iter`、叶节点数、L2 正则化和随机种子不变。

### 第三步：逐个训练

对每个候选记录训练与验证指标。

### 第四步：记录实际迭代数

早停后可以读取 `model.n_iter_`。

实际迭代数可能小于 `max_iter`。

### 第五步：与随机森林比较

重点比较训练方式，而不只比较某一个 RMSE。

## 核心代码骨架

```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

learning_rates = [0.03, 0.05, 0.1, 0.2]
records = []

for rate in learning_rates:
    model = HistGradientBoostingRegressor(
        learning_rate=rate,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    train_scores = regression_metrics(y_train, train_prediction)
    valid_scores = regression_metrics(y_valid, valid_prediction)

    records.append({
        "learning_rate": rate,
        "actual_iterations": model.n_iter_,
        "train_rmse": train_scores["rmse"],
        "valid_rmse": valid_scores["rmse"],
        "valid_r2": valid_scores["r2"],
    })

results = pd.DataFrame(records)
print(results.sort_values("valid_rmse"))
```

今天新增语法：

- `model.n_iter_`：拟合后保存的实际迭代轮数；
- 名称末尾的下划线常表示“经过 fit 后才产生的属性”；
- `validation_fraction=0.15`：从训练数据内部留一部分供早停判断；
- 这部分不是仓库的最终 test。

## 常见错误

- 把 Boosting 说成多棵独立树简单平均；
- 把学习率当成准确率；
- 同时改变学习率、树规模和正则化；
- 认为 `max_iter=300` 就一定执行 300 轮；
- 使用 test 做 early stopping；
- 声称 sklearn 的模型就是 XGBoost；
- 为了课程必做内容直接增加未经验证的依赖；
- 只报告最优候选，隐藏其他配置。

## 完成标准

- 能解释 Bagging 与 Boosting 的区别；
- 能说明学习率与迭代轮数的关系；
- 能解释 `max_iter`；
- 能读取并解释 `n_iter_`；
- 早停不使用 test；
- 所有候选保持相同数据与指标；
- XGBoost 只被标为可选扩展；
- 能写出梯度提升的优势和风险；
- 不声称实验已经迁移到粘合剂数据。

## 自测问题

1. Boosting 中后一个模型主要在做什么？
2. 学习率减小时通常需要怎样改变迭代轮数？
3. `max_iter` 与 `n_iter_` 有什么区别？
4. 早停为什么不能使用 test？
5. 随机森林与梯度提升的树之间关系有何不同？
6. HistGradientBoosting 是否等同于 XGBoost？
7. 今天为什么优先使用 sklearn 自带实现？
8. 如果复杂模型不如随机森林，实验是否失败？

## 导航

- 上一天：[Day 6 Ridge 与缩放](../day06_ridge_scaling/README.md)
- 下一天：[Day 8 数据划分协议](../day08_split_protocol/README.md)
