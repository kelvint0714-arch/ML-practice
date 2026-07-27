# Day 11：用 Pipeline 防止数据泄漏

> 状态：待学习。本文是任务说明，不代表实验已经完成。

## 学习文件导航

按顺序完成以下五个课程文件：

1. [概念讲义](01_concepts.md)
2. [算法推演](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习题](03_exercises.md)
5. [参考答案](04_reference_answers.md)

`tutorial.ipynb` 通过确定性人工数据对照错误与正确统计量，只是课程演示。学习者亲自运行后的证据应另存到 `experiments/day11_pipeline_leakage/`。

## 今天为什么学

模型不仅会从标签中学习，缺失值填补和标准化也会从数据中学习统计量。
如果先在全部数据上计算均值、标准差，再划分训练集和验证集，验证集的信息就提前进入了训练流程。
这种现象叫数据泄漏，它会让验证成绩看起来比真实情况更好。

今天用 scikit-learn 的 `Pipeline` 把预处理和模型串在一起，确保所有需要学习的步骤只在训练数据上调用 `fit()`。

## 前置条件

- 能解释 `X_train`、`y_train`、`X_valid`、`y_valid`；
- 知道 `fit()` 学习参数，`predict()` 只做预测；
- 认识缺失值、均值和标准差；
- 已完成前一天的稳定性学习，或者至少理解单次分数不等于稳定结论。

## 今日产出

完成学习后，你应该得到：

1. 一张“错误流程与正确流程”对照图；
2. 一条包含填补、标准化和 Ridge 的 Pipeline；
3. 一份只含训练集与验证集指标的记录；
4. 三句话说明验证集为什么不能参与预处理器的 `fit()`。

这些产出需要你亲自运行和记录，仓库目前没有声称它们已经完成。

## 核心概念

### 1. 哪些步骤也会学习

| 步骤 | 从数据中学什么 | 能否提前看验证集 |
|---|---|---|
| `SimpleImputer` | 每一列的填充值 | 不能 |
| `StandardScaler` | 每一列的均值和标准差 | 不能 |
| 特征选择 | 哪些列更有用 | 不能 |
| Ridge | 回归系数 | 不能 |
| `predict()` | 不学习新参数 | 可以接收验证输入 |

### 2. Pipeline 的顺序

```text
原始 X
→ 缺失值填补
→ 标准化
→ Ridge
→ 预测值
```

调用 `pipeline.fit(X_train, y_train)` 时，三个步骤依次在训练数据上学习。
调用 `pipeline.predict(X_valid)` 时，前两步只使用已经学到的统计量转换验证数据。

### 3. 公平边界

验证集可以用来评价冻结后的流程，但不能用来计算填充值、缩放参数或模型系数。
测试集今天仍然不使用。

## 分步骤任务

1. 画出“先全表标准化再划分”的错误流程，并标出泄漏发生位置。
2. 检查当天数据变量的形状，确认训练与验证列数一致。
3. 创建 `SimpleImputer`、`StandardScaler` 和 `Ridge` 三个步骤。
4. 只对训练数据调用一次 Pipeline 的 `fit()`。
5. 分别生成训练预测与验证预测。
6. 使用相同的 MAE、RMSE、R² 口径记录结果。
7. 查看 `named_steps`，指出填补器、缩放器和模型分别在哪里。
8. 写下今天没有使用测试集的证据。

## 核心代码骨架

下面假设前一天已经准备好四个数组。不要把测试数组加入这段代码。

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

ridge_pipeline.fit(X_train, y_train)

train_pred = ridge_pipeline.predict(X_train)
valid_pred = ridge_pipeline.predict(X_valid)

def collect_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

train_metrics = collect_metrics(y_train, train_pred)
valid_metrics = collect_metrics(y_valid, valid_pred)

print("train:", train_metrics)
print("valid:", valid_metrics)
print("steps:", list(ridge_pipeline.named_steps))
```

## 只解释今天新增的语法

- `Pipeline([...])` 接收按顺序排列的步骤列表。
- `("imputer", 对象)` 是二元素元组：前面是步骤名，后面是执行对象。
- `named_steps` 按名称访问已经放入 Pipeline 的步骤。
- Pipeline 的 `fit()` 会依次调用预处理步骤的 `fit/transform`，最后训练模型。
- Pipeline 的 `predict()` 会转换输入，但不会重新计算训练集统计量。

`fit`、`predict`、函数定义和字典已经在前面的学习日出现，今天只需关注它们如何被 Pipeline 组织。

## 常见错误

| 错误 | 为什么有问题 | 正确处理 |
|---|---|---|
| 先对全部 `X` 调用 `fit_transform` | 验证信息进入缩放参数 | 划分后对 Pipeline 调用 `fit` |
| 验证集也调用 `fit()` | 模型和预处理被验证集更新 | 验证集只传给 `predict()` |
| 训练与验证分别新建缩放器 | 两边坐标尺度不一致 | 使用同一条已训练 Pipeline |
| 忘记处理缺失值 | Ridge 可能直接报错 | 把填补器放在最前面 |
| 今天查看测试分数 | 测试集变成调参依据 | 保持测试集未使用 |

## 完成标准

- [ ] 我能指出泄漏发生在哪一步；
- [ ] 我能解释 Pipeline 中三个步骤的顺序；
- [ ] 我的代码只对训练数据调用 `fit()`；
- [ ] 我能打印训练与验证指标，但没有测试指标；
- [ ] 我能解释 `named_steps` 中每个名称对应什么；
- [ ] 我写出了“当前结果能说明什么、不能说明什么”。

## 自测问题

1. 为什么标准化也可能造成数据泄漏？
2. `pipeline.predict(X_valid)` 会重新计算验证集均值吗？
3. 为什么随机森林通常不要求标准化，但缺失值处理仍需遵守训练边界？
4. Pipeline 是否自动替你划分训练集和验证集？
5. 如果验证成绩很好，能否立即去测试集反复确认？

## 上一天 / 下一天

- 上一天：[Day 10：多随机种子稳定性](../day10_seed_stability/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 12：不偷看测试集的调参](../day12_tuning_without_test/README.md)
