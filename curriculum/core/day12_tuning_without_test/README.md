# Day 12：不偷看测试集的调参

> 状态：待学习。代码是学习骨架，不表示已经得到最佳参数。

## 今天为什么学

模型参数不同，结果可能不同，但“把测试集分数最高的参数留下”是错误做法。
那样会把测试集变成调参集，最终分数不再代表未知数据上的表现。

今天学习只在训练数据内部做交叉验证，确定参数后再查看外部验证集。
测试集继续保持未使用。

## 前置条件

- 完成 Day 11，知道 Pipeline 为什么防止预处理泄漏；
- 能解释训练集、验证集和测试集的不同职责；
- 知道 Ridge 的 `alpha` 控制正则化强度；
- 能读懂模型参数字典。

## 今日产出

完成学习后，你应该产生：

1. 一张参数候选表；
2. 一份训练集内部交叉验证结果；
3. 冻结参数后的外部验证指标；
4. 一段明确写出“测试集未参与搜索”的实验说明。

今天的目标不是找到宇宙中最好的参数，而是建立正确的选择程序。

本日不把选择结果自动传给 Day 13。因为如果先在同一组 K 折上选择 `alpha`，又把同一组折的成绩当作无偏模型性能，结果会偏乐观。Day 13 使用运行前重新声明的固定基线；更严格的“调参＋无偏比较”需要以后学习嵌套交叉验证。

## 核心概念

### 1. 参数与学到的参数不同

- 超参数：训练前由人设定，例如 Ridge 的 `alpha`；
- 模型参数：`fit()` 后从数据学到，例如回归系数；
- 调参：比较一组预先声明的超参数候选；
- 交叉验证：在训练数据内部轮流划分小训练折和验证折。

### 2. 三层边界

```text
训练集内部 K 折：选择 alpha
        ↓
外部验证集：检查冻结方案
        ↓
测试集：方案最终冻结后才允许一次性使用
```

外部验证结果不理想时，可以回到研究设计，但不能无限尝试后只汇报最好一次。

代码使用普通随机 `KFold` 练习调参流程，不能把它称为 scaffold 交叉验证。
若研究问题要求新骨架泛化，应另外准备带骨架分组的折索引。

### 3. 为什么 Pipeline 仍然重要

交叉验证每一折都必须重新学习填充值和缩放参数。
把预处理放进 Pipeline 后，`GridSearchCV` 会在每一折内正确重训整条流程。

## 分步骤任务

1. 先写出候选 `alpha`，不要运行一次就临时增加只对结果有利的值。
2. 建立包含填补、标准化和 Ridge 的 Pipeline。
3. 建立固定随机种子的 `KFold`。
4. 使用 `GridSearchCV`，输入只能是训练数据。
5. 查看每个候选的平均交叉验证 RMSE。
6. 记录 `best_params_`，然后停止修改候选表。
7. 用 `best_estimator_` 对外部验证集预测一次。
8. 写清训练内交叉验证和外部验证的区别。
9. 确认代码中没有测试数组。

## 核心代码骨架

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])

param_grid = {
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    return_train_score=True,
)

search.fit(X_train, y_train)
valid_pred = search.best_estimator_.predict(X_valid)
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

cv_table = pd.DataFrame(search.cv_results_)
cv_table["cv_rmse"] = -cv_table["mean_test_score"]

print(cv_table[["param_ridge__alpha", "cv_rmse"]])
print("frozen params:", search.best_params_)
print("external validation RMSE:", valid_rmse)
```

## 只解释今天新增的语法

- `"ridge__alpha"` 中两个下划线表示“Pipeline 中 ridge 步骤的 alpha 参数”。
- `KFold(...)` 描述训练集内部怎样分成五折。
- `GridSearchCV` 会组合候选参数并重复训练，不是一个新的预测模型原理。
- `best_params_` 是交叉验证选出的参数字典。
- `best_estimator_` 是使用最佳参数重新拟合后的完整 Pipeline。
- scikit-learn 的负误差评分越大越好；`cv_rmse = -mean_test_score` 把它恢复为越小越好的普通 RMSE。

## 常见错误

| 错误 | 后果 | 处理 |
|---|---|---|
| 把 `X_valid` 传给 `search.fit` | 外部验证失去独立性 | 搜索只接收训练数据 |
| 用测试分数选 `alpha` | 测试集被污染 | 测试集保持封存 |
| 在交叉验证前全表标准化 | 每一折发生泄漏 | 预处理留在 Pipeline 内 |
| 只保存最佳一行 | 无法审查搜索范围 | 保存完整 `cv_results_` |
| 把负评分当成负 RMSE | 解释方向错误 | 说明 scikit-learn 评分约定 |
| 把普通 KFold 写成 scaffold 划分 | 夸大化学泛化证据 | 准确记录折生成方式 |

## 完成标准

- [ ] 我能说清超参数与模型参数的区别；
- [ ] 搜索只使用了 `X_train` 和 `y_train`；
- [ ] 我保存了所有候选的交叉验证结果；
- [ ] 我能解释双下划线参数名；
- [ ] 我只在参数冻结后评价外部验证集；
- [ ] 我没有查看或保存测试分数。

## 自测问题

1. 为什么不能根据测试集 RMSE 选择 `alpha`？
2. 五折交叉验证是否会训练五次以上？
3. 为什么缩放器必须放进被搜索的 Pipeline？
4. `best_estimator_` 和 `best_params_` 分别是什么？
5. 外部验证表现变差时，能否删除不喜欢的结果？

## 上一天 / 下一天

- 上一天：[Day 11：用 Pipeline 防止数据泄漏](../day11_pipeline_leakage/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 13：公平比较多个模型](../day13_fair_comparison/README.md)
