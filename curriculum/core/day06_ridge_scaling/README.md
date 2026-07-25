# Day 6：Ridge、特征缩放与 Pipeline

## 今天为什么学

Ridge 是带 L2 正则化的线性回归模型。在线性模型中，特征尺度会影响系数和正则化的实际作用。

今天不仅比较缩放前后结果，更重要的是学习如何把“缩放”和“模型”绑在一个 Pipeline 中，避免验证信息进入训练过程。

## 前置条件

- 已完成 [Day 5 随机森林](../day05_random_forest/README.md)；
- 知道训练集只能用于学习模型参数；
- 知道验证集用于比较候选方案；
- 能解释 `fit` 和 `predict`；
- 已准备训练与验证数组。

## 今日产出

今天应完成：

1. 一个未缩放的 Ridge 基线；
2. 一个 `StandardScaler + Ridge` Pipeline；
3. 一组固定 `alpha` 候选实验；
4. 缩放前后验证指标比较；
5. 对“缩放是否必然提升 ECFP 模型”的谨慎结论。

这些是待执行任务，不能把代码骨架当成真实结果。

## 核心概念

### 1. 线性模型

线性回归尝试给每个输入特征分配一个系数，再把它们加起来形成预测。

“线性”描述模型组合输入的方式，不表示数据表本身必须是一条直线。

### 2. L2 正则化

Ridge 在拟合误差之外，对较大的系数增加惩罚。

它希望模型不要依赖过大的单个系数。

### 3. `alpha`

`alpha` 控制正则化强度：

- 值较小：更接近普通线性回归；
- 值较大：更强地压缩系数；
- 过大也可能导致欠拟合。

### 4. 标准化

`StandardScaler` 使用训练数据估计每列的均值和标准差，再变换输入。

关键规则：

> Scaler 只能在训练数据上 fit。

### 5. Pipeline

Pipeline 会按顺序执行：

```text
训练集拟合 scaler
→ 变换训练集
→ 拟合 Ridge
→ 用同一个 scaler 变换验证集
→ Ridge 预测
```

这样比手工处理更不容易泄漏。

## 分步骤任务

### 第一步：建立未缩放基线

使用 `Ridge(alpha=1.0)`，记录训练和验证指标。

### 第二步：建立 Pipeline

Pipeline 中先放 `StandardScaler`，再放 `Ridge`。不要在全部数据上先运行 scaler。

### 第三步：固定 alpha 候选

建议在看结果前固定：

```python
[0.01, 0.1, 1.0, 10.0, 100.0]
```

### 第四步：公平比较

所有候选使用相同 train、validation、标签空间和评价函数。

### 第五步：解释而非保证提升

如果缩放没有改善，仍然是有效实验结论。

## 核心代码骨架

```python
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]
records = []

for alpha in alpha_values:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    train_scores = regression_metrics(y_train, train_prediction)
    valid_scores = regression_metrics(y_valid, valid_prediction)

    records.append({
        "alpha": alpha,
        "train_rmse": train_scores["rmse"],
        "train_r2": train_scores["r2"],
        "valid_rmse": valid_scores["rmse"],
        "valid_r2": valid_scores["r2"],
    })

results = pd.DataFrame(records)
print(results.sort_values("valid_rmse"))
```

今天新增语法：

- `Pipeline([...])`：把多个处理步骤按顺序组合；
- `("scale", 对象)`：给 Pipeline 步骤命名；
- `.sort_values("valid_rmse")`：按指定列从小到大排序；
- 科学实验中排序只是方便阅读，不会自动证明第一行是最终模型。

## 常见错误

- 在 train 与 validation 合并后拟合 scaler；
- 先查看 test，再选择 alpha；
- 把 `alpha` 说成学习率；
- 认为缩放一定改善所有模型；
- 将验证 RMSE 最低直接表述为最终测试结果；
- 不保留未缩放 Ridge 作为对照；
- 不同 alpha 使用不同数据；
- 把线性模型表现差解释为所有线性方法都无效。

## 完成标准

- 能解释 Ridge 与普通线性回归的区别；
- 能解释 `alpha` 的方向；
- 能说明 scaler 为什么只能在训练集上 fit；
- 能解释 Pipeline 的执行顺序；
- 保留未缩放基线；
- 候选 alpha 在看结果前固定；
- 同时记录 train 和 validation；
- 没有使用 test 选择配置；
- 结论允许“缩放没有提升”。

## 自测问题

1. Ridge 增加了什么约束？
2. `alpha` 越大通常代表什么？
3. 为什么不能在全数据上先标准化？
4. Pipeline 如何减少数据泄漏风险？
5. ECFP 位为什么与普通连续理化性质不同？
6. 标准化是否保证验证 RMSE 下降？
7. 为什么要保留未缩放基线？
8. 排名第一是否等于已经得到最终模型？

## 导航

- 上一天：[Day 5 随机森林](../day05_random_forest/README.md)
- 下一天：[Day 7 梯度提升](../day07_gradient_boosting/README.md)
