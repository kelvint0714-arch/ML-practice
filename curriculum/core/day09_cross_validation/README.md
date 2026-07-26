# Day 9：K 折交叉验证

## 今天为什么学

一次固定验证划分可能碰巧偏容易或偏困难。

K 折交叉验证让训练区域中的不同样本轮流充当验证折，用来观察结果是否强烈依赖某一次划分。

今天只学习交叉验证。OOF 预测和 Stacking 到 Day 23 再一起学习，避免现在同时引入两个概念。

## 前置条件

- 已完成 [Day 8 数据划分协议](../day08_split_protocol/README.md)；
- 能解释训练、验证和测试；
- 知道模型只能在训练行上调用 `fit`；
- 能计算 RMSE；
- 知道每次训练应该使用新的模型对象。

## 今日产出

1. 一张5折轮换示意图；
2. 一张包含每折训练行数、验证行数和验证 RMSE 的表；
3. 交叉验证 RMSE 的均值与标准差；
4. 一段普通随机 KFold 在分子和配方数据上的局限说明。

## 核心概念

### 1. K 折怎样轮换

假设训练区域分成5份：

```text
第1轮：第1折验证，其余4折训练
第2轮：第2折验证，其余4折训练
第3轮：第3折验证，其余4折训练
第4轮：第4折验证，其余4折训练
第5轮：第5折验证，其余4折训练
```

因此：

- 一共训练5次模型；
- 每个样本恰好当一次折内验证样本；
- 每个样本在其他4轮中属于折内训练区域。

### 2. 每一折都是新模型

第2折不能继续使用第1折已经训练过的模型。

每轮必须创建一个新的、未训练模型，否则前一折的信息会进入下一折。

### 3. 均值不能隐藏折间差异

假设5折 RMSE 是：

```text
0.8、0.9、0.7、1.8、0.8
```

只报告平均值会隐藏第4折明显更难。因此必须保留逐折结果，并同时报告标准差。

### 4. 交叉验证仍不是最终测试

这些折用于开发和比较方案。最终测试集仍然保持封存。

### 5. 普通 KFold 的材料数据局限

相似分子、相同配方的重复试样或同一实验批次如果被拆到不同折，结果可能过于乐观。

真实粘合剂数据到达后，需要根据分子骨架、配方号、批次或实验组考虑分组划分。不能为了得到更好分数临时选择分组字段。

## 分步骤任务

### 第一步：纸上画5折

画5个方框，并让每个方框轮流成为验证折。

### 第二步：固定模型和折数

今天使用一个固定的浅决策树。不要同时扫描树深。

### 第三步：逐折训练

每轮：

```text
创建新模型
→ 当前4折 fit
→ 剩余1折 predict
→ 计算这一折 RMSE
```

### 第四步：保留逐折表

不能只打印均值。

### 第五步：写局限

说明随机 KFold 与 scaffold split、GroupKFold 的目的并不完全相同。

## 核心代码：最小跟做

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

X = np.arange(20, dtype=float).reshape(-1, 1)
y = 0.5 * X.reshape(-1) + np.sin(X.reshape(-1))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rows = []

for fold_number, (fit_indices, valid_indices) in enumerate(cv.split(X), start=1):
    model = DecisionTreeRegressor(max_depth=2, random_state=42)
    model.fit(X[fit_indices], y[fit_indices])

    prediction = model.predict(X[valid_indices])
    fold_rmse = np.sqrt(mean_squared_error(y[valid_indices], prediction))

    fold_rows.append({
        "fold": fold_number,
        "fit_rows": len(fit_indices),
        "valid_rows": len(valid_indices),
        "valid_rmse": float(fold_rmse),
    })

fold_table = pd.DataFrame(fold_rows)

print(fold_table)
print("mean RMSE:", fold_table["valid_rmse"].mean())
print("std RMSE:", fold_table["valid_rmse"].std())
```

今天新增语法：

- `KFold(...)`：建立折轮换规则；
- `cv.split(X)`：每轮返回训练行索引和验证行索引；
- `enumerate(..., start=1)`：同时得到从1开始的折编号；
- `X[fit_indices]`：按整数索引选择当前训练行；
- `.std()`：计算逐折结果的标准差。

## 常见错误

- 把最终测试集加入 KFold；
- 每一折重复使用同一个已拟合模型；
- 不保留逐折结果，只报告均值；
- 每一折使用不同算法参数；
- 把随机 KFold 称为 scaffold split；
- 在交叉验证同时扫描许多参数，却称为单纯稳定性检查；
- 认为交叉验证消除了所有数据泄漏；
- 把人工数据结果当成真实材料实验。

## 完成标准

- 能画出5折轮换；
- 能说明5折为什么要训练5次；
- 每一折创建全新模型；
- 每个样本恰好当一次折内验证样本；
- 表中保留5行逐折结果；
- 同时报告均值与标准差；
- 最终测试集没有参与；
- 能说明普通随机 KFold 对分子或重复配方的局限。

## 自测问题

1. 5折交叉验证需要训练几次？
2. 每个样本会当几次折内验证样本？
3. 为什么每一折要创建新模型？
4. 为什么不能只报告平均 RMSE？
5. 标准差很大说明什么？
6. 交叉验证结果是否等于最终测试结果？
7. 相似分子随机分到不同折可能造成什么问题？
8. OOF 为什么暂时留到 Day 23？

## 导航

- 上一天：[Day 8 数据划分协议](../day08_split_protocol/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 10 随机种子稳定性](../day10_seed_stability/README.md)
