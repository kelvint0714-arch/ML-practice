# Day 9：交叉验证与 OOF 预测

## 今天为什么学

一次固定验证划分可能碰巧偏容易或偏困难。

交叉验证让训练区域中的不同样本轮流充当验证折，从而观察结果对划分的敏感程度。

OOF（out-of-fold）预测还有一个关键用途：

> 为每个训练样本产生一条来自“没有训练过它的模型”的预测。

## 前置条件

- 已完成 [Day 8 数据划分协议](../day08_split_protocol/README.md)；
- 能解释 train、validation 和 test；
- 会使用 NumPy 整数索引选取行；
- 能训练 Ridge 或其他 sklearn 回归器；
- 能调用统一回归指标函数。

## 今日产出

今天应完成：

1. 一个 5 折 `KFold` 对象；
2. 每一折的训练和验证行数；
3. 覆盖全部训练样本的 OOF 预测数组；
4. 一组 OOF 回归指标；
5. 对随机 KFold 在分子任务中的局限说明。

今天的 OOF 只在原训练区域内部产生，不使用原 test。

## 核心概念

### 1. K 折交叉验证

把训练区域分成 K 份。

每一轮：

```text
K-1 份用于 fit
→ 剩余 1 份用于验证
```

循环 K 次后，每份都恰好当过一次验证折。

### 2. 折内模型

每一折都必须创建一个新的、尚未训练的模型。

不能让上一折已经学到的参数进入下一折。

### 3. OOF 预测

OOF 数组长度等于训练样本数。

第 i 个位置保存模型在未用第 i 个样本训练时对它产生的预测。

## 分步骤任务

### 第一步：只使用原训练数组

本日把 `X_train`、`y_train` 作为交叉验证的完整区域。

不要把原 validation 或 test 混入。

### 第二步：创建空 OOF 数组

使用 `np.empty_like(y_train, dtype=float)` 创建同长度容器。

### 第三步：循环折索引

`KFold.split(X_train)` 每轮返回 `fit_indices` 和 `holdout_indices`。

### 第四步：训练全新折模型

使用 `clone(base_model)`，并只在当前 fit indices 上拟合。

### 第五步：填回原位置

把当前 holdout 预测放回 `oof_prediction[holdout_indices]`。

循环结束后，每个位置应被填写一次。

## 核心代码骨架

```python
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

base_model = Ridge(alpha=1.0)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

oof_prediction = np.full(len(y_train), np.nan, dtype=float)

fold_rows = []

for fold_number, (fit_indices, holdout_indices) in enumerate(cv.split(X_train), start=1):
    fold_model = clone(base_model)
    fold_model.fit(X_train[fit_indices], y_train[fit_indices])

    fold_prediction = fold_model.predict(X_train[holdout_indices])
    oof_prediction[holdout_indices] = fold_prediction

    fold_rows.append({
        "fold": fold_number,
        "fit_rows": len(fit_indices),
        "holdout_rows": len(holdout_indices),
    })

assert np.isfinite(oof_prediction).all()

fold_table = pd.DataFrame(fold_rows)
oof_scores = regression_metrics(y_train, oof_prediction)

print(fold_table)
print(oof_scores)
```

今天新增语法：

- `clone(model)`：复制模型设置，但不复制训练状态；
- `enumerate(..., start=1)`：循环时同时获得从 1 开始的编号；
- `X_train[fit_indices]`：按整数位置选择多行；
- `np.full(..., np.nan)`：先建立带缺失标记的容器，方便检查漏填位置。

## 常见错误

- 把原 test 加入交叉验证；
- 每一折重复使用同一个已拟合对象；
- 在全部训练数据拟合后再称预测为 OOF；
- 忘记把预测填回正确索引；
- OOF 数组仍有 NaN 却继续计算；
- 把普通随机 KFold 称为 scaffold split；
- 把折内验证均值当作严格最终测试结果；
- 用不同评价函数比较不同折。

## 完成标准

- 能解释 K 折的轮换过程；
- 每一折使用全新的模型；
- 每个训练样本恰好得到一条 OOF 预测；
- OOF 预测中没有 NaN 或无穷值；
- 能解释 `clone` 与普通变量赋值的区别；
- 没有把原 validation/test 混入折内训练；
- 能说明随机 KFold 的分子任务局限；
- 能说明 OOF 为何有助于后续混合模型避免泄漏；
- 不声称本日已经完成混合模型。

## 自测问题

1. 5 折交叉验证需要训练几次模型？
2. 每个样本会当几次 holdout？
3. OOF 的三个英文单词是什么？
4. 为什么每一折要使用新模型？
5. `clone` 会复制已经学到的系数吗？
6. 为什么先用 NaN 填充 OOF 数组？
7. 普通随机 KFold 对分子数据可能有什么问题？
8. OOF 预测与对训练集直接预测有什么本质区别？

## 导航

- 上一天：[Day 8 数据划分协议](../day08_split_protocol/README.md)
- 下一天：[Day 10 随机种子稳定性](../day10_seed_stability/README.md)
