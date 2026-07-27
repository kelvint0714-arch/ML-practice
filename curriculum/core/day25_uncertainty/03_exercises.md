# Day 25 练习

## A. 形状题

三个模型预测四个样本：

```python
prediction_matrix = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [1.2, 2.1, 2.8, 4.2],
    [0.8, 1.9, 3.2, 3.8],
])
```

1. 矩阵形状是什么？
2. `mean(axis=0)` 的形状是什么？
3. `std(axis=1)` 回答什么问题？

## B. 手算题

对第一个样本 `[1.0, 1.2, 0.8]` 手算均值与总体标准差。

## C. 概念题

1. 为什么所有模型都给出相近错误预测时，分歧仍会很低？
2. 为什么分歧不包含实验重复误差？
3. 什么证据才允许把区间称为“95%”？

## D. 代码题

编写 `ensemble_summary(predictions)`，接收形状 `(n_models, n_samples)` 的数组，返回逐样本均值和标准差；若少于 2 个模型则报错。

## E. 泄漏题

下面采集函数为什么错误？

```python
score = disagreement + np.abs(y_pool - mean_prediction)
query = np.argsort(score)[-5:]
```

## F. 报告改写

把“这些样本有 95% 的概率落在均值±2倍标准差内”改写为合格句子。

## G. 数据边界

1. 为什么 Day 25 要在原 train 内建立 scaffold 隔离诊断 holdout，而不直接用外部 valid 调整采集规则？
2. 若看过诊断结果后继续更换种子和模型，这个 holdout 还算一次性诊断数据吗？
