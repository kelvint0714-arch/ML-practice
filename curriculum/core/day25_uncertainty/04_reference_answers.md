# Day 25 参考答案

## A. 形状题

1. `(3, 4)`；
2. `(4,)`，每个样本一个均值；
3. 得到 `(3,)`，描述每个模型跨四个样本预测值的离散程度，不是逐样本模型分歧。

## B. 手算题

均值为 1.0。总体方差为：

\[
((0)^2+(0.2)^2+(-0.2)^2)/3=0.026666\ldots
\]

总体标准差约为 0.1633。

## C. 概念题

1. 分歧只看成员间差异，共同偏差不会表现为分歧；
2. 成员只从模型随机性产生，没有建模实验重复测量过程；
3. 需要定义区间构造方法，并在独立校准/评估数据上验证接近名义覆盖率。

## D. 代码题

```python
def ensemble_summary(predictions):
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError("predictions must be 2-D")
    if predictions.shape[0] < 2:
        raise ValueError("at least two ensemble members are required")
    return (
        predictions.mean(axis=0),
        predictions.std(axis=0, ddof=0),
    )
```

## E. 泄漏题

`y_pool` 是选择时不可见的候选真实标签。把候选绝对误差加入分数等于先看答案再选题，会严重高估主动策略。

## F. 报告改写

> 这里的均值与标准差来自多个随机种子模型；标准差仅作为启发式模型分歧，尚未校准为具有 95% 覆盖保证的预测区间。

## G. 数据边界

1. 若先用外部 valid 的标签决定分歧算法、种子或阈值，Day 26 再用同一 valid 比较主动与随机策略，就会形成策略层面的重复使用。内部 scaffold holdout 把本日诊断限制在原 train，并保证拟合集与诊断集没有相同 scaffold group。
2. 不算。反复根据该 holdout 改规则会让它逐渐参与开发；必须记录尝试，并另设新的、预先定义的评价协议。课程外部 valid 也已在更早 Day 被查看，因此始终不是严格未见 test。
