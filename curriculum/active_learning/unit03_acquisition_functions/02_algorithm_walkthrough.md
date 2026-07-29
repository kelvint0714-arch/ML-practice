# Unit 03 算法走读：从预测表到排序

## UCB 手算

候选 A 的 \(\mu=8,\sigma=0.2\)，B 的 \(\mu=7,\sigma=1.5\)：

- \(\beta=0\)：A=8，B=7，选择 A；
- \(\beta=1\)：A=8.2，B=8.5，选择 B。

`beta` 不是模型参数，它控制采集阶段的探索权重。

## EI 变量

```python
improvement = mean - best_observed - xi
safe_std = np.maximum(std, 1e-12)
z = improvement / safe_std
ei = improvement * norm.cdf(z) + std * norm.pdf(z)
ei = np.where(std > 0, ei, np.maximum(improvement, 0.0))
```

- `best_observed` 必须来自 `y_labeled`；
- `np.maximum` 避免除以 0；
- `norm.cdf` 是累计概率；
- `norm.pdf` 是概率密度；
- `np.where` 使用零方差极限：若确定性预测已经超过阈值，EI 等于确定的改进，否则为 0。

## 固定排序

```python
order = np.lexsort((pool_ids.astype(str), -score))
query_positions = order[:batch_size]
```

`lexsort` 最后一个键是主要排序键：

1. 先按 `-score`，即分数从高到低；
2. 同分时按候选 ID；
3. 结果可以重复。

## 最小化任务

有两种写法：

1. 对目标乘 `-1`，统一按最大化处理；
2. 单独实现 LCB/最小化 EI。

无论哪种都要在配置中记录，不能一部分指标用原符号、一部分用反号。

## 边际近似 TS

```python
rng = np.random.default_rng(seed)
sampled_values = rng.normal(mean, std)
query_position = np.argmax(sampled_values)
```

这段代码把候选边际独立抽样，只适用于理解接口。标准 GP Thompson Sampling
应使用 `gp.predict(..., return_cov=True)` 得到联合协方差，再从多元正态分布
抽样。固定 `seed` 才能重现实验，但正式比较仍需多个预声明种子。
