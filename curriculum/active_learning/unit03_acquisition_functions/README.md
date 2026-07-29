# Unit 03：采集函数怎样决定下一次测谁

> 状态：在同一预测表上比较策略。先手算，再运行代码。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [算法与手算走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

模型只给预测分布，采集函数才把“预测好”和“模型没把握”变成选样顺序。不同目标、噪声和预算下，不存在永远最好的采集函数。

## 前置条件

- 完成 Unit 02；
- 能解释预测均值和标准差；
- 会看正态分布的概率和累积分布概念。

## 今日产出

1. Greedy、Uncertainty、UCB、PI、EI 与边际近似 TS 分数表；
2. 一次 UCB 与 EI 手算；
3. `beta`/`xi` 改变排序的对照；
4. 最大化与最小化方向检查。

## 核心概念

最大化任务常用：

\[
\mathrm{UCB}(x)=\mu(x)+\beta\sigma(x)
\]

\[
\mathrm{EI}(x)=
(\mu-y^+-\xi)\Phi(z)+\sigma\phi(z)
\]

其中 \(z=(\mu-y^+-\xi)/\sigma\)，\(y^+\) 是已观测最佳值。

## 分步骤任务

1. 固定一张候选预测表。
2. 计算 Greedy 与最大不确定性排序。
3. 分别用两个 `beta` 计算 UCB。
4. 使用已观测最佳值计算 PI/EI。
5. 用固定随机种子演示独立边际近似 TS，并写清它不是联合后验抽样。
6. 给所有策略固定候选 ID 并列规则。
7. 检查目标是最大化还是最小化。

## 核心代码骨架

```python
ucb = mean + beta * std
z = (mean - best_observed - xi) / safe_std
pi = norm.cdf(z)
ei = (mean - best_observed - xi) * norm.cdf(z)
ei += std * norm.pdf(z)
```

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 最小化仍用 `argmax(mean)` | 方向相反 | 转换符号或用 LCB |
| `best_observed` 来自池标签 | 泄漏 | 只用已标注集 |
| `std=0` 直接除 | 数值错误 | 使用 `safe_std` |
| 只有均值/标准差却称标准 GP-TS | 忽略候选相关性 | 明确写“独立边际近似” |
| 结果后调 `beta` | 事后选择 | 预先声明 |

## 完成标准

- [ ] 能独立计算 UCB；
- [ ] 能解释 EI 的“改进幅度 × 可能性”；
- [ ] 能说明 `beta` 和 `xi` 的影响；
- [ ] `best_observed` 只来自已标注集；
- [ ] 最小化/最大化方向无误。

## 自测问题

1. Greedy 为什么容易停止探索？
2. 最大不确定性为什么可能不找到最好材料？
3. UCB 中 `beta` 越大意味着什么？
4. EI 与 PI 的差别是什么？
5. Thompson Sampling 为什么每次可能选不同候选？

## 上一单元 / 下一单元

- 上一单元：[Unit 02](../unit02_surrogates_uncertainty/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一单元：[Unit 04：多轮循环](../unit04_multiround_loop/README.md)
