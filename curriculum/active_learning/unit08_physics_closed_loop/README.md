# Unit 08：物理先验、成本与真实实验闭环

> 状态：从离线池回放进入“怎样连接实验”的系统设计；不连接真实仪器。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [结构化 GP 与状态机走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

真实材料发现不只是模型排序。候选受到相图、配比、工艺、安全、成本和设备约束；推荐还要经过人工批准。可信物理知识可以进入代理模型或采集策略，但错误先验也会误导。

## 前置条件

- 完成 Unit 07；
- 理解 GP 均值/方差和约束过滤；
- 不熟悉回调和状态表修改时先看 [Python 语法速查](../shared/python_patterns.md)；
- 阅读 [论文地图](../PAPER_MAP.md) 中 P07–P10。

## 今日产出

1. 普通 GP、合理物理均值和错误物理均值的三组对照；
2. 成本感知采集分数；
3. `proposed → approved → completed` 状态表，并复用 Unit 06 的 failed/rejected 规则；
4. PINN 与主动学习关系说明；
5. 人工审核边界。

## 核心概念

结构化 GP：

\[
y(x)=m_{\text{physics}}(x)+r_{\text{GP}}(x)
\]

物理模型给总体趋势，GP 学残差和不确定性。采集函数仍负责选点。

## 分步骤任务

1. 写出简单且可解释的物理趋势函数。
2. 普通 GP 直接拟合目标。
3. 残差 GP 拟合 `y - physics_mean(X)`。
4. 为候选重建预测均值与标准差。
5. 加入可行性和实验成本。
6. 固定 proposed 候选，不自动执行。
7. 模拟人工批准与 Oracle 返回。
8. 比较正确先验、无先验和错误先验。

## 核心代码骨架

```python
residual_y = y_labeled - physics_mean(X_labeled)
residual_gp.fit(X_labeled, residual_y)
residual_mean, std = residual_gp.predict(X_pool, return_std=True)
structured_mean = physics_mean(X_pool) + residual_mean

score = (structured_mean + beta * std) / experiment_cost
score[~feasible_mask] = -np.inf
```

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 不可信经验当物理定律 | 先验误导 | 错误先验消融 |
| PINN 自动称为 AL | 概念错误 | 另接采集与反馈 |
| 推荐即自动实验 | 安全风险 | 人工批准状态 |
| 只优化性能不看成本 | 实验不可执行 | 成本/资源约束 |

## 完成标准

- [ ] 能解释结构化 GP 与普通 GP 差别；
- [ ] 无先验、合理先验和错误先验使用同规格 GP 完成消融；
- [ ] 不可行候选不会进入 proposed；
- [ ] proposed 未经批准不能调用真实 Oracle；
- [ ] PINN 被正确定位为代理/物理模块。

## 自测问题

1. 物理知识可以进入闭环哪些位置？
2. 错误先验为什么必须做消融？
3. CAMEO、CAMD、DP-GEN 的闭环模块有何不同？
4. PINN 为什么不是主动学习的同义词？
5. 算法推荐与实验指令之间需要什么审核？

## 上一单元 / 下一单元

- 上一单元：[Unit 07](../unit07_neural_graph_surrogates/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一单元：[Unit 09：综合演练与复现交接](../unit09_capstone/README.md)
