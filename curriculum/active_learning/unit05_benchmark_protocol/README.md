# Unit 05：怎样公平判断主动学习策略好不好

> 状态：从“代码能循环”进入“结论能相信”。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [指标与实验走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

主动学习轨迹高度依赖初始点和随机性。展示一个漂亮例子不能证明策略有效；论文级基准必须固定预算、运行多个种子、保留 Random，并根据研究目标选择评价指标。

## 前置条件

- 完成 Unit 04；
- 理解均值、标准差和学习曲线；
- 不熟悉 `groupby/agg` 或回调时先看 [Python 语法速查](../shared/python_patterns.md)；
- 阅读 [统一实验协议](../shared/experiment_protocol.md)。

## 今日产出

1. 材料发现：Random、Greedy、UCB、EI 的多种子 regret 曲线；
2. 全局模型学习：Random、Uncertainty 的固定测试集 RMSE 曲线；
3. 可审计的候选 ID、预测、采集分数、Oracle 返回和模型版本；
4. 每轮均值、标准差与协议一致性检查；
5. 不夸大结论的中文摘要。

## 核心概念

优化任务常用：

\[
r_t=f(x^*)-\max_{i\le t}f(x_i)
\]

其中 \(r_t\) 是 simple regret。越接近 0 越好。

模型学习任务还要在固定测试集报告 MAE/RMSE，但测试集不能进入候选池或采集函数。

## 分步骤任务

1. 预先声明种子集合、初始样本数和总预算。
2. 为每个种子生成一次初始集，所有策略共享。
3. 对材料发现分别运行 Random、Greedy、UCB、EI。
4. 每轮保存 query 前字段、Oracle 返回、best-so-far 和 regret。
5. 对全局模型学习比较 Random 与最大不确定性，并只由 evaluator 读取固定测试标签。
6. 按 `strategy, round` 分别汇总 regret 和测试 RMSE。
7. 检查每个策略实际标签数相同。
8. 展示全轨迹和失败种子。
9. 写明两类目标不同，结果只适用于当前候选池与协议。

## 核心代码骨架

```python
summary = (
    runs.groupby(["strategy", "round"])
    .agg(
        regret_mean=("simple_regret", "mean"),
        regret_std=("simple_regret", "std"),
        n_runs=("run_seed", "nunique"),
    )
    .reset_index()
)
```

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 只展示最好种子 | 选择性报告 | 展示全部预声明种子 |
| 不同策略预算不同 | 不公平 | 固定标签总数 |
| EI 用更强模型 | 模块混杂 | 一次只换一个因素 |
| 只报告最终点 | 隐藏样本效率 | 报告完整曲线 |
| 用 regret 评价全局预测 | 目标错位 | 固定测试集报告 MAE/RMSE |

## 完成标准

- [ ] 所有策略共享候选池、初始集和预算；
- [ ] 至少多个预声明种子；
- [ ] Random 与 Greedy/主动策略均保留；
- [ ] 报告均值、标准差和完整曲线；
- [ ] 能区分“找最好候选”和“降低全局预测误差”；
- [ ] query log 满足统一实验协议；
- [ ] 结论没有推广到所有材料体系。

## 自测问题

1. 为什么主动学习需要 Random 基线？
2. simple regret 与 best-so-far 有何关系？
3. 优化任务只报告 RMSE 有什么问题？
4. 多种子平均为什么仍要保留逐次结果？
5. 一次只换一个模块是什么意思？

## 上一单元 / 下一单元

- 上一单元：[Unit 04](../unit04_multiround_loop/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一单元：[Unit 06：批量、多样性与约束](../unit06_batch_diversity_constraints/README.md)
