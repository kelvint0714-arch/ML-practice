# Unit 04：从单轮 query 到多轮主动学习

> 状态：把 Day 26 的单轮演示升级成可重复循环。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [算法与状态走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

主动学习的价值来自“选择—实验—更新”的多轮积累。单轮成功可能只是偶然；多轮代码还必须管理已标注索引、剩余候选、停止条件和逐轮日志。

## 前置条件

- 完成 Unit 03；
- 能使用循环、列表和 DataFrame；
- 不熟悉函数参数或列表推导时先看 [Python 语法速查](../shared/python_patterns.md)；
- 通过 [标签泄漏检查表](../shared/leakage_checklist.md)。

## 今日产出

1. 一个 `run_active_learning()` 循环；
2. 每轮 query 前预测和采集分数；
3. query 后 Oracle 返回值；
4. best-so-far 和 simple regret 轨迹；
5. 同初始集的 Random 轨迹。

## 核心概念

每轮状态由两组互斥索引组成：

```text
labeled_indices ∩ pool_indices = ∅
labeled_indices ∪ pool_indices = 全部候选
```

选择后要把同一个候选从 pool 移到 labeled，不能复制、遗漏或重复查询。

## 分步骤任务

1. 固定候选池、隐藏 Oracle、初始索引和预算。
2. 只使用已标注索引训练模型。
3. 只对剩余候选计算预测和采集分数。
4. 固定 query 及逐轮日志的 query 前字段。
5. 调用 Oracle 并更新索引。
6. 计算更新后的 best-so-far。
7. 循环到预算耗尽或池为空。
8. 用完全相同的初始索引运行 Random。

## 核心代码骨架

```python
for round_number in range(total_rounds):
    model.fit(X[labeled_indices], y_labeled)
    mean, std = model.predict(X[pool_indices], return_std=True)
    query_local = select_with_tie_break(
        mean + beta * std,
        candidate_ids[pool_indices],
    )
    query_global = pool_indices[query_local]

    # 标签揭示线
    observed_y = oracle(query_global)
    labeled_indices.append(query_global)
    y_labeled.append(observed_y)
    pool_indices.remove(query_global)
```

教学代码把完整标签封装在离线 Oracle 中；策略函数只接收已标注值和
`oracle(query_global)` 回调。全局最优由 campaign 结束后的 evaluator 读取。

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 局部位置当永久 ID | 更新错样本 | 映射到 global index |
| 选后未从池移除 | 重复实验 | 保持集合互斥 |
| 只保存最终分数 | 无法审计 | 保存逐轮日志 |
| 主动与 Random 初始点不同 | 不公平 | 共用 initial indices |

## 完成标准

- [ ] 多轮循环可用固定种子重现；
- [ ] 同一候选不会查询两次；
- [ ] 每轮 query 前后字段清晰；
- [ ] Random 使用同初始集和预算；
- [ ] 日志能重建整个选择轨迹。

## 自测问题

1. `query_local` 和 `query_global` 有什么区别？
2. 为什么每轮通常要重新训练模型？
3. 什么情况下循环应停止？
4. best-so-far 为什么单调不下降（最大化任务）？
5. 为什么只保存最终最好值不够？

## 上一单元 / 下一单元

- 上一单元：[Unit 03](../unit03_acquisition_functions/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一单元：[Unit 05：公平基准协议](../unit05_benchmark_protocol/README.md)
