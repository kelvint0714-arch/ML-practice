# Unit 06：批量选择、多样性与实验约束

> 状态：从一次选一个候选，进入真实下游任务常见的批次实验。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [批量算法走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

材料实验通常按批次进行。如果只取采集分数最高的前 10 个，它们可能几乎相同；不可制备或危险候选也可能排在前面。真实 query 必须结合多样性、可行性、成本和人工审核。

## 前置条件

- 完成 Unit 05；
- 理解距离和标准化；
- 不熟悉 `.loc`、`zip` 或并列排序时先看 [Python 语法速查](../shared/python_patterns.md)；
- 知道约束只能使用实验前可知信息。

## 今日产出

1. 一个可行性掩码；
2. top-score 批次；
3. score + diversity 批次；
4. 与同预算 Random 的批内最小/平均距离对照；
5. proposed、completed、failed、rejected 状态记录。

## 核心概念

批量选择可以使用贪心方法：

```text
先选采集分数最高候选
→ 对剩余候选计算与已选批次的最小距离
→ 综合采集分数和距离
→ 继续选择直到 batch 满
```

可行性过滤必须发生在 query 前，且规则由化学/实验知识提供。

## 分步骤任务

1. 用已知组成和工艺规则建立 `feasible_mask`。
2. 先删除不可行候选。
3. 对可行池计算采集分数。
4. 生成纯 top-score 批次。
5. 生成兼顾多样性的贪心批次。
6. 比较批内最小/平均距离。
7. 保证 Random 也从同一可行池、同预算选择。
8. 设计实验失败状态，不删除失败行。

## 核心代码骨架

```python
eligible = np.flatnonzero(feasible_mask)
order = np.lexsort((candidate_ids[eligible], -score[eligible]))
selected = [eligible[order[0]]]

while len(selected) < batch_size:
    distance = pairwise_distances(
        X_pool[eligible], X_pool[selected]
    ).min(axis=1)
    combined = normalized_score[eligible] + weight * distance
    combined[np.isin(eligible, selected)] = -np.inf
    order = np.lexsort((candidate_ids[eligible], -combined))
    selected.append(eligible[order[0]])
```

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 先选后过滤 | 批次不足或替换无记录 | 先定义可行池 |
| 未标准化直接算距离 | 大尺度特征主导 | 训练规则内标准化 |
| 删除失败实验 | 幸存者偏差 | 保留状态和原因 |
| Random 不受约束 | 对照不公平 | 共享可行池 |

## 完成标准

- [ ] 所有入选候选满足可行性规则；
- [ ] 批次没有重复编号；
- [ ] 多样性计算不读取隐藏标签；
- [ ] 与纯 top-score 和 Random 同预算比较；
- [ ] 同时报告最小距离和平均距离，不因最小距离并列而误判；
- [ ] 失败和拒绝状态有记录。

## 自测问题

1. 为什么直接取 top-b 容易冗余？
2. 组成比例约束与性能标签有什么权限区别？
3. 为什么距离前通常要标准化？
4. 成本不同的实验怎样进入采集规则？
5. 失败实验为什么不能无痕删除？

## 上一单元 / 下一单元

- 上一单元：[Unit 05](../unit05_benchmark_protocol/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一单元：[Unit 07：神经网络与图代理](../unit07_neural_graph_surrogates/README.md)
