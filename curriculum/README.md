# ML-Learning 课程总览

课程分为三条互相衔接的路线：

- [Day 1–28 核心路线](core/README.md)：传统机器学习、可信评价、MLP、混合模型和总结报告；
- [Day 29–35 可选 GNN](optional_gnn/README.md)：图数据、GCN、GraphSAGE、GAT、GIN 和数据就绪检查；
- [主动学习 Unit 1–9](active_learning/README.md)：代理模型、采集函数、多轮、批量和约束。

如果不确定从哪里开始，先使用 [学习路线选择器](LEARNING_PATHS.md)。实际完成状态只记录在 [唯一学习清单](PROGRESS.md)。

## 教学顺序

```text
算法概念
→ 纸笔推演
→ 最小代码
→ 教学 Notebook
→ 独立练习
→ 参考答案
→ 学习总结
```

Day 2–35 和 Unit 1–9 都提供中文讲义、算法走读、练习、答案和可运行 Notebook。完整 ESOL 基线是 Day 7 综合材料，不是学习起点。

## 共享参考

- [学习包使用方法](shared/day_package_guide.md)
- [机器学习术语表](shared/ml_glossary.md)
- [统一评价协议](shared/experiment_protocol.md)
- [常见错误排查](shared/error_guide.md)
- [主动学习术语表](active_learning/shared/glossary.md)
- [主动学习标签泄漏检查](active_learning/shared/leakage_checklist.md)

## 环境

- 核心路线：[`requirements-learning.txt`](../requirements-learning.txt)
- 主动学习：[`requirements-active-learning.txt`](../requirements-active-learning.txt)
- GNN：[`requirements-gnn.txt`](../requirements-gnn.txt)

## 比较规则

- 同一比较使用相同数据、输入、划分和指标；
- 预处理器只能在训练数据上拟合；
- 模型选择不能反复查看测试集；
- 保留 Dummy 和简单模型；
- 报告逐折或逐种子结果，而不是只报最佳数字；
- OOF 特征必须来自训练折外预测；
- 教学数据结果不能直接外推到新任务。

## 核心结课产物

1. 一套传统模型可信比较；
2. 一个与传统模型同协议的 MLP 基线；
3. 一个无泄漏 OOF Stacking 对照；
4. 一次不确定性与单轮主动学习模拟；
5. 一个数据字典、模型卡和课程总报告。
