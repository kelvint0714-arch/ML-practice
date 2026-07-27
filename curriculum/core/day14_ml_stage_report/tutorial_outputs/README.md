# Day 14 教程输出

- `artifact_kind: deterministic_synthetic_tutorial`
- `learner_evidence: false`
- 来源：[Day 14 教学 Notebook](../tutorial.ipynb)
- 上游：[Day 13 课程 fixture](../../day13_fair_comparison/tutorial_outputs/README.md)
- 协议：审计 Day 13 固定人工数据的随机 5 折结果，再生成汇总和有限定语的报告预览。

本目录只保存课程源 Notebook 生成的教学 fixture。它们不是学习者在
`experiments/` 中亲自运行的证据，不是 ESOL 结果，也不是真实粘合剂研究成果。

Day 13 的 `fit_seconds` 只保留在运行时内存变量中，不进入预存展示或本目录的
持久化 fixture；墙钟时间不是严格复现字段，也不得用于跨机器模型排名。
