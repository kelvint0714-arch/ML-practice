# Day 13 教程输出

- `artifact_kind: deterministic_synthetic_tutorial`
- `learner_evidence: false`
- 来源：[Day 13 教学 Notebook](../tutorial.ipynb)
- 协议：固定人工回归数据、随机种子 42、相同随机 5 折、五个预先声明的传统模型。

本目录保存课程源 Notebook 的小型教学 fixture，供 Day 14 演示跨日文件审计。
这些文件不是学习者在 `experiments/` 中亲自运行的证据，不是 ESOL 结果，也不是
真实粘合剂研究成果。

`fit_seconds` 只保留在 Notebook 运行时内存变量中，不进入预存展示、
`fold_metrics.csv` 或 `model_summary.csv`。墙钟时间受硬件与后台任务影响，
不是严格复现字段，也不能用于跨机器排名。
