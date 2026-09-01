# Day 1–28 核心机器学习路线

这条路线从算法直觉开始，逐步进入可信评价、神经网络、混合模型、主动学习和模型报告。真正执行和勾选只使用 [唯一学习清单](../PROGRESS.md)。

```text
概念与手算
→ 最小代码
→ 传统模型基线
→ 可信评价
→ MLP
→ 无泄漏混合模型
→ 不确定性与主动学习
→ 模型卡和总报告
```

ESOL 是贯穿部分章节的公开回归教学数据，目标是水溶解度 `logS`。人工数据用于隔离单个概念；任何教程结果都不能自动外推到新的真实任务。

## 怎样学习

1. 从任务卡确认今天的问题和前置条件；
2. 按“概念 → 推演 → Notebook → 练习 → 答案”学习；
3. 每段代码都回答输入、变换和输出；
4. 练习先独立完成，再打开参考答案；
5. 个人笔记和输出保存在本地 `learning_outputs/`；
6. 测试集只在方案冻结后使用；
7. 能解释结果和限制后，才更新进度表。

## 第一阶段：机器学习基础（Day 1–7）

| Day | 主题 | 核心产物 |
|---:|---|---|
| [1](day01_beginner/README.md) | 算法、样本、`X/y`、训练与预测 | 手算、伪代码、最小代码 |
| [2](day02_metrics/README.md) | MAE、RMSE、R² | 误差表与指标解释 |
| [3](day03_ridge/README.md) | 线性回归、Ridge、L2 | `alpha` 敏感性表 |
| [4](day04_decision_tree/README.md) | 决策树与过拟合 | 深度对照 |
| [5](day05_random_forest/README.md) | 随机森林与 Bagging | 单树/森林对照 |
| [6](day06_gradient_boosting/README.md) | Gradient Boosting | 顺序修错演示 |
| [7](day07_integrated_baseline/README.md) | 完整 ESOL 基线 | 可追溯基线结果 |

Day 1–6 使用极小例子建立直觉；Day 7 才把这些概念映射回完整、可复现的公开数据流程。

## 第二阶段：可信评价（Day 8–14）

| Day | 主题 | 核心产物 |
|---:|---|---|
| [8](day08_split_protocol/README.md) | 数据划分与测试集边界 | split 检查表 |
| [9](day09_cross_validation/README.md) | K 折交叉验证 | 逐折指标与均值/标准差 |
| [10](day10_seed_stability/README.md) | 多随机种子 | 稳定性表 |
| [11](day11_pipeline_leakage/README.md) | Pipeline 与预处理泄漏 | 无泄漏流程 |
| [12](day12_tuning_without_test/README.md) | 训练内调参 | 候选参数表 |
| [13](day13_fair_comparison/README.md) | 传统模型公平比较 | 逐折与汇总 CSV |
| [14](day14_ml_stage_report/README.md) | 阶段报告 | 可追溯报告 |

这一阶段的目标不是追求最高分，而是回答“这个结果是否可信”。

## 第三阶段：神经网络（Day 15–21）

| Day | 主题 | 核心产物 |
|---:|---|---|
| [15](day15_tensor_shape/README.md) | 数组、张量与 shape | shape 追踪表 |
| [16](day16_mlp_forward/README.md) | MLP 与前向传播 | 前向计算 |
| [17](day17_loss_optimizer/README.md) | 损失、梯度和学习率 | loss 曲线 |
| [18](day18_batch_epoch_loop/README.md) | Batch、Epoch、训练循环 | 训练历史 |
| [19](day19_mlp_esol/README.md) | ESOL 上的 MLP | MLP 基线 |
| [20](day20_regularization_early_stopping/README.md) | 正则化与早停 | 过拟合诊断 |
| [21](day21_mlp_vs_ml/README.md) | MLP 与传统模型公平比较 | 同协议结果表 |

复杂模型可以不获胜。合格学习结果是能解释差异、波动和计算成本。

## 第四阶段：混合模型与课程总结（Day 22–28）

| Day | 主题 | 核心产物 |
|---:|---|---|
| [22](day22_hybrid_risk/README.md) | 混合模型与泄漏风险 | 方案图 |
| [23](day23_oof_stacking/README.md) | OOF Stacking | 无泄漏二层模型 |
| [24](day24_hybrid_ablation/README.md) | 对照与消融 | 消融表 |
| [25](day25_uncertainty/README.md) | 预测不确定性 | 均值与分歧 |
| [26](day26_active_learning/README.md) | 主动学习池模拟 | 候选排序与 Random 对照 |
| [27](day27_paper_to_schema/README.md) | 论文到数据字典 | 可审计字段映射 |
| [28](day28_capstone_handoff/README.md) | 模型卡与课程总报告 | 模型卡、结果表、下一步计划 |

Day 28 完成后，可以进入 [主动学习专题](../active_learning/README.md)，或在满足前置条件后进入 [可选 GNN 路线](../optional_gnn/README.md)。

## 统一比较规则

- 同一轮比较使用相同数据、标签、划分和指标；
- 标准化和缺失值处理只能在训练数据上 `fit()`；
- 模型选择只看训练区或内部验证；
- 保留 Dummy 与简单模型；
- 报告逐折/逐种子结果，不只报告最佳单次分数；
- 记录版本、随机种子、特征和参数；
- 清楚区分课程示例输出与学习者自己的运行结果。

完整规则见 [统一评价协议](../shared/experiment_protocol.md)。

## 运行与记录

教学 Notebook 可以直接打开并从空内核运行。若希望保留原始输出不变，可以先在本机复制一份；个人文件不需要提交课程仓库。

本地记录建议放在：

```text
learning_outputs/dayXX_topic/
├── notes.md
├── result.csv
└── figure.png
```

记录至少包含输入、步骤、输出、解释、限制和仍未解决的问题。
