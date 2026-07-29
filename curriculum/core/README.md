# 算法线 28 天零基础学习路线

这份文件是课程地图；真正执行和勾选只使用 [一步一步学习目录](../PROGRESS.md)。路线解决的不是“28 天学完所有机器学习”，而是让刚入门 Python 的学习者完成一条可以验收的研究流程：

```text
先理解算法问题与直觉
→ 用纸笔和最小代码验证理解
→ 再映射到完整研究代码
→ 掌握传统模型与可信验证
→ 学习 MLP 神经网络
→ 公平比较两类模型
→ 实现无泄漏的混合模型
→ 用不确定性模拟下一轮实验选择
→ 为真实粘合剂数据建立接口
```

ESOL 只用于练习算法流程。它的预测目标是水溶解度 `logS`，不是粘合剂强度。真实粘合剂数据到达前，不得把练习结果写成合作项目结论。

## 怎样使用这套路线

1. 只按 [一步一步学习目录](../PROGRESS.md) 找下一个任务，不在本页跳着学习。
2. 每个单元固定按“中文概念 → 纸笔或图示 → 最小代码 → 结果解释 → 自测”执行。
3. 先说清算法在解决什么问题，再看代码；不要求先学完整 Python 才能开始算法。
4. 每一段新代码都回答“输入、动作、输出”三个问题。
5. 完整工程 Notebook 只在任务明确要求时打开；长逐行讲解只作为按需参考。
6. 当天的必做练习和完成清单没有通过，就不要勾选“完成”。
7. 测试集不是每日排行榜。模型选择只使用训练集和验证集。
8. 遇到代码报错时保存完整报错，不要只截最后一行。
9. Day 29–35 是通过门槛后才开始的 GNN 扩展，不是当前赶工任务。

## 第一阶段：算法优先的传统机器学习入门（Day 1–7）

| 天数 | 主题 | 当天必须产出 |
|---:|---|---|
| [Day 1](day01_beginner/README.md) | 回归、X/y、均值基线与算法骨架 | 手算、伪代码和最小运行 |
| [Day 2](day02_metrics/README.md) | MAE、RMSE、R² | 误差表与三句解释 |
| [Day 3](day03_ridge/README.md) | 线性回归、Ridge 与 L2 | `alpha` 敏感性表 |
| [Day 4](day04_decision_tree/README.md) | 决策树与过拟合 | 不同树深度结果表 |
| [Day 5](day05_random_forest/README.md) | 随机森林与 Bagging | 单树/森林对照 |
| [Day 6](day06_gradient_boosting/README.md) | Gradient Boosting | Bagging/Boosting 对照与固定基线 |
| [Day 7](day07_integrated_baseline/README.md) | 映射并重跑完整 ESOL 基线 | 代码分类、重跑记录和结果解释 |

前六天先使用纸笔和人工小数据理解算法，Day 7 才回到完整 ESOL Notebook。长篇 Python 与 Notebook 逐行教程保留为查询手册，不再作为进入算法前的硬门槛。梯度提升使用 scikit-learn 自带模型；XGBoost 只作为后续可选项。

## 第二阶段：让实验可信（Day 8–14）

| 天数 | 主题 | 当天必须产出 |
|---:|---|---|
| [Day 8](day08_split_protocol/README.md) | 数据划分协议与测试集边界 | split 检查表与使用规则 |
| [Day 9](day09_cross_validation/README.md) | K 折交叉验证 | 逐折指标、均值和标准差 |
| [Day 10](day10_seed_stability/README.md) | 多随机种子稳定性 | 均值、标准差、逐次结果 |
| [Day 11](day11_pipeline_leakage/README.md) | Pipeline 与预处理泄漏 | 错误/正确流程图和一条无泄漏 Pipeline |
| [Day 12](day12_tuning_without_test/README.md) | 不偷看测试集的调参 | 训练内 CV 候选表和一次外部验证 |
| [Day 13](day13_fair_comparison/README.md) | 传统模型公平比较 | 逐折文件和汇总文件 |
| [Day 14](day14_ml_stage_report/README.md) | 传统 ML 阶段报告 | 基于 Day 13 真实产物的阶段报告 |

这一阶段不是追求最高分，而是回答“这个结果能不能相信”。完成后，你应当能发现常见的数据泄漏、单次分数误导和不公平比较。

## 第三阶段：神经网络入门（Day 15–21）

| 天数 | 主题 | 当天必须产出 |
|---:|---|---|
| [Day 15](day15_tensor_shape/README.md) | 神经网络的数据与 shape | 前向计算 shape 记录 |
| [Day 16](day16_mlp_forward/README.md) | MLP 层、激活与前向传播 | 第一个小型 MLP |
| [Day 17](day17_loss_optimizer/README.md) | 损失函数与优化 | loss 变化记录 |
| [Day 18](day18_batch_epoch_loop/README.md) | Batch、Epoch 与训练循环 | 逐 epoch 训练表 |
| [Day 19](day19_mlp_esol/README.md) | ESOL 上的 MLP | MLP 基线结果 |
| [Day 20](day20_regularization_early_stopping/README.md) | 正则化与早停 | 训练曲线和早停对照 |
| [Day 21](day21_mlp_vs_ml/README.md) | MLP 与传统模型公平比较 | 同协议结果表 |

核心路线先使用 NumPy 与 scikit-learn `MLPRegressor`，避免刚入门就被深度学习框架的工程细节淹没。PyTorch 和 PyTorch Geometric 放在可选 GNN 阶段再安装。

## 第四阶段：结合算法并连接项目（Day 22–28）

| 天数 | 主题 | 当天必须产出 |
|---:|---|---|
| [Day 22](day22_hybrid_risk/README.md) | 混合模型思路与泄漏风险 | 三种结合方案图 |
| [Day 23](day23_oof_stacking/README.md) | OOF Stacking | 无泄漏混合模型结果 |
| [Day 24](day24_hybrid_ablation/README.md) | 对照与消融 | 单模型/混合模型消融表 |
| [Day 25](day25_uncertainty/README.md) | 集成不确定性 | 预测均值与标准差 |
| [Day 26](day26_active_learning/README.md) | 主动学习池模拟 | 单轮候选排序、标签揭示边界和随机对照 |
| [Day 27](day27_paper_to_schema/README.md) | 论文到字段映射 | 三篇论文输入/输出表 |
| [Day 28](day28_capstone_handoff/README.md) | 粘合剂接入与总报告 | 数据接入清单和总报告 |

Day 22–24 直接回应导师提出的“传统机器学习与神经网络结合”；Day 25–26 对应小样本实验推荐；Day 27–28 把方法练习转换成化学组能够讨论的数据需求。

## 主动学习专题（Unit 1–9）

完成 Day 26 并掌握 Day 1–14 的可信验证基础后，可以进入
[主动学习 Unit 1–9](../active_learning/README.md)。这条专题路线把 Day 26
的单轮演示扩展为代理模型、不确定性、采集函数、多轮循环、公平基准、
批量约束、神经/图代理、物理先验和综合项目。

Unit 1–6 不要求 GNN；Unit 7 的 GNN 扩展仍需先完成 Day 29–35。

## 可选 GNN 路线（Day 29–35）

Day 29–35 的任务卡也预先写好，但只有通过 [GNN 启动条件](../optional_gnn/README.md) 才开始。

| 天数 | 主题 | 主要目的 |
|---:|---|---|
| [Day 29](../optional_gnn/day29_graph_basics/README.md) | 图、节点、边和邻接关系 | 看懂图数据与表格数据的区别 |
| [Day 30](../optional_gnn/day30_pyg_data/README.md) | PyTorch Geometric `Data` | 认识 `x`、`edge_index`、`y` |
| [Day 31](../optional_gnn/day31_gcn/README.md) | GCN | 完成第一个公开图数据基线 |
| [Day 32](../optional_gnn/day32_graphsage/README.md) | GraphSAGE | 理解邻居采样和聚合 |
| [Day 33](../optional_gnn/day33_gat/README.md) | GAT | 理解注意力加权邻居 |
| [Day 34](../optional_gnn/day34_gin_graph_classification/README.md) | GIN 与图分类 | 从节点任务过渡到分子级预测 |
| [Day 35](../optional_gnn/day35_molecular_graph_gate/README.md) | 分子图与粘合剂启用评审 | 决定项目是否真的需要 GNN |

## 每一天都要回答的五个问题

无论当天模型是什么，都要在学习记录中回答：

1. 一行数据代表什么？
2. `X` 中有哪些输入，`y` 是什么目标？
3. 模型在哪些数据上调用了 `fit()`？
4. 哪些数据用于比较模型，测试集有没有被反复查看？
5. 当前结果能支持什么结论，不能支持什么结论？

## 每天建议的时间安排

| 环节 | 建议时间 | 要做什么 |
|---|---:|---|
| 预习 | 20–30 分钟 | 看当天目标和新术语，不运行代码 |
| 跟做 | 60–90 分钟 | 按顺序输入代码，每段记录输入、动作、输出 |
| 小实验 | 30–60 分钟 | 只修改当天指定的 1–2 个参数 |
| 验收 | 20–30 分钟 | 完成自测问题、检查输出文件 |
| 复盘 | 10–20 分钟 | 用自己的话写三句话，不复制 README |

如果一天无法完成，可以拆成两天。`Day` 是学习单元，不是必须在自然日内赶完的期限。

## 每日文件约定

Day 2–28 的教材已经提前补齐，但教材和个人实验严格分开：

```text
curriculum/core/dayXX_topic/
├── README.md
├── 01_concepts.md
├── 02_algorithm_walkthrough.md
├── 03_exercises.md
├── 04_reference_answers.md
└── tutorial.ipynb
```

真正开始某一天时运行：

```bash
python scripts/start_day.py 2
```

脚本才会建立：

```text
experiments/day02_metrics/
├── README.md
├── day02_metrics.ipynb
├── notes.md
└── results/
    └── README.md
```

这样既可以提前获得完整学习材料，也不会把教材的预存输出冒充成本人已经完成的研究证据。详细规则见[完整 Day 学习包使用方法](../shared/day_package_guide.md)。

## 模型比较的统一规则

- 主回归指标：RMSE；
- 辅助指标：MAE、R²；
- 同一天比较的模型必须使用相同输入、标签和划分；
- 标准化器和缺失值处理器只能在训练数据上 `fit()`；
- 所有随机种子、参数和软件版本都要保存；
- 调参使用训练/验证数据，最终测试集只能在方案冻结后使用一次；
- 复杂模型必须同时与 Dummy 和简单模型比较；
- 单次分数不能写成稳定结论，要报告重复实验的均值与标准差；
- 公开数据上的结果不能改名为“粘合剂预测结果”。

## 真实粘合剂数据到达前后的边界

### 现在可以做

- 在 ESOL 上练习完整算法流程；
- 比较传统模型、MLP 和混合模型；
- 模拟不确定性与主动学习；
- 编写字段检查和读取接口；
- 阅读粘合剂 AI 论文并提取方法。

### 现在不可以做

- 用空白模板训练模型；
- 自己填造化学性质或性能数值；
- 把不同论文、不同测试标准的数据直接拼接；
- 向化学组推荐具体配方并声称已经验证；
- 在没有结构表示时直接启动粘合剂 GNN。

## 核心路线完成后应该获得的能力

完成 Day 28 后，你不需要背出所有数学公式，但应当能够：

- 看懂并修改一个标准回归 Notebook；
- 解释 Ridge、随机森林、Boosting 和 MLP 的基本差别；
- 发现明显的数据泄漏和过拟合；
- 用统一协议比较传统模型、神经网络和混合模型；
- 说明为什么小样本下复杂模型不一定更好；
- 把一份真实粘合剂表格转成待建模字段清单；
- 向导师准确汇报“已经完成什么、证据是什么、下一步缺什么”。
