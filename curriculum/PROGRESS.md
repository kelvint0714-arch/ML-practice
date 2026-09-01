# 一步一步学习目录

这是本仓库**唯一需要从上往下执行的学习清单**。以后不需要在多个文件之间猜“接下来学什么”。

## 使用方法

1. 在本页找到第一个 `[ ]`；
2. 点击这一项的链接；
3. 按链接页面里的顺序完成阅读、代码、产物、完成标准和自测；
4. 确认自己能够解释，而不只是把代码运行出来，再把 `[ ]` 改成 `[x]`；
5. 回到本页，继续下一个未勾选项。

`Day` 和 `Unit` 都表示学习模块，不要求在一个自然日内完成。一个模块学两三天完全正常，也不要为了赶日期跳过前置内容。仓库里已经存在的 Notebook 和结果是学习材料，不代表你本人已经完成，所以本页不会自动勾选。

Day 2–35 的每个任务卡都链接到中文概念、算法推演、练习、参考答案和教学 Notebook。开始前先阅读[完整学习包使用方法](shared/day_package_guide.md)。自己的笔记和输出可保存在本地 `learning_outputs/`，不需要复制整套课程文件。

> **你现在只做一件事：**从下面的 **1.1** 开始。
>
> **第一个建议里程碑：**学到 Day 14，能够完成一套可信的传统机器学习比较；质量优先，不需要提前赶 MLP 或 GNN。

## 第一阶段：先学算法，代码只做伴随练习（Day 1–7）

### Day 1：先建立算法骨架

Day 1 特意拆成 6 个小步骤。先阅读 [Day 1 总说明](core/day01_beginner/README.md)，再按 1.1 到 1.6 的顺序完成。完整工程 Notebook 到 Day 7 才从头运行。

- [√] **1.1 机器学习算法是什么：**阅读 [算法概念](core/day01_beginner/01_concepts.md)，先不用代码，分清算法、模型、样本、`X`、`y`、训练和预测。
- [√] **1.2 亲手完成一次预测：**完成 [纸笔手算](core/day01_beginner/02_hand_calculation.md)，自己计算均值基线、MAE 和 RMSE。
- [√] **1.3 用中文说出流程：**完成 [中文伪代码](core/day01_beginner/03_pseudocode.md)，能够脱离 Python 复述“数据 → 训练 → 预测 → 评价”。
- [√] **1.4 再看最小代码：**逐块运行 [最小 Python 代码](core/day01_beginner/04_minimal_code.md)，每一行都能放回刚刚学过的算法步骤。
- [√] **1.5 只在真实代码里找主线：**使用 [ESOL Notebook 对照地图](core/day01_beginner/05_esol_notebook_map.md)，只定位数据、模型、`fit`、`predict` 和指标，不逐行硬啃工程代码。
- [√] **1.6 Day 1 验收：**完成 [核心验收](core/day01_beginner/exercises.md)。能够口头解释、手算并运行最小代码后，才勾选本项。

### Day 2–7：按天继续

- [ ] **Day 2｜MAE、RMSE、R²：**完成 [Day 2 任务卡](core/day02_metrics/README.md)，建立一个小例子的误差表并解释三个指标。
- [ ] **Day 3｜线性回归与 Ridge：**完成 [Day 3 任务卡](core/day03_ridge/README.md)，理解权重、L2 正则化和 `alpha`。
- [ ] **Day 4｜决策树与过拟合：**完成 [Day 4 任务卡](core/day04_decision_tree/README.md)，比较不同树深度。
- [ ] **Day 5｜随机森林与 Bagging：**完成 [Day 5 任务卡](core/day05_random_forest/README.md)，比较单棵树与随机森林。
- [ ] **Day 6｜Gradient Boosting：**完成 [Day 6 任务卡](core/day06_gradient_boosting/README.md)，理解模型怎样顺序修正错误。
- [ ] **Day 7｜完整 ESOL 基线：**完成 [Day 7 任务卡](core/day07_integrated_baseline/README.md)，这时才从空内核运行完整 Notebook，并把长代码映射回算法主线。

## 第二阶段：让实验结果可信（Day 8–14）

- [ ] **Day 8｜数据划分与测试集边界：**完成 [Day 8 任务卡](core/day08_split_protocol/README.md)，写出 split 检查表和测试集使用规则。
- [ ] **Day 9｜K 折交叉验证：**完成 [Day 9 任务卡](core/day09_cross_validation/README.md)，保存逐折结果并报告均值与标准差。
- [ ] **Day 10｜随机种子稳定性：**完成 [Day 10 任务卡](core/day10_seed_stability/README.md)，报告多次结果、均值和标准差。
- [ ] **Day 11｜Pipeline 与数据泄漏：**完成 [Day 11 任务卡](core/day11_pipeline_leakage/README.md)，建立无泄漏预处理流程。
- [ ] **Day 12｜不偷看测试集的调参：**完成 [Day 12 任务卡](core/day12_tuning_without_test/README.md)，只在训练集内部 K 折选择 `alpha`，冻结后只检查一次外部 validation。
- [ ] **Day 13｜传统模型公平比较：**完成 [Day 13 任务卡](core/day13_fair_comparison/README.md)，保存逐折 `fold_metrics.csv` 和汇总 `model_summary.csv`。
- [ ] **Day 14｜传统机器学习阶段报告：**完成 [Day 14 任务卡](core/day14_ml_stage_report/README.md)，从 Day 13 真实产物形成第一份可以向评审者解释的阶段报告。

## 第三阶段：神经网络入门（Day 15–21）

- [ ] **Day 15｜数据与 shape：**完成 [Day 15 任务卡](core/day15_tensor_shape/README.md)，记录每一步数组形状。
- [ ] **Day 16｜MLP 层、激活与前向传播：**完成 [Day 16 任务卡](core/day16_mlp_forward/README.md)，建立第一个小型 MLP。
- [ ] **Day 17｜损失函数与优化：**完成 [Day 17 任务卡](core/day17_loss_optimizer/README.md)，观察并解释 loss 的变化。
- [ ] **Day 18｜Batch、Epoch 与训练循环：**完成 [Day 18 任务卡](core/day18_batch_epoch_loop/README.md)，形成逐 epoch 训练记录。
- [ ] **Day 19｜ESOL 上的 MLP：**完成 [Day 19 任务卡](core/day19_mlp_esol/README.md)，得到 MLP 基线。
- [ ] **Day 20｜正则化与早停：**完成 [Day 20 任务卡](core/day20_regularization_early_stopping/README.md)，比较训练曲线和早停结果。
- [ ] **Day 21｜MLP 与传统模型公平比较：**完成 [Day 21 任务卡](core/day21_mlp_vs_ml/README.md)，在相同输入和验证协议下比较模型。

## 第四阶段：混合模型、主动学习与课程总结（Day 22–28）

- [ ] **Day 22｜混合模型与泄漏风险：**完成 [Day 22 任务卡](core/day22_hybrid_risk/README.md)，画出三种传统 ML 与神经网络结合方案。
- [ ] **Day 23｜OOF Stacking：**完成 [Day 23 任务卡](core/day23_oof_stacking/README.md)，实现无泄漏混合模型。
- [ ] **Day 24｜混合模型消融：**完成 [Day 24 任务卡](core/day24_hybrid_ablation/README.md)，公平比较单模型和混合模型。
- [ ] **Day 25｜预测不确定性：**完成 [Day 25 任务卡](core/day25_uncertainty/README.md)，得到预测均值与标准差。
- [ ] **Day 26｜主动学习模拟：**完成 [Day 26 任务卡](core/day26_active_learning/README.md)，跑通单轮无标签泄漏的候选排序与随机对照。
- [ ] **Day 27｜从论文到数据字典：**完成 [Day 27 任务卡](core/day27_paper_to_schema/README.md)，把论文中的任务、输入、目标和评价协议映射成可审计字段。
- [ ] **Day 28｜模型卡与课程总报告：**完成 [Day 28 任务卡](core/day28_capstone_handoff/README.md)，形成模型卡、结果表和下一步学习计划。

完成 Day 28 代表核心路线结束：你已经具备定义任务、建立基线、执行可信评价并沟通模型限制的基础。

## 主动学习专题：按 Unit 系统深化（Unit 1–9）

专题入口与开始条件见 [主动学习路线](active_learning/README.md)。Day 26 只完成单轮入门；本专题继续学习完整采集函数、多轮基准、批量约束、神经/GNN 代理、物理先验和闭环接口。Unit 1–6 不要求 GNN，Unit 7 的 GNN 扩展需先完成 Day 29–35。

- [ ] **Unit 01｜主动学习闭环与问题定义：**完成 [Unit 01 任务卡](active_learning/unit01_foundations/README.md)。
- [ ] **Unit 02｜代理模型与不确定性：**完成 [Unit 02 任务卡](active_learning/unit02_surrogates_uncertainty/README.md)。
- [ ] **Unit 03｜采集函数：**完成 [Unit 03 任务卡](active_learning/unit03_acquisition_functions/README.md)。
- [ ] **Unit 04｜多轮主动学习循环：**完成 [Unit 04 任务卡](active_learning/unit04_multiround_loop/README.md)。
- [ ] **Unit 05｜公平基准协议：**完成 [Unit 05 任务卡](active_learning/unit05_benchmark_protocol/README.md)。
- [ ] **Unit 06｜批量、多样性与约束：**完成 [Unit 06 任务卡](active_learning/unit06_batch_diversity_constraints/README.md)。
- [ ] **Unit 07｜神经网络与图代理模型：**完成 [Unit 07 任务卡](active_learning/unit07_neural_graph_surrogates/README.md)。
- [ ] **Unit 08｜物理先验与真实闭环：**完成 [Unit 08 任务卡](active_learning/unit08_physics_closed_loop/README.md)。
- [ ] **Unit 09｜人工综合演练与论文复现交接：**完成 [Unit 09 任务卡](active_learning/unit09_capstone/README.md)。

专题阶段说明和论文对应关系分别见 [Unit 路线索引](active_learning/PROGRESS.md)
与 [论文地图](active_learning/PAPER_MAP.md)；完成状态仍只在本页记录。

## 可选阶段：GNN（Day 29–35）

先阅读 [GNN 路线与启动条件](optional_gnn/README.md)。核心路线完成后，Day 29–34 在公开图数据上学习；Day 35 检查一个数据集是否具备使用 GNN 的输入、标签、划分和基线条件。若当前学习优先级更高，停在 Day 28 继续巩固也完全正确。

- [ ] **Day 29｜图、节点、边和邻接关系：**完成 [Day 29 任务卡](optional_gnn/day29_graph_basics/README.md)。
- [ ] **Day 30｜PyTorch Geometric `Data`：**完成 [Day 30 任务卡](optional_gnn/day30_pyg_data/README.md)。
- [ ] **Day 31｜GCN：**完成 [Day 31 任务卡](optional_gnn/day31_gcn/README.md)。
- [ ] **Day 32｜GraphSAGE：**完成 [Day 32 任务卡](optional_gnn/day32_graphsage/README.md)。
- [ ] **Day 33｜GAT：**完成 [Day 33 任务卡](optional_gnn/day33_gat/README.md)。
- [ ] **Day 34｜GIN 与图分类：**完成 [Day 34 任务卡](optional_gnn/day34_gin_graph_classification/README.md)。
- [ ] **Day 35｜GNN 数据就绪评审：**完成 [Day 35 任务卡](optional_gnn/day35_molecular_graph_gate/README.md)。

## 每个学习单元的完成规则

- `[ ]`：还没做，或代码能运行但你无法解释；
- `[x]`：必做任务、产物、完成清单和自测全部通过；
- 卡住时把完整报错和不理解的代码记下来，再向 Codex 提问；
- 不要复制参考答案冒充自己的理解；
- 公开教学数据的结果不能直接外推到新的真实任务。

## 学习记录模板

完成一个 Day 后，把下面内容记录到本地 `learning_outputs/` 或自己的笔记工具中。

```markdown
# Day N 学习记录

## 今天完成了什么

-

## 输入、动作、输出

- 输入：
- 模型做的动作：
- 输出：

## 我今天新认识的 Python 语法

-

## 结果怎样解释

-

## 当前结果不能说明什么

-

## 遇到的报错及解决方式

- 完整报错：
- 原因：
- 怎样修复：

## 我还不能独立回答的问题

-
```
