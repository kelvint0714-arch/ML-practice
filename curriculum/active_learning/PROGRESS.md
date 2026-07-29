# 主动学习 Unit 路线索引

这里解释 Unit 的阶段关系和验收重点，不保存完成状态。实际学习时只在
[全仓库唯一进度表](../PROGRESS.md)勾选，避免两份清单不同步。

## 使用规则

1. 先满足 [开始条件](README.md#开始条件)；
2. 按任务卡规定顺序阅读、手算、运行、练习；
3. 使用 `python scripts/start_unit.py 单元编号` 建立个人副本；
4. 能解释输入、动作、输出和标签权限后，回到唯一进度表勾选；
5. 教材已存在不等于本人已完成。

## 第一阶段：闭环基础

1. **Unit 01｜主动学习到底是什么：**完成 [任务卡](unit01_foundations/README.md)，画出表示、代理模型、不确定性、采集函数和 Oracle 的关系。
2. **Unit 02｜代理模型与不确定性：**完成 [任务卡](unit02_surrogates_uncertainty/README.md)，比较 GP 与 RF 集成，并说明分歧不等于 95% 区间。
3. **Unit 03｜采集函数：**完成 [任务卡](unit03_acquisition_functions/README.md)，独立计算 UCB、PI 和 EI，处理最大化/最小化方向。

## 第二阶段：从单轮到可信实验

4. **Unit 04｜多轮循环：**完成 [任务卡](unit04_multiround_loop/README.md)，保存每轮 query、标签返回和 best-so-far。
5. **Unit 05｜公平基准：**完成 [任务卡](unit05_benchmark_protocol/README.md)，分别完成材料发现和全局模型学习基准。
6. **Unit 06｜批量、多样性与约束：**完成 [任务卡](unit06_batch_diversity_constraints/README.md)，先过滤不可行候选，再形成不冗余批次。

## 第三阶段：研究模型与真实闭环

7. **Unit 07｜神经网络、GNN、PBNN 与 DKL：**完成 [任务卡](unit07_neural_graph_surrogates/README.md)，指出每种算法位于表示、代理、不确定性还是采集模块。
8. **Unit 08｜物理先验与实验闭环：**完成 [任务卡](unit08_physics_closed_loop/README.md)，比较无先验、合理先验与错误先验，并实现人工批准状态。
9. **Unit 09｜综合演练：**完成 [任务卡](unit09_capstone/README.md)，先形成可审计的人工候选池研究包，再进入公开论文复现。

## 完成 Unit 09 后应能回答

- 主动学习和普通监督学习、贝叶斯优化分别有什么关系？
- RF、GP、MLP、GNN、PBNN、DKL、PINN 会出现在闭环哪一步？
- 采集函数为什么不能读取候选真实性能？
- 怎样公平比较两个策略？
- 为什么要报告学习曲线、regret、多随机种子和随机基线？
- 离线候选池怎样替换成真实化学实验 Oracle？
- 当前证据能说明什么，不能说明什么？
