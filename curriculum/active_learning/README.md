# 主动学习专题：Unit 1–9

Day 25–26 只介绍不确定性和单轮候选选择。本专题继续学习完整的池式主动学习闭环：代理模型、采集函数、多轮循环、公平基准、批量约束和可审计报告。

## 开始条件

必需：

- 完成 Day 1–14，能够进行可信的回归模型比较；
- 完成 Day 25–26，理解候选池、标签揭示线和 Random 对照；
- 能区分 `X_labeled`、`y_labeled`、`X_pool` 和固定测试集。

建议完成 Day 15–24，以便理解神经代理模型和 OOF 风险。Unit 1–6 不要求 GNN。

## Unit 路线

| Unit | 主题 | 核心产出 |
|---:|---|---|
| [01](unit01_foundations/README.md) | 主动学习问题与完整闭环 | 五模块流程图、无泄漏单轮 query |
| [02](unit02_surrogates_uncertainty/README.md) | 代理模型与不确定性 | GP 与 RF 集成对照 |
| [03](unit03_acquisition_functions/README.md) | 采集函数 | Greedy、Uncertainty、UCB、PI、EI、TS |
| [04](unit04_multiround_loop/README.md) | 多轮循环 | query log 与 best-so-far 曲线 |
| [05](unit05_benchmark_protocol/README.md) | 公平基准 | regret 与固定测试 RMSE |
| [06](unit06_batch_diversity_constraints/README.md) | 批量、多样性与约束 | 可行性过滤和去冗余批次 |
| [07](unit07_neural_graph_surrogates/README.md) | 神经网络与图代理 | 代理模型角色和同预算对照 |
| [08](unit08_physics_closed_loop/README.md) | 物理先验与闭环接口 | 先验对照和批准状态机 |
| [09](unit09_capstone/README.md) | 综合演练 | 配置、逐轮日志、学习曲线和报告 |

完成状态只记录在 [唯一学习清单](../PROGRESS.md)。[Unit 路线索引](PROGRESS.md)只解释阶段关系；论文和课程的对应关系见 [论文地图](PAPER_MAP.md)。

## 固定学习顺序

```text
README 任务卡
→ 01_concepts.md
→ 02_algorithm_walkthrough.md
→ tutorial.ipynb
→ 03_exercises.md
→ 04_reference_answers.md
```

课程 Notebook 中的人工候选池用于学习算法，不代表真实实验。个人笔记和输出可保存在本地 `learning_outputs/active_learning/`。

## 环境

```bash
conda activate esol
python -m pip install -r requirements-active-learning.txt
python scripts/learn.py doctor --track active-learning
```

主路线使用 scikit-learn 的 GP、随机森林和 MLP，不要求 BoTorch、GPyTorch 或真实仪器接口。

## 课程维护检查

```bash
python scripts/run_active_learning_notebooks.py --first-unit 1 --last-unit 9
python scripts/check_repository.py
```

执行器默认只在内存中运行，不改写课程源 Notebook。

## 必须保持的边界

- 代理模型不是采集函数；
- Random 是采集策略基线，不是预测模型；
- query 固定前不能读取候选标签；
- 固定测试集不能被 query，也不能参与调参；
- 不同策略必须使用相同初始集、预算、重复和评价点；
- 离线 Oracle 只是隐藏标签查表，不是真实实验；
- 单轮或单种子胜负不能证明策略稳定更优；
- 最终报告必须同时说明优化目标、全局模型目标和证据等级。
