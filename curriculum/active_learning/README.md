# 主动学习专题：Unit 1–9

这是一条独立的主动学习专题路线。现有 Day 25–26 只负责“不确定性与单轮候选选择”的入门预览；本专题把它扩展成可重复、多轮、有基线、有论文依据、能连接真实材料实验的完整方法。

本课程不是论文复现仓库。这里用小型、可解释、可运行的教学实验学习算法；论文原始代码、环境锁定和论文数值核对继续放在 [Reproduction of Active Learning in Materials Science](https://github.com/kelvint0714-arch/Reproduction-of-Active-Learning-in-Materials-Science)。

## 开始条件

必需：

- 完成 Day 1–14，能够训练传统回归模型并进行可信比较；
- 完成 Day 25–26，理解模型分歧、候选池、标签揭示线和随机对照；
- 能解释 `X_labeled`、`y_labeled`、`X_pool` 的区别。

建议：

- 完成 Day 15–24，以便理解 Unit 7 的神经网络部分；
- GNN 扩展需完成 Day 29–35，Unit 1–6 不要求 GNN。

## Unit 路线

| Unit | 主题 | 核心产出 |
|---:|---|---|
| [01](unit01_foundations/README.md) | 主动学习问题与完整闭环 | 五模块流程图、无泄漏单轮 query |
| [02](unit02_surrogates_uncertainty/README.md) | 代理模型与不确定性 | GP 与 RF 集成均值/分歧对照 |
| [03](unit03_acquisition_functions/README.md) | 采集函数 | Greedy、Uncertainty、UCB、PI、EI、TS 排序表 |
| [04](unit04_multiround_loop/README.md) | 多轮主动学习循环 | 可重复 query log 与 best-so-far 曲线 |
| [05](unit05_benchmark_protocol/README.md) | 公平基准协议 | 材料发现 regret 与全局模型学习测试 RMSE 两类曲线 |
| [06](unit06_batch_diversity_constraints/README.md) | 批量、多样性与约束 | 可行性过滤和去冗余批量推荐 |
| [07](unit07_neural_graph_surrogates/README.md) | 神经网络、GNN、PBNN 与 DKL | 模型角色图和同预算代理模型对照 |
| [08](unit08_physics_closed_loop/README.md) | 物理先验与真实闭环 | 结构化 GP、成本与人工批准状态机 |
| [09](unit09_capstone/README.md) | 人工综合演练与复现交接 | 配置、逐轮日志、学习曲线、报告与公开论文复现入口 |

全仓库唯一需要勾选的清单是
[一步一步学习目录](../PROGRESS.md)。本目录的
[Unit 路线索引](PROGRESS.md)只解释专题阶段和验收重点，不重复保存完成状态。
论文与课程的对应关系见 [论文地图](PAPER_MAP.md)。
Unit 4–9 遇到长函数、`groupby`、回调或 `.loc` 时，先查
[主动学习 Python 语法速查](shared/python_patterns.md)。

## 每个 Unit 的固定顺序

```text
README 任务卡
→ 01_concepts.md 中文概念
→ 02_algorithm_walkthrough.md 手算、伪代码和语法走读
→ tutorial.ipynb 可运行教学实验
→ 03_exercises.md 独立练习
→ 04_reference_answers.md 最后核对
```

课程 Notebook 的预存输出只证明教材可以运行，不代表学习者已经完成。开始个人实验时运行：

```bash
python scripts/start_unit.py 1
```

个人副本会建立在 `experiments/active_learning/`，不会覆盖课程源文件。

## 环境

Unit 1–9 主路线使用现有 `esol` 环境：

```bash
conda activate esol
python -m pip install -r requirements-active-learning.txt
```

主路线先使用 scikit-learn 的 GP、随机森林和 MLP，不在入门阶段引入 BoTorch、GPyTorch、JAX 或真实仪器依赖。

## 课程维护命令

学习者只运行 `start_unit.py`，不要运行构建器。课程维护者修改
`scripts/build_active_learning_notebooks.py` 后，必须按顺序执行：

```bash
python scripts/build_active_learning_notebooks.py
python scripts/run_active_learning_notebooks.py --in-place
python scripts/build_active_learning_notebooks.py --check
python scripts/check_repository.py
```

第一条会覆盖 9 份课程源 Notebook，第二条重新生成可核对的预存输出；
第三条确认课程源单元格仍与构建器一致。遗漏第二条会让仓库检查因执行计数
为空而失败。

## 证据边界

- 教学数据是人工候选池，不是粘合剂实验数据；
- 离线 `oracle` 只是从隐藏标签中查值，不是真实实验；
- 单轮或单种子胜负不能证明策略更优；
- GNN、随机森林、PBNN 和 DKL 是表示/代理或不确定性模块，不是主动学习循环本身；
- PINN 只有接入候选选择与反馈循环后，才是主动学习系统的一部分；
- 真实候选必须经过化学可行性、安全、成本和权限审核。
