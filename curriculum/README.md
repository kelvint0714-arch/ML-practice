# B. 算法预研线

## 目标

在等待真实粘合剂数据期间，建立一套可复现、可比较、能迁移到小样本材料表格数据的算法流程，并完成老师要求的“传统机器学习与神经网络结合”方法调研。

当前已有一份可复现的 [ESOL 基线 Notebook](../experiments/esol/day01_baseline/esol_baseline.ipynb)，但它是 Day 7 的综合材料，不再是学习起点。ESOL 只是方法练习数据，不代表粘合剂项目数据。

## 从这里开始

当前课程已拆成 [Day 1–28 核心路线](core/README.md)、[Day 29–35 可选 GNN 路线](optional_gnn/README.md) 和 [主动学习 Unit 1–9 专题路线](active_learning/README.md)。Day 26 是主动学习桥接课，Unit 路线负责多轮基准、采集函数、批量约束、神经/图代理和物理闭环的系统深化。

如果不确定需要完成全部内容还是只学其中一段，先使用[学习路线选择器](LEARNING_PATHS.md)。它会按当前基础和目标给出推荐起点与停止点；选好后仍以本目录的 `PROGRESS.md` 为唯一完成清单。

第一次学习只打开 [一步一步学习目录](PROGRESS.md)，找到第一个 `[ ]` 并按顺序执行。路线采用“算法概念 → 纸笔手算 → 中文伪代码 → 最小代码 → 真实 Notebook”的顺序。Day 2–35 每天都配有分层中文讲义、练习、参考答案和教学 Notebook；Day 1 不要求逐行阅读完整工程代码，Day 7 才把已经学过的算法映射回完整 ESOL 实验。Day 29–34 可以在公开图上学习；Day 35 决定能否把 GNN 用到真实粘合剂项目。

配套入口：

- [一步一步学习目录（唯一任务清单）](PROGRESS.md)
- [Day 1–28 核心课程](core/README.md)
- [共享词典、实验协议与报错排查](shared/README.md)
- [完整 Day 学习包使用方法](shared/day_package_guide.md)
- [Day 29–35 可选 GNN 与启动条件](optional_gnn/README.md)
- [主动学习 Unit 1–9 与开始条件](active_learning/README.md)
- [主动学习论文地图](active_learning/PAPER_MAP.md)

核心路线运行代码时使用根目录的 [`requirements-learning.txt`](../requirements-learning.txt)，不要求安装 XGBoost、PyTorch 或 PyTorch Geometric。主动学习主路线复用 [`requirements-active-learning.txt`](../requirements-active-learning.txt)，先使用 scikit-learn GP/RF/MLP。Day 29–35 使用单独的 [`requirements-gnn.txt`](../requirements-gnn.txt)；安装与运行命令见 [完整 Day 学习包使用方法](shared/day_package_guide.md)。

## 实验路线

| 阶段 | 要做什么 | 主要输出 | 状态 |
|---|---|---|---|
| B0 算法入门与传统基线 | Dummy、Ridge、决策树、随机森林、Boosting | 手算、最小代码和完整 ESOL 基线映射 | Day 1–7 |
| B1 可信验证 | 划分、K 折、多随机种子、Pipeline、无测试集调参 | 逐折结果、均值、标准差、协议和配置 | Day 8–14 |
| B2 神经网络基线 | 小型 MLP，与传统模型使用相同输入和评估口径 | 学习曲线、验证指标、过拟合诊断 | Day 15–21 |
| B3 混合模型 | OOF 原理、Stacking、对照和消融 | 与 B0、B2 的公平对照 | Day 22–24 |
| B4 实验推荐 | 集成不确定性、主动学习池模拟 | 候选选择逻辑及边界 | Day 25–26 |
| B5 材料接入 | 论文字段映射、真实数据接口和阶段报告 | 可供导师与化学组确认的接入清单 | Day 27–28 |
| B6 可选 GNN | 公开图任务、分子图和启用评审 | 只在门槛通过后执行 | Day 29–35 |
| B7 主动学习专题 | 多采集函数、多轮基准、批量约束、神经/物理代理与闭环 | 逐轮日志、学习曲线和公开数据复现入口 | Unit 1–9 |

## 比较规则

- 同一轮比较必须使用相同的数据、划分、输入信息和指标；
- 所有随机种子、版本、特征参数和超参数必须保存；
- 模型选择只看训练集/验证集，最终测试集不能反复查看；
- 如果神经网络输出再交给树模型，必须使用训练折内或 OOF（out-of-fold）特征，避免把验证标签泄漏进特征；
- 小数据上更复杂不等于更好，必须保留 Dummy 和简单模型；
- 只有拿到 SMILES、分子图或可靠结构后，才进入 GNN 路线。

## 文献调研如何分类

正式报告要把“材料领域依据”和“算法结合依据”分开，并准确标注发表状态。

| 文献/页面 | 报告中应标注的状态 | 本项目借鉴点 |
|---|---|---|
| [Pruksawan et al., 2019](https://doi.org/10.1080/14686996.2019.1673670) | 同行评审期刊论文 | 小样本环氧胶黏剂、Gradient Boosting、主动学习 |
| [Active and transfer learning with partially Bayesian neural networks](https://openreview.net/forum?id=rxbXpedQJc) | AI4Mat–ICLR 2025 Workshop | 材料小样本、不确定性、主动/迁移学习 |
| [GBDT-Guided Piecewise-Linear Embeddings](https://openreview.net/forum?id=57Fx3Ck4ot) | OpenReview 页面所示 ICLR 2026 投稿；未核实录用前不得写成已录用 | 树模型引导神经网络表格表示 |

“2026 顶会论文”清单必须逐篇核对主会/Workshop、投稿/录用、代码和数据状态。只在 OpenReview 看见投稿页面，不等于已经发表在顶会。

## 本阶段交付

1. 一张文献表：状态、数据、输入、模型、指标、代码、可迁移点；
2. 一个传统模型与 MLP 使用相同协议的可运行 Notebook；
3. 一个传统 ML＋神经网络混合模型的无泄漏对照实验；
4. 一张模型结果表和一页结论边界；
5. 真实粘合剂数据接入说明。

## 可以和不可以得出的结论

可以说明某个流程在公开数据上是否可复现、是否过拟合，以及混合模型是否在同一验证协议下优于基线。真实数据到达前，不能声称已经预测了合作项目的粘合剂性能，也不能据此向化学组推荐具体配方。
