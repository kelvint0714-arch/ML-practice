# ML-practice

面向化学与材料机器学习的学习与实验仓库。仓库把“课程任务”“真实实验”“数据接口”和“参考资料”分开，避免把已写好的任务卡误认为已经完成的实验。

## 现在从这里开始

1. 打开唯一导航：[一步一步学习目录](curriculum/PROGRESS.md)
2. 永远从目录里的第一个 `[ ]` 开始，完成并通过自测后再勾选
3. 目录会从 Day 1 带到 Day 35，并在核心路线后提供按 Unit 编排的主动学习专题；Day 29–34 可用公开图学习 GNN，但把 GNN 用到粘合剂项目仍要通过 Day 35 门槛

`Day` 和 `Unit` 都是学习单元，不是必须一天完成的期限。路线先学算法概念和手算，再用最小代码验证理解；Day 2–35 和主动学习 Unit 1–9 均提供中文讲义、算法推演、练习、参考答案和可运行教学 Notebook。完整 ESOL 基线 Notebook 到 Day 7 才从头运行；GNN 或主动学习文件已经存在不等于本人已经完成。

## 两条研究工作线

| 工作线 | 当前状态 | 入口 | 现阶段交付 |
|---|---|---|---|
| 粘合剂数据线 | 等待导师与化学组确认体系、字段和真实样例 | [数据线说明](data/adhesive/README.md) | 数据字典、3–5 行格式样例、首批获授权真实数据 |
| 算法预研线 | ESOL 参考基线 E01 已可复现；课程从 Day 1 算法概念开始 | [算法线说明](curriculum/README.md) | 传统模型、MLP、混合模型与主动学习的统一对照流程 |

两条线的边界和衔接条件见 [项目路线图](docs/project_roadmap.md)。公开数据只用于方法开发；真实粘合剂数据到达前，不把 ESOL 结果表述为粘合剂实验结论。没有可靠结构表示前，不启动粘合剂 GNN。

## 仓库结构

```text
ML-practice/
├── .github/workflows/       # GitHub 自动仓库检查
├── curriculum/             # 核心 Day、可选 GNN 与主动学习 Unit 专题
├── data/                   # 公开数据说明与粘合剂数据接口
├── experiments/            # 已运行参考实验＋未开始的完整实验工作区
├── docs/                   # 项目路线、参考资料和旧方案归档
├── scripts/                # 建立个人 Day、只读运行 Notebook、仓库检查
├── requirements.txt        # 已复现实验的基础环境
├── requirements-learning.txt
├── requirements-active-learning.txt
└── requirements-gnn.txt    # Day 29–35 的独立 GNN 环境
```

详细入口：

- [一步一步学习目录](curriculum/PROGRESS.md)
- [课程总览](curriculum/README.md)
- [Day 1–28 核心课程](curriculum/core/README.md)
- [Day 29–35 可选 GNN](curriculum/optional_gnn/README.md)
- [主动学习 Unit 1–9 专题](curriculum/active_learning/README.md)
- [完整 Day/Unit 学习包使用方法](curriculum/shared/day_package_guide.md)
- [数据目录](data/README.md)
- [实验目录](experiments/README.md)
- [项目文档](docs/README.md)

## 当前可复现实验

已有 ESOL 参考基线使用固定 scaffold 划分、1024 维 ECFP，以及 Dummy、Ridge、决策树和随机森林。它作为 Day 7 的综合代码材料；实验说明、结果和证据边界保存在 [实验目录](experiments/esol/day01_baseline/README.md)，公开数据来源见 [ESOL 数据说明](data/public/esol.md)。

Day 02–35 和主动学习 Unit 01–09 的实验起始文件也已全部放入
[`experiments/` 完整索引](experiments/INDEX.md)。这些工作区含完整代码，
但保存输出已清空并统一标记为“待本人运行”。因此任务卡和工作区已经写好
都不等于本人已经完成；只有亲自运行、检查、解释并保存的结果才是学习证据。

## 环境与复现

核心路线已验证环境为 Python 3.10.20、DeepChem 2.8.0、RDKit 2026.3.3 和 scikit-learn 1.7.2。

```bash
conda create -n esol python=3.10.20 -y
conda activate esol
python -m pip install -r requirements-learning.txt

python -m nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  --inplace \
  experiments/esol/day01_baseline/esol_baseline.ipynb
```

首次运行可能需要联网下载 ESOL；缓存写入 `.cache/deepchem/`，不进入 Git。

Day 02–35 的个人起始副本已经预建。若某个目录被误删，可用下列命令
安全补建，已有个人文件不会被覆盖：

```bash
python scripts/start_day.py 2
```

脚本会把该日 `tutorial.ipynb` 复制到 `experiments/dayXX_topic/`，同时建立个人 `notes.md` 和 `results/` 说明。

主动学习 Unit 01–09 的副本也已经预建；下列命令用于单独补建：

```bash
python scripts/start_unit.py 1
```

个人 Unit 实验会放入 `experiments/active_learning/unitXX_topic/`，课程源文件保持不变。

真正开始和完成一个工作区时，用状态命令留下诚实边界：

```bash
python scripts/set_workspace_status.py day02 in_progress
python scripts/set_workspace_status.py day02 completed
```

完成命令会检查 Notebook 是否全部执行且学习笔记是否已经填写，但不会替你
勾选 `curriculum/PROGRESS.md`。

Day 29–35 使用单独的 GNN 环境，避免改变已经验证的核心环境。以下版本已在本仓库的全部 GNN 教学 Notebook 上验证：

```bash
conda create -n gnn python=3.10.20 -y
conda activate gnn
python -m pip install -r requirements-gnn.txt
python -m ipykernel install --user --name gnn --display-name "Python 3 (gnn)"
```

安装原则来自 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 和 [PyTorch Geometric 官方安装说明](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html)。本课程只使用 PyG 的基础功能，不要求安装额外的编译扩展。

Day 34 首次运行会下载公开 MUTAG 到 `.cache/pyg/`；该缓存不进入 Git。

## 仓库自检

```bash
python scripts/check_repository.py
python scripts/run_curriculum_notebooks.py --first-day 2 --last-day 28
python scripts/run_curriculum_notebooks.py \
  --first-day 29 \
  --last-day 35 \
  --kernel-name gnn
python scripts/run_active_learning_notebooks.py \
  --first-unit 1 \
  --last-unit 9
```

第一条验证目录结构、35 个 Day、9 个主动学习 Unit、Markdown 链接、
Python 代码块、43 份教学 Notebook、43 个完整实验起始工作区、ESOL
结果文件和 Excel 包结构；其余命令从头执行核心、GNN 与主动学习教程，
但默认只在内存中检查，不修改教材。只有课程维护者确实要刷新预存输出时
才显式添加 `--in-place`。GitHub Actions 会运行结构检查，但不会自动
标记任何 Day 或 Unit 完成。

## 分支规则

`main` 是唯一长期分支。课程、数据、实验和文档修改使用短期分支，检查通过并合并后删除。完整规则见 [分支与仓库维护规则](docs/development_workflow.md)。

## 数据安全边界

- 仓库中的粘合剂 Excel 是空白字段讨论稿，不是真实数据集；
- 真实配方或实验记录上传前必须确认保密、署名和共享权限；
- 未授权数据放在受控位置，不提交到 GitHub；
- 3–5 行样例只用于格式验收，不能作为可靠训练数据。
