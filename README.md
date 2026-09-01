# ML-practice：从零开始的机器学习实验课

一套面向中文学习者、化学与材料研究者的机器学习自学课程。仓库不只提供“能运行的代码”，还要求学习者解释输入、算法、输出、误差来源和结论边界。

课程主线覆盖传统机器学习、可信验证、MLP、混合模型与主动学习；图神经网络（GNN）作为满足数据条件后再进入的可选路线。

> 第一次打开？先完成 [10 分钟上手](docs/getting_started.md)，然后从 [唯一学习清单](curriculum/PROGRESS.md) 的第一个未完成项开始。

## 这套课程适合谁

- 会一点 Python，但还不能独立完成机器学习实验的初学者；
- 希望系统补齐数据划分、交叉验证、泄漏防护和公平比较的研究生；
- 准备用机器学习处理化学、材料或小样本表格数据的研究者；
- 想学习池式主动学习、代理模型、不确定性和采集函数的进阶学习者。

如果你只想复制一个训练脚本，这个仓库可能显得太慢；如果你希望知道“为什么这样做、结果能说明什么”，它会更合适。

## 学完能做到什么

完成核心 Day 1–28 后，你应能独立：

1. 把研究问题整理为特征 `X`、目标 `y`、数据划分和评价指标；
2. 建立 Dummy、Ridge、树模型、集成模型和小型 MLP 基线；
3. 使用交叉验证、多随机种子和 Pipeline 得到更可信的结果；
4. 在不偷看测试集的前提下调参和比较模型；
5. 识别预处理泄漏、训练内预测泄漏和错误的模型结论；
6. 使用 OOF Stacking 组合传统模型与神经网络；
7. 区分预测、不确定性、采集函数、Oracle、主动学习与贝叶斯优化；
8. 把公开数据上的方法证据与真实材料结论严格分开。

完整能力地图与不同学习路线见 [学习路线选择器](curriculum/LEARNING_PATHS.md)。

## 课程地图

| 阶段 | 内容 | 学习单元 | 主要产物 |
|---|---|---:|---|
| 机器学习基础 | 指标、Ridge、决策树、随机森林、Boosting | Day 1–7 | 可解释的回归基线 |
| 可信实验 | 划分、交叉验证、种子、Pipeline、调参 | Day 8–14 | 无泄漏模型比较与阶段报告 |
| 神经网络 | shape、前向传播、优化、MLP、早停 | Day 15–21 | 与传统模型同协议的 MLP 基线 |
| 混合与推荐 | OOF Stacking、不确定性、主动学习桥接 | Day 22–28 | 消融、候选选择和数据接入清单 |
| 可选 GNN | 图数据、GCN、GraphSAGE、GAT、GIN | Day 29–35 | 公开图实验与 GNN 启动评审 |
| 主动学习专题 | 代理模型、采集函数、多轮与批量闭环 | Unit 1–9 | 学习曲线、Random 对照与闭环协议 |

`Day` 和 `Unit` 是学习模块，不要求一天完成。课程采用同一种学习节奏：

```text
任务卡 → 中文概念 → 算法推演 → 教学 Notebook → 独立练习 → 参考答案 → 个人实验记录
```

## 10 分钟开始学习

### 1. 获取仓库

```bash
git clone https://github.com/kelvint0714-arch/ML-practice.git
cd ML-practice
```

如果你已经在仓库中，直接进入下一步。

### 2. 建立核心课程环境

```bash
conda create -n esol python=3.10.20 -y
conda activate esol
python -m pip install -r requirements-learning.txt
```

### 3. 检查环境和课程状态

```bash
python scripts/learn.py doctor --track core
python scripts/learn.py status
```

`doctor` 只检查，不会安装或删除软件；`status` 会显示完成度和下一项任务。

### 4. 打开第一项未完成课程

当前学习进度以 [curriculum/PROGRESS.md](curriculum/PROGRESS.md) 为唯一来源。Day 2–35 和 Unit 1–9 已准备好个人实验副本；开始运行前标记为进行中：

```bash
python scripts/set_workspace_status.py day02 in_progress
jupyter lab
```

更完整的安装说明、常见失败和首次运行检查见 [新手上手指南](docs/getting_started.md)。

## 三条使用规则

1. **先解释，再运行。** 每个代码块都要能说清输入、变换和输出。
2. **测试集不是练习答案。** 选模型、调参和修流程只使用训练区或内部验证。
3. **代码存在不等于实验完成。** 只有本人运行、检查、解释并保存的结果才是学习证据。

课程源文件与个人实验严格分开：

| 位置 | 作用 | 是否是个人实验成果 |
|---|---|---|
| `curriculum/` | 教材、教学 Notebook、练习与参考答案 | 否 |
| `experiments/` 中未运行的起始副本 | 个人工作区模板 | 否 |
| `experiments/` 中本人运行并解释的内容 | 个人 Notebook、笔记与结果 | 通过自测后可以 |

## 仓库结构

```text
ML-practice/
├── curriculum/              # 35 个 Day、9 个 Unit 与共享参考
├── experiments/             # 个人实验工作区和 ESOL 参考实验
├── data/                    # 公开数据说明与粘合剂数据接口
├── docs/                    # 上手、教学、路线与维护文档
├── scripts/                 # 学习助手、工作区和质量检查工具
├── tests/                   # 不依赖训练环境的工具测试
├── requirements-learning.txt
├── requirements-active-learning.txt
└── requirements-gnn.txt
```

常用入口：

- [学习路线选择器](curriculum/LEARNING_PATHS.md)
- [唯一学习清单](curriculum/PROGRESS.md)
- [Day 1–28 核心课程](curriculum/core/README.md)
- [Day 29–35 可选 GNN](curriculum/optional_gnn/README.md)
- [主动学习 Unit 1–9](curriculum/active_learning/README.md)
- [机器学习术语表](curriculum/shared/ml_glossary.md)
- [常见错误排查](curriculum/shared/error_guide.md)
- [实验工作区索引](experiments/INDEX.md)

## 环境说明

核心路线的已验证环境为 Python 3.10.20、DeepChem 2.8.0、RDKit 2026.3.3 和 scikit-learn 1.7.2。主动学习主路线复用核心环境。

GNN 使用独立环境，避免与核心路线互相影响：

```bash
conda create -n gnn python=3.10.20 -y
conda activate gnn
python -m pip install -r requirements-gnn.txt
python -m ipykernel install --user --name gnn --display-name "Python 3 (gnn)"
python scripts/learn.py doctor --track gnn
```

首次运行 ESOL 或 MUTAG 课程可能需要联网下载公开数据。缓存写入 `.cache/`，不会提交到 Git。

## 课程维护与验证

快速结构检查不运行模型训练：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

从头执行全部教学 Notebook：

```bash
python scripts/run_curriculum_notebooks.py --first-day 2 --last-day 28
python scripts/run_curriculum_notebooks.py --first-day 29 --last-day 35 --kernel-name gnn
python scripts/run_active_learning_notebooks.py --first-unit 1 --last-unit 9
```

这些命令默认只在内存中检查课程源 Notebook。贡献课程前请阅读 [贡献指南](CONTRIBUTING.md)；使用课程授课可参考 [教学与自学指南](docs/teaching_guide.md)。

## 研究与数据边界

- ESOL、MUTAG 和人工数据只用于方法学习，不能替代真实粘合剂证据；
- 仓库中的粘合剂 Excel 是空白字段讨论稿，不是真实数据集；
- 未授权配方和实验记录不得提交到 GitHub；
- 真实数据接入前，需要确认单位、重复实验、批次、分组和测试条件；
- 只有获得可追溯结构表示后，才评估是否启动真实材料 GNN。

两条研究工作线和衔接条件见 [项目路线图](docs/project_roadmap.md)。

## 当前状态

- Day 1 的基础步骤已经标记完成；其余完成情况以个人进度表为准；
- Day 2–35 和 Unit 1–9 的教学包与未运行实验工作区已准备；
- [ESOL 参考基线 E01](experiments/esol/day01_baseline/README.md) 已保存可复现结果，数据来源见 [ESOL 说明](data/public/esol.md)；
- 当前仓库自检覆盖课程结构、链接、代码块、Notebook、实验工作区和数据模板。

项目尚未声明开源许可证。在许可证确定前，可以阅读和学习仓库内容，但不要默认拥有复制、再发布或商用授权。
