# ML-Learning：从零开始的机器学习课程

一个面向中文学习者的系统化机器学习教学库。它强调的不只是“代码能够运行”，还包括算法直觉、手算推演、数据边界、可复现评价和结果解释。

仓库现在只承担教学用途，不保存个人实验工作区、真实项目数据或研究项目交付物。

> 第一次使用：完成 [10 分钟上手](docs/getting_started.md)，再从 [唯一学习清单](curriculum/PROGRESS.md) 的第一个未完成项开始。

## 这套课程适合谁

- Python 初学者，希望第一次系统学习机器学习；
- 会调用 `fit()`，但不确定数据划分、调参和评价是否正确的学习者；
- 希望比较传统模型、MLP、混合模型和 GNN 的进阶学习者；
- 想理解池式主动学习、代理模型、不确定性和采集函数的人；
- 需要一套可用于自学、读书会或组内培训的中文课程。

## 学完能做到什么

完成 Day 1–28 后，你应能独立：

1. 把一个回归问题整理为特征 `X`、目标 `y`、划分和指标；
2. 建立 Dummy、Ridge、决策树、随机森林和 Gradient Boosting 基线；
3. 使用交叉验证、多随机种子和 Pipeline 得到可信结果；
4. 在不偷看测试集的前提下完成调参和模型比较；
5. 训练小型 MLP，并诊断过拟合、正则化和早停；
6. 使用 OOF Stacking 避免混合模型中的标签泄漏；
7. 区分预测模型、不确定性、采集函数、主动学习和贝叶斯优化；
8. 用模型卡和可复现报告说明结果、限制与适用范围。

Day 29–35 提供可选 GNN 路线，Unit 1–9 系统讲解主动学习。不同目标对应的起点和停止点见 [学习路线选择器](curriculum/LEARNING_PATHS.md)。

## 课程地图

| 阶段 | 内容 | 单元 | 主要学习产物 |
|---|---|---:|---|
| 机器学习基础 | 指标、Ridge、树模型、集成模型 | Day 1–7 | 可解释的回归基线 |
| 可信评价 | 划分、交叉验证、种子、Pipeline、调参 | Day 8–14 | 无泄漏模型比较与阶段报告 |
| 神经网络 | shape、前向传播、优化、MLP、早停 | Day 15–21 | 与传统模型同协议的 MLP 基线 |
| 混合与应用 | OOF Stacking、不确定性、主动学习、模型卡 | Day 22–28 | 消融、候选选择和课程总结 |
| 可选 GNN | 图数据、GCN、GraphSAGE、GAT、GIN | Day 29–35 | 图学习基线与数据就绪检查 |
| 主动学习专题 | 代理模型、采集函数、多轮、批量和约束 | Unit 1–9 | 学习曲线、Random 对照和完整闭环 |

每个 Day/Unit 都采用相同的学习节奏：

```text
任务卡 → 中文概念 → 算法推演 → 教学 Notebook → 独立练习 → 参考答案 → 学习总结
```

`Day` 和 `Unit` 是学习模块，不要求在一个自然日内完成。

## 10 分钟开始学习

### 1. 下载并进入仓库

```bash
git clone https://github.com/kelvint0714-arch/ML-practice.git
cd ML-practice
```

### 2. 创建核心课程环境

```bash
conda create -n esol python=3.10.20 -y
conda activate esol
python -m pip install -r requirements-learning.txt
```

### 3. 检查环境与进度

```bash
python scripts/learn.py doctor --track core
python scripts/learn.py status
```

`doctor` 只检查环境，不安装或删除软件；`status` 只读取进度，不会自动勾选课程。

### 4. 打开下一课

例如当前下一项是 Day 2：

```text
curriculum/core/day02_metrics/README.md
```

按照任务卡顺序阅读并运行 `tutorial.ipynb`。自己的笔记和运行产物可放在本地 [`learning_outputs/`](learning_outputs/README.md)；除说明文件外，该目录默认不提交 Git。

## 三条学习规则

1. **先解释，再运行。** 每段代码都要能说清输入、变换和输出。
2. **测试集不是调参工具。** 模型选择只能使用训练区或内部验证。
3. **运行成功不等于理解。** 完成一个单元前，必须能解释结果和常见错误。

教学 Notebook 中保存的输出只证明课程示例可以执行，不代表学习者已经掌握。建议 Restart Kernel and Run All，再独立完成练习。

## 仓库结构

```text
ML-Learning/
├── curriculum/              # 35 个 Day、9 个主动学习 Unit
├── data/public/             # 公开教学数据说明
├── docs/                    # 上手、教学和维护文档
├── learning_outputs/        # 本地学习笔记与输出（默认忽略）
├── scripts/                 # 学习助手和课程验证工具
├── tests/                   # 学习工具测试
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
- [统一评价协议](curriculum/shared/experiment_protocol.md)
- [常见错误排查](curriculum/shared/error_guide.md)

## 环境说明

核心路线使用 Python 3.10.20、DeepChem 2.8.0、RDKit 2026.3.3 和 scikit-learn 1.7.2。主动学习专题复用核心环境。

GNN 使用独立环境：

```bash
conda create -n gnn python=3.10.20 -y
conda activate gnn
python -m pip install -r requirements-gnn.txt
python -m ipykernel install --user --name gnn --display-name "Python 3 (gnn)"
python scripts/learn.py doctor --track gnn
```

首次运行 ESOL 或 MUTAG 课程可能需要联网下载公开数据，缓存写入 `.cache/`，不会提交 Git。

## 课程验证

快速检查：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

从头执行教学 Notebook：

```bash
python scripts/run_curriculum_notebooks.py --first-day 2 --last-day 28
python scripts/run_curriculum_notebooks.py --first-day 29 --last-day 35 --kernel-name gnn
python scripts/run_active_learning_notebooks.py --first-unit 1 --last-unit 9
```

这些命令默认只在内存中执行，不改写课程源文件。贡献课程前请阅读 [贡献指南](CONTRIBUTING.md)，授课或组织读书会可参考 [教学与自学指南](docs/teaching_guide.md)。

## 数据与内容边界

- 仓库只包含公开教学数据说明和小型人工示例；
- 不提交真实项目数据、账号凭据、API key 或未授权材料；
- 教学示例结果不能直接外推到新的真实任务；
- 模型比较必须使用相同数据、划分、指标和选择规则；
- 练习与参考答案保持分离。

项目目前尚未声明开源许可证。在许可证确定前，不要默认拥有复制、再发布或商用授权。
