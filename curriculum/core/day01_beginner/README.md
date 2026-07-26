# Day 1：先理解算法，再接触代码

Day 1 不再要求你先读完整 ESOL Notebook，也不要求一次学完 Python 语法。今天先回答一个更根本的问题：

> 机器学习算法究竟接收什么、学习什么、输出什么？

代码只在概念和手算之后出现，而且只保留能说明算法骨架的最小代码。完整工程 Notebook 放到 Day 7 再运行。

## 今天的目标

完成 Day 1 后，你应该能够：

1. 区分“算法”“训练后的模型”和“Python 代码”；
2. 解释样本、特征 `X`、目标 `y` 和预测值；
3. 说明回归任务在解决什么问题；
4. 解释训练集、验证集和测试集的不同职责；
5. 手算均值基线、MAE 和 RMSE；
6. 用中文伪代码复述机器学习流程；
7. 运行并解释一段最小代码；
8. 在完整 ESOL Notebook 中找到数据、模型、训练、预测和评价的位置。

## 必须按这个顺序

| 顺序 | 学习材料 | 今天要做到什么 |
|---:|---|---|
| 1 | [算法概念](01_concepts.md) | 先不用代码，理解机器学习在做什么 |
| 2 | [纸笔手算](02_hand_calculation.md) | 自己算一次均值预测、MAE 和 RMSE |
| 3 | [中文伪代码](03_pseudocode.md) | 把算法步骤与 Python 语法分开 |
| 4 | [最小 Python 代码](04_minimal_code.md) | 只运行刚刚理解过的步骤 |
| 5 | [ESOL Notebook 对照地图](05_esol_notebook_map.md) | 在真实代码里找到五个核心位置，不逐行硬啃 |
| 6 | [Day 1 核心验收](exercises.md) | 不看答案完成验收 |

完成一个文件后，回到 [一步一步学习目录](../../PROGRESS.md) 勾选对应小项。前一项没有理解时，不要打开后一项。

## 现有长教程怎样使用

下面的材料保留为查询手册，但**不再是 Day 1 必读内容**：

- [Python 语法字典](python_basics.md)：遇到看不懂的符号时，只查询对应小节；
- [完整 Notebook 逐行讲解索引](notebook_line_by_line.md)；
- [逐行讲解 Part 1](notebook_line_by_line_part1.md)；
- [逐行讲解 Part 2](notebook_line_by_line_part2.md)；
- [扩展练习](exercises_extended.md)：核心验收完成后再选择性练习。

这些材料详细解释路径、Git、缓存、JSON、pandas 和结果保存等工程内容。它们对复现研究有用，但不是理解算法的前置条件。

## Day 1 不要求掌握

- DeepChem、RDKit 和 ECFP 的内部实现；
- Ridge、随机森林的数学推导；
- Git、路径、JSON 和 CSV 保存代码；
- 所有模型的超参数；
- GNN、Transformer 或主动学习。

今天只需要建立正确的算法骨架。下一步是 [Day 2：评价预测误差](../day02_metrics/README.md)。
