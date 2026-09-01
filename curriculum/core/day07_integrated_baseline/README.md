# Day 7：把算法映射到完整 ESOL 基线

## 今天为什么学

前六天已经分别学习了：

- 回归、`X`、`y`、训练与预测；
- MAE、RMSE、R²；
- Ridge；
- 决策树；
- 随机森林；
- 梯度提升。

今天才第一次从头观察完整 ESOL Notebook。重点不是逐行掌握工程代码，而是确认：

> 真实项目虽然代码更长，算法主线仍然是“数据 → 模型 → fit → predict → metric”。

## 前置条件

- 已完成 [Day 6 梯度提升](../day06_gradient_boosting/README.md)；
- 能口头解释前六天的算法概念；
- 已阅读 [ESOL Notebook 对照地图](../day01_beginner/05_esol_notebook_map.md)；
- 已按根目录说明安装 `requirements-learning.txt`；
- 今天不修改模型参数，也不使用测试集选择模型。

## 完整学习包（按顺序）

1. [概念：算法主线与工程支持](01_concepts.md)
2. [算法推演：10 个代码单元审计与重跑](02_algorithm_walkthrough.md)
3. [课程提供的静态审计 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

本日 Tutorial 只审计已有文件，不重复训练。真正的运行对象仍是 [ESOL 参考基线 Notebook](../../../curriculum/core/day07_integrated_baseline/reference_baseline/esol_baseline.ipynb)；只有你本人 Restart + Run All 成功并记录状态后，才算个人学习证据。不要复制同一基线到另一个实验目录并冒充第二次独立实验。

## 今日产出

1. 一张10个代码单元的“算法核心/工程支持”分类表；
2. 一次从空内核开始的完整运行检查记录；
3. 对 warning、traceback 和 `AssertionError` 的区别说明；
4. 一段当前模型结果的中文解释；
5. 一份“哪些代码今天仍然可以不会”的清单。

## 核心概念

### 1. 完整代码为什么更长

研究代码除了算法，还要处理：

- 导入软件包；
- 确认仓库路径；
- 下载与缓存数据；
- 检查数据形状和重叠；
- 固定随机种子；
- 保存参数、版本和结果；
- 自动检查流程是否符合约定。

这些属于可复现研究需要的工程支持。它们不等于新的机器学习算法。

### 2. Notebook 内核保存状态

Notebook 内核会记住已经创建的变量。

如果跳过前面单元，后面仍可能错误地使用旧变量。因此某一格能运行，不代表整套流程可以从头复现。

安全检查顺序：

```text
Restart Kernel
→ 清除旧变量
→ Run All
→ 从第一格顺序执行
```

### 3. warning、traceback 与断言

| 输出 | 含义 |
|---|---|
| warning | 提醒潜在问题，程序可能继续 |
| traceback | Python 异常，当前执行已经中断 |
| `AssertionError` | 代码主动发现条件不符合并停止 |
| 普通 `print` | 显示运行信息，不是错误 |

DeepChem 提示缺少某些可选深度学习框架，不一定影响当前 ECFP＋sklearn 实验。出现 traceback 或断言失败时才应停下来检查。

### 4. 结果不是算法本身

Notebook 保存了一次已经运行过的结果。你重新运行是在验证这套实现是否可以复现，不是在证明随机森林对所有数据都最好。

当前 ESOL 结果也不能改名为下游任务预测结果。

这份已保存的基线包含 Dummy、Ridge、两棵决策树和随机森林，不包含 Day 6 的梯度提升。Day 6 使用人工数据学习算法直觉；梯度提升会在后面的公平比较中再加入，因此今天不需要在 Notebook 中强行找到它。

## 分步骤任务

### 第一步：先看地图，不运行

打开：

- [Day 1 Notebook 对照地图](../day01_beginner/05_esol_notebook_map.md)；
- [真实 ESOL Notebook](../../../curriculum/core/day07_integrated_baseline/reference_baseline/esol_baseline.ipynb)。

把10个代码单元分类：

```text
算法核心：
工程支持：
```

### 第二步：确认环境

在仓库根目录检查：

```bash
conda activate esol
which python
python --version
python -c "import deepchem, rdkit, sklearn, numpy, pandas; print('imports ok')"
```

如果你使用了别的环境名，请把第一行替换成自己的名字。不要在缺少依赖的裸 Python 或 `base` 环境中继续。

终端环境正确后，还要确认 Jupyter 当前 Kernel 指向同一个环境。终端导入成功而 Notebook Kernel 选错，Notebook 仍然会报 `ModuleNotFoundError`。

如果出现 traceback，先停止，不要继续点击后面的代码单元。

### 第三步：从空内核运行

在 Jupyter 中选择 Restart Kernel and Run All。

记录：

- 执行号是否从1开始连续；
- 是否出现 traceback；
- 是否出现 `AssertionError`；
- 最后一个代码单元是否执行；
- 结果表是否重新产生；
- 本轮是否仍未生成新的测试集预测。

不要修改模型参数。运行后如果 Notebook 出现无意义的元数据变化，不要急着提交 GitHub。

### 第四步：只跟踪五个核心动作

在 Notebook 中指出：

```text
X 和 y 在哪里形成
模型在哪里创建
fit 在哪里调用
predict 在哪里调用
指标在哪里计算
```

### 第五步：解释结果

阅读参考基线保存的指标表和运行配置，用自己的话回答：

- 当前验证 RMSE 最低的是哪个固定候选；
- 为什么不限深决策树表现出过拟合；
- 为什么一次固定验证集不能证明算法普遍最好；
- 为什么这些数字不是下游任务结果。

## 核心代码：只认骨架

完整 Notebook 很长，但今天真正必须认出的仍然只有：

下面是结构示意，`SomeModel` 和 `metric` 是占位名称，不能作为独立代码直接运行。

```python
model = SomeModel()
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
score = metric(y_valid, prediction)
```

遇到陌生代码时，先判断它是否属于这四步。若不属于，它很可能是数据准备、检查、记录或保存代码，可以以后再查。

需要时使用：

- [Python 语法字典](../day01_beginner/python_basics.md)；
- [Notebook 逐行讲解索引](../day01_beginner/notebook_line_by_line.md)；
- [逐行讲解 Part 1](../day01_beginner/notebook_line_by_line_part1.md)；
- [逐行讲解 Part 2](../day01_beginner/notebook_line_by_line_part2.md)。

不要把四份参考材料从头全部重读。

## 常见错误

- 在还没理解算法前重新逐行硬啃全部工程代码；
- 只运行最后几格，没有重启内核；
- 把 warning 一律当成实验失败；
- 出现 traceback 后继续相信后面的旧输出；
- 看到保存的旧结果就以为本次已经运行；
- 修改多个模型参数后仍称为原始基线；
- 使用 test 反复比较模型；
- 把“仓库可复现”误认为“自己已经掌握”；
- 把 ESOL 结果写成真实下游任务结论。

## 完成标准

- 能区分算法核心代码与工程支持代码；
- 能从空内核顺序运行；
- 能分辨 warning、traceback 和断言；
- 能找到 `X/y → fit → predict → metric`；
- 能解释当前结果中的一个过拟合例子；
- 能说明当前证据边界；
- 没有要求自己一次掌握全部路径、Git、JSON 和 pandas 代码；
- 没有读取测试集进行模型选择。

## 自测问题

1. 为什么完整 Notebook 比四行算法骨架长很多？
2. Restart Kernel 会清除什么？
3. warning 与 traceback 的核心区别是什么？
4. 哪些代码属于算法，哪些属于可复现工程？
5. 为什么某一格能运行不代表整套流程可复现？
6. 当前使用什么数据选择候选模型？
7. 为什么随机森林当前较好不等于它普遍最好？
8. 为什么 ESOL 结果不能作为下游任务结果提交？

## 导航

- 上一天：[Day 6 梯度提升](../day06_gradient_boosting/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 8 数据划分协议](../day08_split_protocol/README.md)
