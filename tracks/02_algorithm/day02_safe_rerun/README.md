# Day 2：安全重跑 Day 1 实验

## 今天为什么学

Day 1 已经提供了一份能够运行的 ESOL 基线，但“仓库里有结果”不等于你会独立运行。

今天不增加新算法，只练习一件非常重要的事：

> 从干净内核按顺序重跑实验，并判断一次运行究竟成功还是失败。

如果这一关没有掌握，后面修改参数时很容易使用旧变量、旧输出或错误环境。

## 前置条件

- 已阅读 [Day 1 零基础学习入口](../day01_beginner/README.md)；
- 知道 Notebook 包含 Markdown 单元和代码单元；
- 能认出变量、函数调用、参数和返回值；
- 已安装仓库 `requirements.txt` 中的依赖；
- 今天不修改 Day 1 的模型参数。

## 今日产出

今天应由你亲自得到以下内容：

1. 一次从空内核开始的完整运行；
2. 一张“运行前—运行中—运行后”检查清单；
3. 对 warning、traceback 和 assertion 的区别说明；
4. 一段用来检查 ESOL 数据形状的短代码；
5. 三句话记录本次运行是否可信。

这些是学习任务，不表示仓库中的实验已经由你完成。

## 核心概念

### 1. 内核保存运行状态

Notebook 内核会记住已经创建的变量。

如果跳过前面的单元，后面仍可能暂时使用旧变量，因此“某一格能运行”不代表整个流程正确。

### 2. Restart 和 Run All

安全重跑的基本动作是：

```text
Restart Kernel
→ 清除旧变量
→ Run All
→ 从第一格依次执行
```

执行编号应该从 1 开始连续增加。

### 3. warning 不等于失败

- warning：程序提醒潜在问题，但可能继续运行；
- traceback：Python 抛出异常，当前流程已经中断；
- `AssertionError`：代码主动检查到条件不符合；
- 普通打印信息：用于说明当前进度，不是报错。

## 分步骤任务

### 第一步：确认终端位置

在终端进入仓库根目录，然后运行 `pwd` 和 `git status`。

只观察状态，不要使用 reset、checkout 或清理命令。

### 第二步：确认 Python 环境

运行 `python --version`，确认使用的是课程环境，而不是另一个 base 环境。

再运行一条只读导入检查：

```bash
python -c "import deepchem, sklearn, numpy, pandas; print('imports ok')"
```

### 第三步：打开并观察 Day 1 Notebook

从仓库中的真实 Day 1 Notebook 打开，选择 Restart Kernel and Run All。

记录：

- 是否从第一格开始；
- 执行号是否连续；
- 是否出现 traceback；
- 是否出现 `AssertionError`；
- 最后一格是否执行；
- 模型结果表是否重新显示。

### 第四步：检查数据形状

在自己的临时练习单元中运行下面的核心代码。

## 核心代码骨架

```python
import numpy as np
import deepchem as dc

SEED = 42

featurizer = dc.feat.CircularFingerprint(
    radius=2,
    size=1024,
)

tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer=featurizer,
    splitter="scaffold",
    transformers=[],
)

train_dataset, valid_dataset, test_dataset = datasets

X_train = np.asarray(train_dataset.X)
y_train = np.asarray(train_dataset.y).reshape(-1)
X_valid = np.asarray(valid_dataset.X)
y_valid = np.asarray(valid_dataset.y).reshape(-1)

print("train:", X_train.shape, y_train.shape)
print("valid:", X_valid.shape, y_valid.shape)
print("test rows:", len(test_dataset))

assert X_train.shape[0] == y_train.shape[0]
assert X_valid.shape[0] == y_valid.shape[0]
assert np.isfinite(X_train).all()
assert np.isfinite(y_train).all()
```

今天新增语法只有两点：

- `.all()`：要求数组中的所有布尔值都为真；
- `assert 条件`：条件为假时立即停止，防止错误继续传播。

不要在今天对 `test_dataset` 调用模型预测。

## 常见错误

- 在 base 环境运行，导致包版本与 Day 1 不同；
- 只运行最后几格，没有重启内核；
- 把 DeepChem 可选依赖 warning 当成当前实验失败；
- 出现 traceback 后仍继续引用旧变量；
- 看到旧输出就认为本次已经运行；
- 为了“清理”仓库而执行破坏性的 Git 命令；
- 把 test 样本数打印出来误解为已经评价 test；
- 修改 Day 1 参数后仍把结果叫作原始基线。

## 完成标准

只有同时满足以下条件，Day 2 才算完成：

- 能从干净内核按顺序运行；
- 能确认 Python 环境和仓库位置；
- 能分辨 warning、traceback 和 `AssertionError`；
- 能解释四个 shape 中每个数字的含义；
- 能说明输出为什么可能是历史快照；
- 没有训练、调参或反复查看 test；
- 写下本次运行的日期、环境和是否成功；
- 能指出运行失败时应先停在哪一格检查。

## 自测问题

1. 为什么只运行最后一个代码单元不可靠？
2. Restart Kernel 会清除什么？
3. warning 与 traceback 的核心区别是什么？
4. `np.isfinite(y_train).all()` 在检查什么？
5. `assert` 是训练模型还是检查条件？
6. 为什么打印 test 样本数不等于使用 test 调参？
7. 如何判断 Notebook 中的结果是本次重新生成的？
8. 如果中间一格失败，为什么不应继续相信后面的旧输出？

## 导航

- 上一天：[Day 1 零基础学习入口](../day01_beginner/README.md)
- 下一天：[Day 3 评价指标](../day03_metrics/README.md)
