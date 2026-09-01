# Day 8：数据划分协议与测试集边界

## 今天为什么学

模型看过哪些数据，决定了一个指标能够说明什么。如果一边调整模型一边反复查看 test，test 就会逐渐参与模型选择，不再代表真正未见数据。

今天不追求更好的数字，而是把 train、validation 和 test 的职责写成明确协议。

## 前置条件

- 已完成 [Day 7 完整 ESOL 基线](../day07_integrated_baseline/README.md)；
- 知道模型使用训练集执行 `fit`；
- 知道验证集用于开发阶段比较；
- 已在正确的课程环境中从仓库根目录启动 Notebook；
- 今天不增加新模型。

## 完整学习包（按顺序）

1. [概念：split 职责、泄漏与 test 边界](01_concepts.md)
2. [算法推演：split 表与精确 ID 检查](02_algorithm_walkthrough.md)
3. [课程提供的可运行 Tutorial](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [折叠参考答案](04_reference_answers.md)

`tutorial.ipynb` 只加载 split 做结构检查，不训练模型、不产生 test 预测。个人检查表、政策和笔记应在实际学习时保存到 `learning_outputs/day08_split_protocol/`；课程预存输出不代表你已经完成。

## 今日产出

今天应完成：

1. 一张 split 样本数检查表；
2. 三个 split 的 ID 重叠检查；
3. 一份测试集使用规则；
4. 对当前 ESOL test 已暴露事实的说明；
5. 一个不调用 test 预测的模型开发流程图。

这些检查只证明精确 ID 是否重叠，不能证明所有分子骨架完全独立。

## 核心概念

### 1. 训练集

训练集用于：

- 拟合模型参数；
- 拟合 scaler；
- 建立树的切分；
- 学习线性系数。

### 2. 验证集

验证集用于开发阶段：

- 比较事先固定的候选模型；
- 选择超参数；
- 观察过拟合；
- 决定是否继续改进。

反复使用同一个验证集也会产生开发偏差，因此需要记录尝试次数。

### 3. 测试集

理想情况下，test 只在方案完全冻结后使用一次。

当前仓库的旧实验已经查看过原有 ESOL test，因此它不能被重新包装成严格未见证据。

今天仍然不产生新的 test 预测。

### 4. 数据泄漏

泄漏是模型训练过程直接或间接获得本不应该得到的信息。

常见例子：

- 在全数据上拟合 scaler；
- 先看 test 再调参数；
- 同一样本同时进入 train 和 valid；
- 用验证标签构造训练特征。

## 分步骤任务

### 第一步：从空内核重新加载数据

Day 8 的代码必须自包含，不能默认继承 Day 7 内核中的旧变量。Restart Kernel 后，从下面代码第一行开始运行。

### 第二步：写明每个 split 的用途

不要只写样本数，要写“谁可以 fit、谁可以比较、谁暂不使用”。

### 第三步：建立 split 表

表中至少包含 split 名称、样本数、特征维数、ID 数和当前用途。train、validation 可以检查标签；test 只检查 `X`、样本数和 ID，不读取 `test_dataset.y`。

### 第四步：检查有限值

确认三个 split 的 `X` 没有无穷或 NaN，并只确认 train、validation 的 `y` 没有无穷或 NaN。这里主动不读取 test 标签，是把“暂不使用 test”落实到代码边界。

### 第五步：检查 ID 交集

分别检查 train-valid、train-test、valid-test。

### 第六步：写测试集政策

写明：“核心路线 Day 2–28 的模型开发阶段不使用原 test 选择模型；只有方案预先冻结并明确最终评价协议后，才允许一次性评价真正未见的测试数据。”

## 核心代码骨架

```python
from pathlib import Path
import deepchem as dc
import numpy as np
import pandas as pd

data_dir = Path(".cache/deepchem")
data_dir.mkdir(parents=True, exist_ok=True)
featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)

tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer=featurizer,
    splitter="scaffold",
    transformers=[],
    reload=True,
    data_dir=str(data_dir),
    save_dir=str(data_dir),
)

train_dataset, valid_dataset, test_dataset = datasets

assert transformers == []

split_datasets = {
    "train": train_dataset,
    "valid": valid_dataset,
    "test": test_dataset,
}

rows = []

for split_name, dataset in split_datasets.items():
    X = np.asarray(dataset.X)
    ids = np.asarray(dataset.ids)
    may_inspect_labels = split_name != "test"
    y = np.asarray(dataset.y).reshape(-1) if may_inspect_labels else None

    rows.append({
        "split": split_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_ids": ids.shape[0],
        "n_labels_checked": y.shape[0] if may_inspect_labels else None,
        "all_X_finite": bool(np.isfinite(X).all()),
        "all_y_finite": bool(np.isfinite(y).all()) if may_inspect_labels else None,
        "label_policy": "允许完整性检查" if may_inspect_labels else "未读取",
    })

split_table = pd.DataFrame(rows)

id_sets = {
    name: set(dataset.ids)
    for name, dataset in split_datasets.items()
}

overlaps = {
    "train_valid": len(id_sets["train"] & id_sets["valid"]),
    "train_test": len(id_sets["train"] & id_sets["test"]),
    "valid_test": len(id_sets["valid"] & id_sets["test"]),
}

print(split_table)
print(overlaps)
```

今天新增语法：

- `Path(".cache/deepchem")`：使用仓库内的数据缓存位置；
- `tasks, datasets, transformers = ...`：接收加载函数返回的三部分；
- `set(...)`：建立去重集合；
- `集合A & 集合B`：求两个集合的交集；
- `dict.items()`：同时取得字典键和值；
- `bool(...)`：把 NumPy 布尔结果转成普通 Python 布尔值。

## 常见错误

- 把 validation 当作最终 test；
- 每调一次参数都查看 test；
- 在全部数据上先拟合 StandardScaler；
- 只检查样本数，不检查 ID 重叠；
- 认为精确 ID 不重叠就证明化学结构完全独立；
- 把已暴露 test 重新表述为严格未见；
- 为了提高指标删除验证集中的难样本；
- 不记录数据划分方式。

## 完成标准

- 能准确说明三个 split 的职责；
- split 表中三个 split 的样本数、特征数和 ID 数一致，train/validation 的标签数也一致；
- 代码没有读取 `test_dataset.y`；
- 三组精确 ID 交集均被检查；
- 能解释精确重叠检查的能力边界；
- scaler 和模型只在训练数据上 fit；
- Day 8 不产生新的 test 预测；
- 写下明确测试集政策；
- 能指出至少三种数据泄漏方式；
- 不把当前验证结果称为最终性能。

## 自测问题

1. 训练集可以用于哪些操作？
2. validation 与 test 的核心区别是什么？
3. 为什么反复看 test 会使它失去意义？
4. scaler 为什么也属于需要 fit 的对象？
5. 集合交集能够发现什么？
6. 精确 ID 无重叠为什么仍不能证明骨架完全独立？
7. 当前 ESOL test 为什么不能重新称为严格未见？
8. 今天为什么不需要训练一个新模型？

## 导航

- 上一天：[Day 7 完整 ESOL 基线](../day07_integrated_baseline/README.md)
- 完成验收后：[返回一步一步学习目录](../../PROGRESS.md)
- 下一天：[Day 9 K 折交叉验证](../day09_cross_validation/README.md)
