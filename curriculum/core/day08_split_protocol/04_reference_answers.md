# Day 8.4：参考答案

> 请先写自己的测试集政策。复制一段正确文字不等于能在真实代码中识别泄漏。

<details>
<summary>A. split 职责</summary>

| 动作 | 使用范围 |
|---|---|
| 拟合 Ridge 系数 | train |
| 拟合 scaler | train |
| 比较三个 `alpha` | validation（模型仍只在 train fit） |
| 决定继续改模型 | validation |
| 冻结后最终评价 | 真正未见的 test |
| 训练神经网络权重 | train |
| 选择早停轮次 | validation |

validation 标签用于评价当前候选，但不传入模型或预处理器的 `fit`。它参与开发决策，却不参与参数学习。

</details>

<details>
<summary>B. 找泄漏</summary>

情形 1 在 train+valid 上拟合 scaler，valid 分布流入预处理参数。修复：

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
```

情形 2 用 test 选择 `alpha`。应在 validation 或训练内交叉验证比较，并保持 test 不用：

```python
for alpha in [0.01, 0.1, 1, 10]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = rmse(y_valid, model.predict(X_valid))
```

情形 3 用 validation 标签选择特征，再在 train 拟合；validation 信息已经流入模型设计。特征选择应只在 train 或训练折内完成。

</details>

<details>
<summary>C. shape 契约</summary>

`y.reshape(-1)` 后是 `(113,)`。但 IDs 只有 112 个，而 X/y 有 113 个，行数契约失败。

不能随意删 X 的最后一行，因为不知道缺失的是哪个 ID，也可能使特征与标签错位。应回到数据加载和索引构造处找原因。

</details>

<details>
<summary>D. 集合交集</summary>

```text
train ∩ valid = {D}，大小 1
train ∩ test  = {A}，大小 1
valid ∩ test  = {}，大小 0
```

不应继续训练并报告可信评价。先修复划分，使同一 ID 不跨 split，再从头运行。

</details>

<details>
<summary>E. 能力边界</summary>

交集为 0 只直接支持第 1 项：“没有完全相同字符串 ID”。它不能保证不同写法不是同一分子、没有相似骨架、没有批次关联，也不能推出各 split 样本数相同。

</details>

<details>
<summary>F. test 暴露</summary>

1. 开发者已经获得分数信息，删除文件不会删除由此产生的决策影响。
2. 不自动可信；若看完结果再选择种子，仍可能挑出有利划分，而且划分单位和协议也需合理。
3. 在任何针对该外层数据的模型开发之前预先定义。
4. 可以做样本数、`X.shape`、`X` 有限值和 ID 重叠等完整性检查；本日不读取 test 标签，更不查看标签表现。
5. Day 8 不计算 test RMSE。

</details>

<details>
<summary>G. 政策示例</summary>

一种合格写法：

1. 所有会学习数据统计量的预处理器只在 train 或训练折内拟合；
2. 候选模型和参数只使用 validation 或训练内交叉验证比较；
3. 每次尝试均记录，模型、指标与代码冻结后才进入最终评价；
4. 历史已暴露的 ESOL test 只保留为历史记录，不再称为严格未见；
5. 新的最终结论使用预先保留、开发期间不可见的外层数据，并按预注册协议评价一次；
6. 最终评价后若继续修改模型，该结果转为开发证据，需要新的外层评价。

</details>

<details>
<summary>H. 检查函数</summary>

```python
import numpy as np
import pandas as pd

def validate_splits(split_datasets):
    split_rows = []
    id_sets = {}
    expected_n_features = None

    for split_name, dataset in split_datasets.items():
        X = np.asarray(dataset.X)
        ids = np.asarray(dataset.ids)
        may_inspect_labels = split_name != "test"
        y = np.asarray(dataset.y).reshape(-1) if may_inspect_labels else None

        if X.ndim != 2:
            raise ValueError(f"{split_name}: X 必须是二维")
        if len(X) != len(ids):
            raise ValueError(f"{split_name}: X/ids 行数不一致")
        if not np.isfinite(X).all():
            raise ValueError(f"{split_name}: X 含非有限值")
        if may_inspect_labels:
            if len(X) != len(y):
                raise ValueError(f"{split_name}: X/y 行数不一致")
            if not np.isfinite(y).all():
                raise ValueError(f"{split_name}: y 含非有限值")

        if expected_n_features is None:
            expected_n_features = X.shape[1]
        elif X.shape[1] != expected_n_features:
            raise ValueError(f"{split_name}: 特征维数不一致")

        id_sets[split_name] = set(map(str, ids))
        split_rows.append({
            "split": split_name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_ids": len(ids),
            "n_labels_checked": len(y) if may_inspect_labels else None,
            "all_X_finite": True,
            "all_y_finite": True if may_inspect_labels else None,
            "label_policy": "允许完整性检查" if may_inspect_labels else "未读取",
        })

    required = {"train", "valid", "test"}
    if set(id_sets) != required:
        raise ValueError("必须提供 train、valid、test")

    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]
    overlap_rows = []
    for left, right in pairs:
        overlap = id_sets[left] & id_sets[right]
        overlap_rows.append({
            "pair": f"{left}_{right}",
            "exact_id_overlap": len(overlap),
        })
        if overlap:
            raise ValueError(f"{left}/{right}: 发现精确 ID 重叠")

    return pd.DataFrame(split_rows), pd.DataFrame(overlap_rows)
```

</details>

核对后返回 [Day 8 任务卡](README.md)，用自己的话提交测试集政策。
