# Day 8.2：建立 split 表、重叠检查和测试集政策

配套代码见 [tutorial.ipynb](tutorial.ipynb)。Notebook 会加载三个 split 做结构检查，但不会调用任何模型的 test 预测。

## 1. 从仓库内定位缓存

不要依赖当前内核已有的 Day 7 变量：

```python
from pathlib import Path

def find_repo_root(start=None):
    current = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("请从 ML-Learning 仓库内部运行")

repo_root = find_repo_root()
data_dir = repo_root / ".cache" / "deepchem"
data_dir.mkdir(parents=True, exist_ok=True)

print("repository:", repo_root.name)
print("cache exists:", data_dir.exists())
```

输出只显示仓库名和相对含义，不把个人绝对路径写进提交的结果。

## 2. 显式加载 ESOL

```python
import deepchem as dc

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
print("task:", tasks)
print("split lengths:", [len(dataset) for dataset in datasets])
```

`transformers=[]` 保持原始 `logS` 目标空间。

## 3. 写明用途，而不只写名称

```python
split_datasets = {
    "train": train_dataset,
    "valid": valid_dataset,
    "test": test_dataset,
}

split_purposes = {
    "train": "允许 fit 模型和预处理器",
    "valid": "开发阶段比较；不允许 fit",
    "test": "历史已暴露；Day 8 不预测、不选参",
}
```

## 4. 建立输入契约表

```python
import numpy as np
import pandas as pd

rows = []

for split_name, dataset in split_datasets.items():
    X = np.asarray(dataset.X)
    ids = np.asarray(dataset.ids)
    may_inspect_labels = split_name != "test"
    y = np.asarray(dataset.y).reshape(-1) if may_inspect_labels else None

    rows.append({
        "split": split_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_ids": int(ids.shape[0]),
        "n_labels_checked": int(y.shape[0]) if may_inspect_labels else None,
        "all_X_finite": bool(np.isfinite(X).all()),
        "all_y_finite": bool(np.isfinite(y).all()) if may_inspect_labels else None,
        "label_policy": "允许完整性检查" if may_inspect_labels else "未读取",
        "purpose": split_purposes[split_name],
    })

split_table = pd.DataFrame(rows)
print(split_table.to_string(index=False))
```

## 5. 将检查写成断言

```python
expected_sizes = {
    "train": 902,
    "valid": 113,
    "test": 113,
}

assert split_table["n_features"].eq(1024).all()
assert split_table["all_X_finite"].all()
assert split_table.loc[
    split_table["split"].isin(["train", "valid"]),
    "all_y_finite",
].all()

for row in split_table.to_dict(orient="records"):
    split_name = row["split"]
    assert row["n_samples"] == expected_sizes[split_name]
    assert row["n_samples"] == row["n_ids"]
    if split_name != "test":
        assert row["n_samples"] == row["n_labels_checked"]
    else:
        assert pd.isna(row["n_labels_checked"])
```

若数据版本或加载协议故意改变，先更新书面协议，再更新断言；不要为了让红灯消失而随意删除检查。

## 6. 精确 ID 交集

```python
id_sets = {
    split_name: set(map(str, dataset.ids))
    for split_name, dataset in split_datasets.items()
}

overlap_rows = [
    {
        "pair": "train_valid",
        "exact_id_overlap": len(id_sets["train"] & id_sets["valid"]),
    },
    {
        "pair": "train_test",
        "exact_id_overlap": len(id_sets["train"] & id_sets["test"]),
    },
    {
        "pair": "valid_test",
        "exact_id_overlap": len(id_sets["valid"] & id_sets["test"]),
    },
]

overlap_table = pd.DataFrame(overlap_rows)
print(overlap_table.to_string(index=False))
assert overlap_table["exact_id_overlap"].eq(0).all()
```

结论只能写“没有发现精确 ID 重叠”，不能写“所有骨架和化学空间完全独立”。

## 7. 明确没有模型预测

本 Notebook 不创建 `model`，也不调用：

```text
model.predict(test_dataset.X)
```

加载 test 的 `X`、样本数和 ID 进行 shape、特征有限值与 ID 交集检查，不等于查看 test 标签。今天的代码不访问 `test_dataset.y`，也不计算 test 指标。

## 8. 输出政策文本

```python
test_policy = (
    "Day 2–28 的开发阶段不使用原 ESOL test 选择模型；"
    "原 test 历史上已暴露，不能重新称为严格未见。"
    "只有方案、指标和代码预先冻结，并建立真正未见的外层数据后，"
    "才进行一次最终评价。"
)

print(test_policy)
```

## 9. 怎样保存个人结果

真正学习时再创建：

```text
learning_outputs/day08_split_protocol/
├── README.md
├── day08_split_protocol.ipynb
├── notes.md
└── results/
    ├── split_summary.csv
    ├── exact_id_overlaps.csv
    └── test_policy.txt
```

只保存实际检查输出，不保存 test 预测或 test 指标。`notes.md` 必须单独写清“精确 ID 检查能发现什么、不能发现什么”。

## 10. 课程教程不是个人证据

[tutorial.ipynb](tutorial.ipynb) 中的预存输出证明供应示例曾顺序执行，不代表你已经检查过自己的环境。复制到个人练习目录、从空内核运行并解释后，才形成个人学习证据。

下一步：[Day 8 练习](03_exercises.md)。
