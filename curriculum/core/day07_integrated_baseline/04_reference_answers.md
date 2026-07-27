# Day 7.4：参考答案

> 本页不能替代真实 Notebook 的 Restart + Run All。静态文件中的预存输出也不是你的重跑证据。

<details>
<summary>A. 代码分类</summary>

| 动作 | 主要分类 |
|---|---|
| `fit` | 算法核心 |
| 查找仓库根目录 | 工程支持 |
| 构造 ECFP | 数据准备 |
| `predict` | 算法核心 |
| 记录版本 | 工程支持 |
| 检查 ID 交集 | 工程支持/数据检查 |
| RMSE | 算法评价 |
| 保存 JSON | 工程支持 |
| validation 排序 | 结果解释/模型开发 |
| 断言无 test | 工程验收 |

</details>

<details>
<summary>B. 五个核心动作</summary>

当前参考 Notebook：

| 动作 | 代码单元 | 关键位置 |
|---|---:|---|
| 形成 X/y | 4 | `X_train`、`y_train`、`X_valid`、`y_valid` |
| 创建模型 | 6 | `models = {...}` |
| fit | 7 | `model.fit(X_train, y_train)` |
| predict | 7 | `model.predict(X_split)` |
| metric | 5 和 7 | `regression_metrics(...)` |

数据加载在代码单元 2，数据检查在代码单元 3。编号指代码单元，不把 Markdown 单元计入。

</details>

<details>
<summary>C. 运行状态</summary>

1. 不能：只是查看保存状态；
2. 不能：可能依赖旧内核变量；
3. 可以：满足从空状态顺序完整执行的核心条件；
4. 不能：第 4 格失败，后方文字可能是旧输出；
5. 不能直接称为完整复现：执行号不连续表明运行顺序或重复执行发生变化，需要重新 Restart + Run All。

</details>

<details>
<summary>D. 输出类型</summary>

warning 不一定失败，应判断它是否影响当前执行路径。traceback 表示当前单元中断，后方旧输出与当前状态不一致。`AssertionError` 是主动验收条件失败，应停止调查；普通 `print` 只展示信息。

DeepChem 缺少某些可选深度学习后端的 warning 不一定阻止 ECFP＋sklearn 路径，但必须确认核心导入、特征化和模型运行确实成功。

</details>

<details>
<summary>E. 结果契约</summary>

5 个模型 × 2 个 split = 10 行。`split` 唯一值应为 `train`、`valid`，不出现 test，因为 test 不参与本轮开发评价。

统一列才能公平过滤、排序和审计。CSV 保存逐行指标，便于程序读取；JSON 保存数据配置、参数、版本、Git 状态和 test 政策。两者回答的问题不同。

</details>

<details>
<summary>F. 证据边界改写</summary>

1. 在当前一次固定 ESOL scaffold validation 和五个固定候选中，随机森林的 validation RMSE 最低；不能外推到所有分子任务。
2. 不限深树训练表现很强，但 train/valid 差距明显，表现出过拟合信号，不能只按训练 R² 选它。
3. 当前结果只来自公开 ESOL 水溶解度练习，不能作为粘合剂强度证据。
4. 旧实验已暴露原 test；重新运行不会消除开发者已经获得的信息，因此不能重新称为严格未见。

</details>

<details>
<summary>G. 静态审计示例</summary>

```python
import csv
import json
from pathlib import Path

repo_root = Path.cwd()
while not (repo_root / ".git").exists():
    if repo_root.parent == repo_root:
        raise RuntimeError("not inside repository")
    repo_root = repo_root.parent

experiment = repo_root / "experiments" / "esol" / "day01_baseline"
notebook = json.loads(
    (experiment / "esol_baseline.ipynb").read_text(encoding="utf-8")
)
code_cells = [
    cell for cell in notebook["cells"]
    if cell["cell_type"] == "code"
]
assert len(code_cells) == 10
assert [cell["execution_count"] for cell in code_cells] == list(range(1, 11))
assert not any(
    output.get("output_type") == "error"
    for cell in code_cells
    for output in cell.get("outputs", [])
)

with (experiment / "results" / "baseline_metrics.csv").open(
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

config = json.loads(
    (experiment / "results" / "run_config.json").read_text(encoding="utf-8")
)
assert len(rows) == 10
assert {row["split"] for row in rows} == {"train", "valid"}
valid_rows = [row for row in rows if row["split"] == "valid"]
best = min(valid_rows, key=lambda row: float(row["rmse_logS"]))
assert best["model"] == config["best_model_by_validation_rmse"]
```

静态审计只能证明当前文件内部一致。它无法证明这些输出由你刚才的环境和操作产生；这仍需从空内核实际重跑记录支持。

</details>

核对后返回 [Day 7 任务卡](README.md)，完成真实重跑，而不是停在答案页。
