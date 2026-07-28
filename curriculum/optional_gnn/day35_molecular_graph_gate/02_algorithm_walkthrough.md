# Day 35.2：读取实际模板并作出证据化决策

## 1. 输入、动作、输出

| 项目 | 内容 |
|---|---|
| 输入 | 仓库实际 v0.3 `.xlsx` |
| 动作 | 检查 sheet、字段、真实行、标签和结构来源字段 |
| 输出 | 门槛表与当前 No-Go |
| 禁止动作 | 补造 SMILES、配方、性能或调用模型 |

## 2. 自动定位实际工作簿

```python
from pathlib import Path

def find_repo_root(start=Path.cwd().resolve()):
    for candidate in (start, *start.parents):
        workbook = (
            candidate
            / "data"
            / "adhesive"
            / "templates"
            / "粘合剂重要化学性质_数据格式_v0.3.xlsx"
        )
        if workbook.is_file():
            return candidate
    raise FileNotFoundError("没有找到 v0.3 工作簿")

repo_root = find_repo_root()
workbook = (
    repo_root
    / "data"
    / "adhesive"
    / "templates"
    / "粘合剂重要化学性质_数据格式_v0.3.xlsx"
)
```

这样课程源 Notebook 和 `experiments/` 中的个人副本都能找到同一个仓库根目录，不写死个人电脑的绝对路径。

## 3. 正确读取第 4 行字段名

v0.3 的前三行是标题与填写说明，正式列名在第 4 行：

```python
import pandas as pd

raw_data = pd.read_excel(
    workbook,
    sheet_name="01_核心化学性质",
    header=3,
    engine="openpyxl",
)
data = raw_data.dropna(how="all")
```

`header=3` 中的 3 是从 0 开始的行编号，所以代表第 4 行。

## 4. 只报告文件事实

```python
structure_keywords = ("smiles", "sdf", "inchi", "结构文件", "结构来源")
structure_columns = [
    str(column)
    for column in data.columns
    if any(keyword in str(column).lower() for keyword in structure_keywords)
]

label_count = int(data["实测性能值"].notna().sum())
label_protocol_verified = False
structure_traceability_verified = False
```

如果表是 0 行，不能把空单元格当作零值。没有结构列也不能根据“主树脂/基体名称”自行生成结构。
即便以后发现候选列或非空值，也必须人工核查结构内容、来源以及标签的目标、单位和测试方法；因此字段或值的存在本身不能把门槛改成 `True`。

## 5. 门槛表怎样生成

```python
gate_rows = [
    {
        "门槛": "有真实样本",
        "当前证据": f"{len(data)} 行",
        "通过": len(data) > 0,
    },
    {
        "门槛": "有可追溯结构字段",
        "当前证据": f"候选字段：{structure_columns or '未发现'}；内容/来源未审核",
        "通过": structure_traceability_verified,
    },
    {
        "门槛": "有一致实测标签",
        "当前证据": f"{label_count} 个非空标签；一致性未审核",
        "通过": label_protocol_verified,
    },
]
```

样本定义、结构审核、多组分表示、权限、独立配方数量和表格基线不能仅靠空白模板推断，应显式写“待导师/化学组确认”，通过值为 `False`。

## 6. 决策逻辑

中文伪代码：

```text
读取实际模板
如果没有真实行：
    No-Go
否则如果没有可追溯结构：
    No-Go
否则如果没有一致实测标签：
    No-Go
否则如果样本定义/多组分/权限/划分/基线任一未确认：
    补数据后 Go 或继续评审
否则：
    Go
```

自动化检查只能阻止明显越界，不能代替领域判断。最终 Go 必须由导师、化学组和数据负责人共同确认。

## 7. 当前输出应该是什么

在当前仓库版本中，合理输出是：

```text
数据行数：0
结构字段：未发现
非空实测标签：0
决定：No-Go（当前模板为空且没有可追溯结构）
```

如果你的输出不是这样，应先检查是否读错了 sheet/header，或工作簿是否已被未授权修改；不要为了让断言通过而更改空表事实。

## 8. 下一步交接

1. 由导师冻结首个粘合剂体系和主要目标；
2. 化学组定义一行样本及组分关系；
3. 决定结构格式、来源和审核责任；
4. 用 3–5 行获准真实样例验接口；
5. 对首批数据做单位、缺失、重复、批次和权限审计；
6. 先运行表格传统模型基线；
7. 再重新召开 GNN Go/No-Go 评审。

现在运行 [教学 Notebook](tutorial.ipynb)，再独立完成 [练习](03_exercises.md)。
