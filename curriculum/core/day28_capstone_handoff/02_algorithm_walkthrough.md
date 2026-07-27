# Day 28 走读：结构检查与交接包

> 本教程只检查空白模板结构，不调用模型。输出中的行数和非空标签数是工作簿结构事实，不是实验结果。

## 步骤 1：定位仓库和工作簿

```python
from pathlib import Path

def find_repo_root(start=Path.cwd().resolve()):
    for candidate in (start, *start.parents):
        if (candidate / "data" / "adhesive").exists():
            return candidate
    raise RuntimeError("请从 ML-practice 仓库内运行")

repo_root = find_repo_root()
workbook = repo_root / "data/adhesive/templates/粘合剂重要化学性质_数据格式_v0.3.xlsx"
assert workbook.exists()
```

不要写死用户电脑的绝对路径。

## 步骤 2：先不设表头读取

```python
raw_sheets = pd.read_excel(workbook, sheet_name=None, header=None)
print(list(raw_sheets))
```

这一步用于观察工作表和说明行，不把说明文字误当字段。

## 步骤 3：按模板约定读取正式字段

```python
data = pd.read_excel(
    workbook,
    sheet_name="01_核心化学性质",
    header=3,
)
```

`header=3` 表示第 4 行作为列名。若模板版本改变，不能默默沿用；应核对版本和说明。

## 步骤 4：结构断言

```python
required = [
    "样品编号",
    "粘合剂化学体系",
    "实测性能名称",
    "实测性能值",
    "实测性能单位",
]
missing = [name for name in required if name not in data.columns]
assert not missing, f"缺少关键字段: {missing}"
```

断言通过只表示列名存在。

## 步骤 5：安全统计

```python
measured_count = int(data["实测性能值"].notna().sum())
```

若为 0，明确停止建模。即使大于 0，也要先判断说明行、真实样例、单位和权限，不能自动训练。

## 步骤 6：形成评审清单

建立三张表：

1. `field_review`：字段名、含义、类型、单位、必填、负责人、状态；
2. `permission_review`：数据类别、可建模、可共享、可公开、批准人；
3. `modeling_gate`：每项门槛、证据路径、状态、阻塞人。

空白状态用“待确认”，不能用“已通过”作示例默认值。

## 步骤 7：导师交接摘要

```text
已完成：公开 ESOL 方法路线、无泄漏比较教程、draft schema
未完成：真实粘合剂数据建模、实验推荐、项目性能结论
需要决定：体系、目标、行定义、标准、分组、权限
下一里程碑：冻结 v1.0 → 真实样例接口验收 → 首批数据审计
```

## 停止条件

出现任一情况就不建模：

- 标签为空或含义不一致；
- 行定义不清；
- 重复/批次无法分组；
- 权限未确认；
- 只有模板说明或几行接口样例；
- 真实配方可能进入公开 Git 历史。
