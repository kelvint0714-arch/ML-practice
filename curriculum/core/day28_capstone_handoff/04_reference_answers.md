# Day 28 参考答案

> 以下为交接模板答案，不是项目完成证明。

## A. 判断题

1. 错。列存在只通过结构检查；
2. 错。3–5 行用于接口验收；
3. 对；
4. 错。Git 历史、fork、缓存和 clone 可能保留内容；
5. 错。先建立可信表格基线，再按门槛评估 GNN。

## B. 结构检查示例

```python
def inspect_template(path, sheet, header_row, required_columns):
    raw = pd.read_excel(path, sheet_name=None, header=None)
    if sheet not in raw:
        raise KeyError(f"missing sheet: {sheet}")
    data = pd.read_excel(path, sheet_name=sheet, header=header_row)
    missing = [c for c in required_columns if c not in data.columns]
    target_count = (
        int(data["实测性能值"].notna().sum())
        if "实测性能值" in data.columns else None
    )
    return {
        "sheets": list(raw),
        "column_count": len(data.columns),
        "missing_required": missing,
        "row_count": len(data),
        "non_null_target_count": target_count,
    }
```

## C. 门槛项目示例

体系、行定义、首要目标、单位、测试标准、重复关系、批次关系、缺失机制、异常规则、原始版本、建模权限、公开权限。每项都应有负责人和可定位证据。

## D. 权限场景

可以在批准的组内受控空间按合同开展分析；不能上传公开仓库、发给未授权人员或将原始配方用于合同外目的。应确认派生特征、模型参数、汇总图表和论文发表各自的权限。

## E. 阶段摘要示例

> 已完成：公开 ESOL 上的传统模型、MLP、无泄漏 stacking、分歧与池模拟教程，以及粘合剂 draft schema。未完成：尚无获准建模的真实粘合剂数据，未产生真实性能模型或实验候选。当前阻塞为体系、目标、行定义、测试标准、分组与权限待化学组确认。下一步先冻结 v1.0，用获准的少量真实行验收接口，再审计首批数据并建立传统模型基线。

## F. GNN 门槛

需要可靠结构表示、图构建规则、足够且可合法使用的数据、稳定无泄漏划分、可信表格基线和合理算力。粘合剂配方可能由多组分、比例、固化过程、基材和界面共同决定，单一 SMILES/分子图无法完整表达一行样本。

## G. 风险发现

先查模板说明和版本；再核对行 ID、来源与批准人；确认是否为说明行；核对单位/标准/批次；查数据权限记录；只有确认为真实获准记录后才进入受控数据审计。即使是真实 3 行，也仍不可据此训练可靠模型。
