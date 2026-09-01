# Day 27.2：来源到数据字典的算法走读

## 输入、变换、输出

| 项目 | 内容 |
|---|---|
| 输入 | 一篇论文或数据卡中的任务描述 |
| 变换 | 拆分任务、样本、字段、单位、可见性和许可 |
| 输出 | 来源表、字段表、合并检查表 |

## 步骤 1：记录来源

至少记录 source_id、title、task_type、sample_definition、target_definition、data_access 和 license_status。unknown 比猜测更安全。

## 步骤 2：建立字段表

每个字段一行，至少包含 field、role、dtype、unit、source、missing_rule 和 visible_during_query。

## 步骤 3：检查标签权限

如果字段由目标直接计算、在 query 后才产生、或只用于最终评价，应标记为不可用于候选选择。

## 步骤 4：建立合并矩阵

~~~python
merge_checks = {
    "same_sample_definition": False,
    "same_target_meaning": False,
    "unit_compatible": False,
    "protocol_compatible": False,
    "license_checked": False,
}

may_merge = all(merge_checks.values())
~~~

只有全部关键检查为真时，may_merge 才能为真。

## 步骤 5：输出缺失信息

把所有 pending 字段集中成清单。下一步是查原文、补数据卡或缩小任务，不是编造默认值。

## 验收

- 来源和数据行没有混淆；
- 字段角色与可见性明确；
- 合并结论来自检查项；
- 未获得的信息保持 pending；
- 本节没有模型训练。
