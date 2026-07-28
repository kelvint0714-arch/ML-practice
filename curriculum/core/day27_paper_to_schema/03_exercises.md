# Day 27 练习

> 所有练习都是 schema 设计，不代表真实数据。

## A. 分类题

把下列字段分为身份、配方、工艺/接头、标签、质量/权限：

- 样品编号；
- 固化温度；
- 搭接剪切强度；
- 性能单位；
- 环氧/胺比；
- 基材；
- 实验批次；
- 可否上传 GitHub。

## B. 通用与专用

判断哪些是通用字段类别，哪些更可能是体系专用：

- component_identity；
- test_standard；
- epoxy_equivalent_weight；
- wood_moisture_content；
- replicate_count；
- NCO_content。

## C. 不可拼表

论文 A 报告金属搭接剪切强度 MPa，论文 B 报告木材拉伸剪切强度 N/mm²。即使单位可换算，为什么仍不能直接合并？至少列出四项检查。

## D. YAML 改错

找出风险：

```yaml
status: ready_for_training
adhesive_family: epoxy
target_unit: MPa
may_upload_to_github: true
```

已知化学组尚未确认体系、单位和权限。请改为安全模板。

## E. DOI 边界

用两句话说明 DOI 能证明什么、不能证明什么。

## F. 评审问题

写出发给化学组的 10 个问题，覆盖体系、行定义、标签、配方、工艺、接头、重复、批次、缺失和权限。
