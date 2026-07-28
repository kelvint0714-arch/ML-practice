# Day 27 参考答案

> 以下均为教学模板答案，不是粘合剂实验记录。

## A. 分类题

- 身份：样品编号；
- 配方：环氧/胺比；
- 工艺/接头：固化温度、基材；
- 标签：搭接剪切强度、性能单位；
- 质量/权限：实验批次、可否上传 GitHub。

## B. 通用与专用

通用类别：`component_identity, test_standard, replicate_count`。更可能专用：`epoxy_equivalent_weight, wood_moisture_content, NCO_content`。是否最终采用仍由研究体系决定。

## C. 不可拼表

还需检查测试标准、试样几何、加载模式、基材、表面处理、温湿度、失效模式、行定义和统计汇总方式。单位可换算只解决量纲表达，不保证标签物理意义一致。

## D. 安全模板

```yaml
status: example_template_not_data
adhesive_family: 待化学组确认
target_unit: 待确认
permissions:
  may_use_for_modeling: 待确认
  may_upload_to_github: false
```

## E. DOI 边界

DOI 能唯一追溯到作为字段依据的论文。它不能证明已取得原始数据、许可允许使用、论文已复现，或论文结果属于本组。

## F. 评审问题示例

1. 首个研究体系是什么？
2. 每行代表配方、试样还是重复？
3. 首要预测性能、单位和测试标准是什么？
4. 组分身份如何编码？
5. 用量采用质量份、质量分数还是摩尔比？
6. 固化温度/时间是否完整记录？
7. 基材、表面处理和胶层厚度是否可追溯？
8. 重复试样如何关联？
9. 批次和缺失原因如何记录？
10. 哪些字段可建模、共享或公开？
