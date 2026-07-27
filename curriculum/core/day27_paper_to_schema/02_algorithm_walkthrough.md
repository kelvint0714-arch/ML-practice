# Day 27 文档走读：逐篇映射到 schema

> 本流程只创建**文献字段映射示例**。任何 `example_*`、`待确认` 或空值都不得用于模型训练。

## 步骤 1：建立证据表

每篇论文一行，至少记录：

```text
paper_id
doi
adhesive_system
input_formulation
input_process
substrate_or_joint
target_property
test_context
data_availability_checked
notes
```

最后一列明确区分“论文提到”与“本组已取得数据”。

## 步骤 2：拆分输入、条件和标签

不要把所有词放进一个“features”单元格：

- 配方：组分身份、分子量、比例或用量；
- 工艺：固化温度/时间、预处理；
- 接头：基材、表面、胶层；
- 标签：性能名、数值、单位、标准、温度；
- 质量：重复、批次、误差与来源。

## 步骤 3：标记候选字段状态

```text
common_candidate：跨体系可保留的类别
system_specific：只对某体系适用
pending_chemistry：含义或可用性待确认
exclude_for_now：当前范围不使用
```

状态来自项目范围，不来自模型喜欢什么。

## 步骤 4：建立不可拼表记录

为每对来源回答：

1. 目标物理量是否相同？
2. 单位能否合法转换？
3. 测试标准和环境是否可比？
4. 行定义是否一致？
5. 材料体系是否在共同研究范围？
6. 数据许可是否允许使用？

任一关键项未确认时，默认不合并。

## 步骤 5：形成 draft schema

```yaml
schema_version: draft-v1.0
status: example_template_not_data
research_scope:
  adhesive_family: 待化学组确认
  row_definition: 待确认
target:
  property_name: 待确认
  unit: 待确认
  test_standard: 待确认
permissions:
  may_use_for_modeling: 待确认
  may_upload_to_github: false
```

显著的 `status` 防止模板被误读为数据集。

## 步骤 6：化学组评审

优先问：

- 首个材料体系和首要性能是什么？
- 一行代表什么？
- 哪些配方/工艺信息实际可提供？
- 测试标准和单位是否统一？
- 重复与批次如何追溯？
- 哪些数据可建模、组内共享或公开？

## 验收

- 每个字段能追溯到业务需要或文献依据；
- 专用字段没有强加给不相关体系；
- 未声称 DOI 等于开放数据；
- 未从不同论文抄数值拼接；
- 本日没有调用任何模型。
