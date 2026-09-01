# Day 27 参考答案

## A

- 标识：sample_id
- 输入：feature_1
- 目标：target_value
- 条件：temperature
- 分组：batch_id
- 元数据：measurement_unit

## B

类型和单位应来自来源文件；未知时写 pending。目标在训练已标注集可见，但在候选池 query 前隐藏。

## C

不能直接合并。至少检查样本定义、目标含义、单位、标注流程、特征可见阶段、分组关系和许可。

## D

两者都使用了目标产生后的信息。query 前使用会把答案泄漏进输入，使离线分数虚高。

## E

明确写 pending 或 unknown，同时记录需要回到哪个来源核实。未知不等于 0、空字符串或允许公开。

## F

~~~python
def may_merge(checks):
    if not checks:
        raise ValueError("checks 不能为空")
    return all(value is True for value in checks.values())
~~~
