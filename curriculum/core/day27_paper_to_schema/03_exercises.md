# Day 27 练习

先独立完成，再查看答案。

## A. 字段角色

把 sample_id、temperature、target_value、batch_id、measurement_unit 和 feature_1 分为标识、输入、目标、条件、分组或元数据。

## B. 字段定义

为 feature_1 和 target_value 写出 dtype、unit、missing_rule 和 visibility。

## C. 合并判断

来源 A 和 B 的目标名称相同，但样本定义、标注流程和单位未知。能否直接合并？写出至少五项检查。

## D. 泄漏判断

为什么“任务完成后的质量等级”或“由目标值分箱得到的类别”不能作为 query 前输入？

## E. 未知信息

单位、许可和缺失含义都未说明时，安全的数据字典应怎样写？

## F. Python

实现一个函数：只有所有合并检查都为 True 时返回 True，并拒绝空检查表。
