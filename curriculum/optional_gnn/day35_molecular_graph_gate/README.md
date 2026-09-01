# Day 35：GNN 数据就绪评审

## 学习顺序

1. [概念精讲](01_concepts.md)
2. [算法走读](02_algorithm_walkthrough.md)
3. [教学 Notebook](tutorial.ipynb)
4. [独立练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

能够训练 GCN 或 GIN，不代表任何新数据集都适合 GNN。图对象、关系、标签、划分、样本量和基线必须先满足最低条件。

## 前置条件

- 完成 Day 29–34；
- 理解节点、边、图级标签和 pooling；
- 知道复杂模型不等于更好；
- 接受评审可以得到 No-Go。

## 今日产出

1. 一张七项就绪门槛表；
2. 一个 Go / Conditional Go / No-Go 决策；
3. 一份缺失证据清单；
4. 一份公平比较计划。

## 核心概念

GNN 的输入不是“任何二维数组”，而是具有明确对象和关系语义的图。若边只是为了使用 GNN 而任意构造，模型归纳偏置就没有依据。

## 分步骤任务

1. 定义一张图代表什么；
2. 定义节点、边和特征来源；
3. 核对标签与任务层级；
4. 设计防泄漏划分；
5. 检查样本量与标签覆盖；
6. 冻结简单基线；
7. 根据证据表作出决策。

## 核心代码

~~~python
gate = {
    "graph_object_defined": True,
    "edge_semantics_defined": True,
    "label_protocol_defined": False,
    "leakage_safe_split_defined": False,
    "simple_baseline_available": True,
}

decision = "Go" if all(gate.values()) else "No-Go"
~~~

## 常见错误

- 把普通表格行强行连接成图；
- 图级任务使用节点级标签；
- 近重复图跨训练和测试；
- GNN 获得额外输入，却声称与弱基线公平；
- 样本很少时只报告单个种子；
- 为了得到 Go 删除失败门槛。

## 完成标准

- [ ] 图对象、节点和边语义明确；
- [ ] 标签层级与任务一致；
- [ ] 划分能隔离重复和相关组；
- [ ] 有 Dummy 和简单基线；
- [ ] 决策来自门槛而不是偏好；
- [ ] 能写出补齐证据后的复审条件。

## 自测问题

1. 什么情况下表格数据不应转成图？
2. 图级和节点级标签有什么区别？
3. 为什么随机划分可能产生图数据泄漏？
4. GNN 使用额外输入时怎样公平比较？
5. No-Go 为什么也可能是正确结果？

## 导航

- 上一天：[Day 34 GIN 与图分类](../day34_gin_graph_classification/README.md)
- 课程进度：[唯一学习清单](../../PROGRESS.md)
- 路线入口：[GNN 课程目录](../README.md)
