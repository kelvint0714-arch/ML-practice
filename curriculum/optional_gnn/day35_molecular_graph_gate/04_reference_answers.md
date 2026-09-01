# Day 35 参考答案

## 练习 1

不应。随机边没有任务语义，只会人为引入邻域关系。先使用表格基线；只有存在可解释关系时才考虑图。

## 练习 2

图标签描述整张图，复制后并没有产生真实节点标签，会制造伪监督并改变任务定义。

## 练习 3

模型可能记住共享结构而不是泛化到新对象，测试分数会偏高。应按原始对象、组或时间划分。

## 练习 4

Conditional Go 或 No-Go，取决于这些缺口是否能在建模前补齐；当前不能正式训练并宣称可信比较。

## 练习 5

增加使用相同节点/边信息构造的简单图统计基线、同输入消融，以及去掉边或关键节点特征的 GNN 消融。

## 练习 6

~~~python
def gnn_readiness(gates):
    missing = [name for name, passed in gates.items() if not passed]
    return {
        "decision": "Go" if not missing else "No-Go",
        "missing": missing,
    }
~~~
