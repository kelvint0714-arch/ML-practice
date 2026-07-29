# Unit 07：神经网络、GNN、PBNN 与 DKL 放在哪里

> 状态：先理清模块，再比较模型。主线不要求安装 GNN。

## 本单元完整学习包

1. [概念精讲](01_concepts.md)
2. [模型角色与代码走读](02_algorithm_walkthrough.md)
3. [可运行教程 Notebook](tutorial.ipynb)
4. [练习](03_exercises.md)
5. [参考答案](04_reference_answers.md)

## 今天为什么学

GNN、随机森林、MLP 和 GP 本身不是“主动学习”。它们负责材料表示、代理预测或不确定性；采集函数再使用这些输出选样。本单元避免把算法名称简单堆在一起。

## 前置条件

- 完成 Unit 06；
- 至少完成核心课程 Day 15–21；
- GNN 代码扩展需完成 Day 29–35，主 Notebook 不要求。

## 今日产出

1. RF、MLP、GNN、PBNN、DKL 的模块位置表；
2. RF 集成与 MLP 集成的同数据候选预测；
3. 已检查收敛状态的 MLP，以及相同 UCB/预算下的选择对照；
4. 一段“复杂模型没有优势时保留负结果”的说明。

## 核心概念

```text
表格/指纹 → RF 或 MLP ensemble → 均值/分歧 → UCB
分子图     → GNN ensemble       → 均值/分歧 → UCB
高维输入   → DNN/GNN embedding  → GP        → UCB/EI
表格输入   → PBNN               → 后验分布  → Uncertainty/UCB
```

模型更复杂不等于主动学习更好，比较必须使用相同初始集、池、预算和采集函数。

## 分步骤任务

1. 用相同表格特征建立 RF 和 MLP 集成。
2. 对同一候选池输出成员预测矩阵。
3. 计算集成均值和分歧。
4. 使用相同 `beta` 计算 UCB。
5. 比较候选排序和运行成本。
6. 标明本 Notebook 没有训练 GNN/PBNN/DKL。
7. 将论文 P04–P06 的模型映射到同一模块图。
8. 说明真实分子图需要哪些数据条件。

## 核心代码骨架

```python
rf_matrix = ensemble_predict(rf_members, X_pool)
mlp_matrix = ensemble_predict(mlp_members, X_pool)

rf_ucb = rf_matrix.mean(0) + beta * rf_matrix.std(0)
mlp_ucb = mlp_matrix.mean(0) + beta * mlp_matrix.std(0)
```

## 常见错误

| 错误 | 后果 | 正确处理 |
|---|---|---|
| 把 GNN 称为采集函数 | 模块混淆 | GNN 是表示/代理 |
| 不同模型用不同特征后直接归因 | 因素混杂 | 分开表示与模型消融 |
| 小数据从零训练大 GNN | 高方差和过拟合 | 先 RF/GP 或预训练表示 |
| 隐藏 MLP 收敛告警 | 未训练完成却参与比较 | 缩放 X/y 并检查迭代次数 |
| 只报告复杂模型最好一次 | 选择性报告 | 多种子同预算 |

## 完成标准

- [ ] 能指出每种算法位于哪个模块；
- [ ] RF 与 MLP 使用相同数据和 UCB；
- [ ] MLP 没有达到迭代上限，且未屏蔽训练失败；
- [ ] 分歧来源和限制写清；
- [ ] 没有声称 Notebook 已实现 GNN/PBNN/DKL；
- [ ] 明确记录单轮不能证明优势；复杂模型无优势时保留结果。

## 自测问题

1. GNN 在主动学习中可以承担哪两种角色？
2. PBNN 与普通 MLP 的关键差别是什么？
3. DKL 为什么是“神经网络 + GP”？
4. DP-GEN 中多个神经网络分歧用于什么？
5. 什么时候不应使用 GNN？

## 上一单元 / 下一单元

- 上一单元：[Unit 06](../unit06_batch_diversity_constraints/README.md)
- GNN 先修：[Day 29–35 路线](../../optional_gnn/README.md)
- 完成验收后：[返回唯一学习目录](../../PROGRESS.md)
- 下一单元：[Unit 08：物理先验与真实闭环](../unit08_physics_closed_loop/README.md)
