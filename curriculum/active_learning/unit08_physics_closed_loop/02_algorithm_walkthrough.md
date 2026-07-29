# Unit 08 算法走读：结构化 GP 与审批状态

## 物理均值函数

```python
def physics_mean(X):
    temperature = X[:, 0]
    ratio = X[:, 1]
    return 0.03 * temperature - 2.0 * (ratio - 0.6) ** 2
```

它必须只使用候选输入和事先确认的关系，不能从隐藏池标签反推后再称为先验。

## 残差学习

```python
residual_y = y_labeled - physics_mean(X_labeled)
residual_gp.fit(X_labeled, residual_y)

residual_mean, std = residual_gp.predict(X_pool, return_std=True)
prediction_mean = physics_mean(X_pool) + residual_mean
```

形状保持 `(n_pool,)`。GP 方差来自残差模型；完整不确定性是否还应包括物理参数不确定性，需要更严格模型。

## 成本和约束

```python
score = (prediction_mean + beta * std) / experiment_cost
score = np.where(feasible_mask, score, -np.inf)
```

若成本为 0 会除零，因此成本必须为正并经过数据检查。

## 状态机

```text
proposed
├── approved → running → completed
│                      └→ failed
└── rejected
```

算法只产生 `proposed`。真实系统中，只有 `approved` 才能进入实验队列。

## 错误先验消融

至少比较：

- 普通 GP；
- 正确/合理物理趋势 + 残差 GP；
- 故意错误趋势 + 残差 GP。

如果错误先验在小数据阶段明显降低样本效率，就说明方法需要先验可靠性检查。

教程中的三组候选是三种模型各自提出的“替代方案”，不是要无条件执行的
同一实验批次。若多个模型给出同一候选，真实 campaign 应按永久候选 ID
去重后再提交审核。
