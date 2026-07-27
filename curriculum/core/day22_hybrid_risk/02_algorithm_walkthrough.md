# Day 22 算法走读：从错误堆叠到 OOF 方案

## 场景

有两个基础模型 A 和 B，以及一个线性第二层模型。目标是在外部验证集上比较单模型、简单平均和 stacking。

## 错误流程

```python
model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)
meta_X = np.column_stack([
    model_a.predict(X_train),
    model_b.predict(X_train),
])
meta_model.fit(meta_X, y_train)
```

问题：`meta_X` 是基础模型对自己训练样本的预测。若基础模型记忆训练数据，第二层得到的输入过度乐观。

## 正确流程：按组逐折生成 OOF

```python
oof_a = np.empty(len(y_train))
oof_b = np.empty(len(y_train))

inner_splits = list(GroupKFold(5).split(
    X_train, y_train, groups=groups_train
))
for fit_idx, hold_idx in inner_splits:
    assert set(groups_train[fit_idx]).isdisjoint(
        set(groups_train[hold_idx])
    )
    fold_a = clone(model_a)
    fold_b = clone(model_b)
    fold_a.fit(X_train[fit_idx], y_train[fit_idx])
    fold_b.fit(X_train[fit_idx], y_train[fit_idx])
    oof_a[hold_idx] = fold_a.predict(X_train[hold_idx])
    oof_b[hold_idx] = fold_b.predict(X_train[hold_idx])

meta_X = np.column_stack([oof_a, oof_b])
meta_model.fit(meta_X, y_train)
```

每个位置只由没有用该位置标签、也没见过该位置所属 group 的折模型填入。

## 外部预测阶段

OOF 只用于训练第二层。预测外部验证时：

1. 用全部外部训练集重新拟合 A、B；
2. 对 `X_valid` 得到两列预测；
3. 把两列交给已经训练好的第二层。

```python
model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)
valid_meta_X = np.column_stack([
    model_a.predict(X_valid),
    model_b.predict(X_valid),
])
valid_prediction = meta_model.predict(valid_meta_X)
```

外部验证标签直到计算最终指标时才出现。

## 覆盖检查

初始化一个计数数组：

```python
coverage = np.zeros(len(y_train), dtype=int)
for fit_idx, hold_idx in inner_splits:
    assert set(groups_train[fit_idx]).isdisjoint(
        set(groups_train[hold_idx])
    )
    coverage[hold_idx] += 1
assert np.all(coverage == 1)
```

还要先断言外部边界：

```python
assert set(groups_train).isdisjoint(set(groups_valid))
```

覆盖、内部组互斥和外部组互斥缺一不可；这些断言仍不能自动证明
预处理无泄漏，基础模型中的缩放、插补和选特征仍需放在 Pipeline 内。

## 预注册模板

```text
主方案：A + B 的 OOF stacking
对照：A、B、简单平均
第二层：Ridge(alpha=1.0)
内部 CV：GroupKFold(5)，group 来源在运行前固定
外部验证：固定 scaffold valid
失败条件：未稳定超过最强单模型/简单平均，或成本无合理收益
test：封存
```

## 今天的停止点

完成流程图、风险清单和预注册后停止。Day 23 才实际运行完整 stacking。设计完成不等于方法有效。
