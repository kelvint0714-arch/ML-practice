# Day 11 算法推演：从泄漏流程改成 Pipeline

## 1. 极小例子

训练列：

```text
[0, 1, 2, 缺失]
```

验证列：

```text
[100, 110]
```

若训练中位数为 `1`，正确填补器应把训练缺失值填成 `1`。若先把训练与验证合在一起计算统计量，验证区的极端值可能改变均值、尺度，甚至某些填补策略的结果。

即使使用中位数时这个特定例子的填充值没有变化，缩放器的均值和标准差也会被验证值强烈影响。泄漏的定义取决于信息边界，而不是必须观察到更好的分数。

## 2. 错误算法

```python
X_all_imputed = imputer.fit_transform(X_all)
X_all_scaled = scaler.fit_transform(X_all_imputed)
X_train_scaled = X_all_scaled[train_ids]
X_valid_scaled = X_all_scaled[valid_ids]
model.fit(X_train_scaled, y_train)
```

泄漏发生在前两行，因为 `fit_transform(X_all)` 已经接触验证输入。

## 3. 正确算法

```python
pipeline.fit(X_train, y_train)
valid_prediction = pipeline.predict(X_valid)
```

内部数据流为：

```text
fit:
X_train → 学中位数 → 学均值/标准差 → 学 Ridge 系数

predict:
X_valid → 用训练中位数 → 用训练均值/标准差 → 用固定系数预测
```

## 4. 验证缩放器确实只看训练数据

`StandardScaler` 接收的是“已按训练中位数填补”的训练数据，因此手工复核也要先做同样转换：

```python
imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]

train_imputed = imputer.transform(X_train)
expected_mean = train_imputed.mean(axis=0)

assert np.allclose(scaler.mean_, expected_mean)
```

不能直接拿含 `NaN` 的 `X_train.mean(axis=0)` 比较。

## 5. 训练与验证指标

同一条已拟合 Pipeline 可以预测两边：

```python
train_pred = pipeline.predict(X_train)
valid_pred = pipeline.predict(X_valid)
```

训练指标用于观察拟合程度，验证指标用于评价当前冻结流程。二者都不能证明测试性能，更不能用验证数据更新模型。

## 6. 与随机森林的关系

随机森林通常不需要 `StandardScaler`，但这不代表它不需要训练边界：

- 缺失值填补若需要统计量，仍应在折内拟合；
- 特征选择、降维或目标编码仍可能泄漏；
- 分组和标签可获得性仍需人工审查。

可按模型需要构造不同 Pipeline，只要所有模型面对相同样本、相同输入信息和相同验证折。

## 7. 从固定划分到交叉验证

固定划分：

```python
pipeline.fit(X_train, y_train)
pipeline.predict(X_valid)
```

K 折：

```python
cross_validate(pipeline, X_train, y_train, cv=cv)
```

scikit-learn 会 clone Pipeline，使每一折都得到未拟合的新填补器、缩放器和模型。这是 Day 12 调参无泄漏的基础。

## 8. 失败检查

若代码报错，按顺序检查：

1. `X_train.shape[1] == X_valid.shape[1]`；
2. `len(X_train) == len(y_train)`；
3. 缺失值是否由 Pipeline 第一项处理；
4. 是否误对 `X_valid` 调用了 `fit` 或 `fit_transform`；
5. `named_steps` 名称是否与访问时一致；
6. 预测和标签是否同为一维且长度一致。
