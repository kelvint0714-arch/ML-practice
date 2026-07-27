# Day 9 算法推演：让每个样本恰好验证一次

## 1. 先用 10 个编号理解轮换

有 10 个训练区域样本，编号为 `0` 到 `9`。为了便于手算，先使用 5 折且不打乱：

```text
折1验证：[0, 1]    训练：[2, 3, 4, 5, 6, 7, 8, 9]
折2验证：[2, 3]    训练：[0, 1, 4, 5, 6, 7, 8, 9]
折3验证：[4, 5]    训练：[0, 1, 2, 3, 6, 7, 8, 9]
折4验证：[6, 7]    训练：[0, 1, 2, 3, 4, 5, 8, 9]
折5验证：[8, 9]    训练：[0, 1, 2, 3, 4, 5, 6, 7]
```

检查：

- 每一轮训练 8 行、验证 2 行；
- 每个编号恰好出现在一组验证索引中；
- 同一轮的训练索引与验证索引交集为空。

## 2. 写成不依赖模型的伪代码

```text
输入：训练区域 X、y，折生成器，未拟合模型配置
建立空的逐折记录

对每一组 fit_indices、valid_indices：
    新建一个模型
    用 X[fit_indices]、y[fit_indices] 拟合
    对 X[valid_indices] 预测
    计算这一折的指标
    保存折号、样本数、指标

输出：逐折表、均值、标准差
```

索引由折生成器决定，模型只消费索引。这样可以单独检查“划分是否正确”和“模型是否正确”。

## 3. 手算一折 RMSE

某一折有两个验证样本：

| 样本 | 真实值 | 预测值 | 误差 | 平方误差 |
|---:|---:|---:|---:|---:|
| A | 1.0 | 1.4 | 0.4 | 0.16 |
| B | 3.0 | 2.2 | -0.8 | 0.64 |

因此：

```text
MSE  = (0.16 + 0.64) / 2 = 0.40
RMSE = sqrt(0.40) ≈ 0.632
```

这一数值只属于当前折。五折结束后，才对五个“折 RMSE”求均值和标准差。不要把五折所有预测误差先混在一起，再假装它一定等于“折 RMSE 的平均值”；当折大小不同，两种聚合口径可能不同。

## 4. 映射到 scikit-learn

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_number, (fit_ids, valid_ids) in enumerate(cv.split(X), start=1):
    model = DecisionTreeRegressor(max_depth=2, random_state=42)
    model.fit(X[fit_ids], y[fit_ids])
    valid_pred = model.predict(X[valid_ids])
```

逐行含义：

1. `KFold(...)` 只定义索引规则，不训练模型；
2. `cv.split(X)` 依次给出两组整数索引；
3. 循环体内创建模型，保证折间状态隔离；
4. `fit()` 只接收 `fit_ids`；
5. `predict()` 只接收当前 `valid_ids`。

## 5. 加三条结构检查

```python
assert set(fit_ids).isdisjoint(valid_ids)
assert len(fit_ids) + len(valid_ids) == len(X)
validation_counts[valid_ids] += 1
```

循环结束后：

```python
assert np.all(validation_counts == 1)
```

这些断言不会证明研究设计一定合理，但能发现索引重叠、漏样本或重复验证等实现错误。

## 6. 为什么要先固定协议

如果一边看逐折成绩，一边改变树深、折数、是否打乱和随机种子，你将无法判断差异来自哪里。本日协议只允许：

- 固定一个模型；
- 固定 5 折；
- 固定随机种子；
- 观察折间差异。

调参会在 Day 12 单独学习。

## 7. 从教学小数据迁移到 ESOL

迁移时仍需保留同样的循环，但要额外记录：

```text
数据：ESOL / Delaney
输入：1024 维 ECFP
标签：原始 logS
数据区域：只使用课程允许的 train 区域
折方式：普通随机 KFold（若确实如此）
缓存：仓库 .cache/deepchem
```

如果研究问题是新骨架泛化，应改用按骨架分组的折，而不是把普通 `KFold` 改名为 scaffold 交叉验证。
