# Day 11 练习参考答案

## A. 判断题

1. 错。缩放器的均值和尺度已经看过验证输入，验证不再独立。
2. 错。`predict` 使用训练时保存的填充值、均值与尺度，不重新拟合。
3. 错。填补、特征选择、目标编码、分组错误和未来信息都可能造成泄漏。
4. 对。前提是把原始数据和整条 Pipeline 一起交给交叉验证工具。
5. 错。现实可获得性属于问题与数据审查，需要研究者判断。

## B. 流程题

6. 合格答案：

```text
train + valid
→ * fit imputer
→ * fit scaler
→ 划分
→ model.fit(train)
```

星号步骤已接触验证信息。

7. 正确流程：

```text
fit:     X_train → imputer.fit/transform → scaler.fit/transform → ridge.fit
predict: X_valid → imputer.transform     → scaler.transform     → ridge.predict
```

8. 填补器保存每列填充值；缩放器保存每列均值和尺度；Ridge 保存回归系数和截距。
9. 验证集职责是模拟当前流程面对未见输入。若其分布先进入训练变换，训练流程已经针对未来输入做了适配。

## C. 代码题

10. 新增列后要保证训练与验证列数相同，且至少有训练中的非缺失值可估计填充值。
11. 三个数组通常都与特征数一致；Ridge 单输出回归的 `coef_` 通常是一维、长度为特征数。
12. 示例：

```python
imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]
train_imputed = imputer.transform(X_train)
assert np.allclose(scaler.mean_, train_imputed.mean(axis=0))
```

13. 形状只检查数组维度，不检查坐标系来源。训练和验证分别拟合缩放器，会把两者放在不同统计坐标系中。
14. `cross_validate` 会 clone 估计器。可检查原对象的 Ridge 在运行后仍没有 `coef_`：

```python
assert not hasattr(pipeline.named_steps["ridge"], "coef_")
```

## D. 研究边界题

15. 通常不合理。如果预测时该信息尚不存在，它就是未来信息；Pipeline 不能把不可用特征变得可用。
16. 不能。泄漏由信息边界定义，不由这一次成绩方向定义。
17. 测试集用于方案冻结后的最终检查；今天仍在学习和修改流程；现在查看会使后续选择受到测试信息影响。
18. 例如同一配方的重复样、同一制备批次或同一原料批次应作为组留在同一侧。Pipeline 不会自动识别这些关系。
