# Day 6.3：Gradient Boosting 练习

请先完成，再展开 [参考答案](04_reference_answers.md)。

## A. 一轮修正手算

给定：

```text
y_true = [0, 4]
current_prediction = [2, 2]
tree_correction = [-1.5, 1.0]
learning_rate = 0.2
```

1. 当前残差是多少？
2. 缩放后的修正是多少？
3. 新预测是多少？
4. 新残差是多少？
5. 哪个样本仍然预测偏低？

## B. Bagging 与 Boosting

填写：

| 问题 | 随机森林 | Gradient Boosting |
|---|---|---|
| 树是否相对独立 |  |  |
| 每棵树主要学习什么 |  |  |
| 最终怎样汇总 |  |  |
| `n_estimators` 的作用 |  |  |

## C. 概念判断

说明下列说法为什么正确或错误：

1. `learning_rate=0.1` 表示模型准确率是 10%；
2. 学习率更小通常可能需要更多树；
3. 后一棵树可以在完全不考虑前面模型的情况下独立训练；
4. 训练误差不断下降就证明 validation 一定改善；
5. Gradient Boosting 能自动修复数据泄漏。

## D. staged prediction

已训练的 `model` 有 50 个阶段。写代码：

1. 收集 validation 的逐阶段预测；
2. 验证共有 50 个；
3. 计算每阶段 validation RMSE；
4. 只显示阶段 1、10、25、50；
5. 核对最后阶段等于 `model.predict(X_valid)`。

## E. 控制变量

设计一个只观察学习率的实验。写出：

- 固定的三个学习率；
- 必须保持不变的至少五项；
- 记录的列；
- 为什么不能查看 test。

## F. 读假想学习过程

| stage | train RMSE | valid RMSE |
|---:|---:|---:|
| 1 | 2.0 | 2.2 |
| 20 | 0.9 | 1.2 |
| 50 | 0.4 | 1.0 |
| 100 | 0.1 | 1.4 |

回答：

1. train 怎样变化？
2. validation 怎样变化？
3. 哪一段出现过拟合信号？
4. 能否用 test 来选第 50 阶段？
5. 今天是否已经要求实现正式早停？

## G. 找错题

```python
for learning_rate, trees in [(0.01, 500), (0.1, 100), (0.5, 20)]:
    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=trees,
    )
    model.fit(X_train, y_train)
```

如果实验问题是“学习率单独造成什么变化”，这段设计为什么回答不了？怎样修复？

## 验收清单

- [ ] 能手算残差与缩放修正；
- [ ] 能解释顺序依赖；
- [ ] 能区分随机森林和 Boosting 的汇总方式；
- [ ] 能使用 `staged_predict`；
- [ ] 能解读 train/valid 随阶段变化；
- [ ] 知道今天只观察，不用 test 正式选阶段；
- [ ] 没有把 Gradient Boosting 等同于 XGBoost。

完成后再核对：[折叠参考答案](04_reference_answers.md)。
