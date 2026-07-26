# Day 1.6：核心验收

先关闭其他讲解页面，再回答。能够运行代码但无法回答以下问题，仍不算完成 Day 1。

## A. 必须口头解释

1. 算法、训练后的模型和 Python 代码有什么区别？
2. ESOL 的一行数据代表什么？
3. `X` 和 `y` 分别是什么？
4. 为什么 ESOL 是回归，而不是分类？
5. `fit` 和 `predict` 分别做什么？
6. 训练集、验证集、测试集分别有什么用途？
7. 为什么需要保留均值基线？
8. 训练成绩很好、验证成绩很差通常说明什么？

## B. 必须手算

训练答案是：

```text
[1, 2, 3, 6]
```

验证答案是：

```text
[2, 5]
```

完成以下任务：

1. 计算训练答案均值；
2. 写出均值基线的两个验证预测；
3. 写出两个绝对误差；
4. 计算 MAE；
5. 计算 RMSE；
6. 说明 MAE 和 RMSE 哪个方向更好。

答案：

```text
训练均值 = 3
验证预测 = [3, 3]
绝对误差 = [1, 2]
MAE = 1.5
RMSE = √2.5 ≈ 1.581
两者都越小越好
```

## C. 必须读懂四行代码

```python
model = SomeModel()
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
score = metric(y_valid, prediction)
```

逐行标注：

```text
第1行：
第2行：
第3行：
第4行：
```

## D. 必须完成的最小运行

- 从头运行 [最小 Python 代码](04_minimal_code.md) 的五个小代码块；
- 运行前先猜输出；
- 没有把代码粘贴进正式 ESOL Notebook；
- 能解释每个变量保存什么；
- 能说明为什么 Dummy 和手算均值预测相同。

## E. 必须完成的 Notebook 定位

在真实 ESOL Notebook 中指出：

- 数据在哪里加载；
- `X_train` 和 `y_train` 在哪里产生；
- 模型在哪里创建；
- `fit` 在哪里；
- `predict` 在哪里；
- 指标在哪里计算。

今天不要求解释 Git、路径、JSON、CSV 保存、复杂断言和 pandas 结果重排。

## 完成判断

只有 A–E 全部完成，才在 [一步一步学习目录](../../PROGRESS.md) 中勾选 Day 1 的最后一项。

如果想继续练语法和真实代码，可以使用 [扩展练习](exercises_extended.md)，但它不阻挡进入 Day 2。

下一步：[Day 2：评价预测误差](../day02_metrics/README.md)。
