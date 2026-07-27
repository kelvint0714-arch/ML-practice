# Day 3.3：Ridge 练习

先独立作答，不要打开 [参考答案](04_reference_answers.md)。

## A. 手算线性预测

模型参数：

```text
intercept = -0.5
weights = [2.0, -1.0, 0.25]
sample = [3.0, 4.0, 8.0]
```

1. 写出完整加权和；
2. 算出预测值；
3. 指出第二个特征对预测的贡献；
4. 如果第二个特征从 4 变成 5，其他不变，预测怎样变化？

## B. shape 题

训练数据有 80 个样本、12 个特征，是单目标回归。

填写：

```text
X_train.shape =
y_train.shape =
model.coef_.shape =
model.predict(X_train).shape =
```

说明为什么 `y_train.shape` 右侧没有第二个数字。

## C. 参数与超参数

把下列项目分类为“训练数据学习的参数”“训练前设置的超参数”或“评价结果”：

- `coef_`
- `intercept_`
- `alpha`
- validation RMSE
- `fit_intercept`

## D. 预测趋势

保持数据不变，只把 `alpha` 从 `0.01` 增大到 `1000`。对以下说法判断“通常可能”“不一定”或“概念错误”，并说明理由：

1. 系数会受到更强压缩；
2. validation RMSE 必然一直下降；
3. train RMSE 可能变大；
4. 模型会变成决策树；
5. 极大 `alpha` 可能造成欠拟合。

## E. 补全算法代码

完成函数，使它返回一个每个 `alpha` 一行的 DataFrame：

```python
def compare_ridge_alphas(X_train, y_train, X_valid, y_valid, alphas):
    records = []
    for alpha in alphas:
        # 创建、训练、预测、评价、记录
        pass
    # 返回 DataFrame
```

要求：

- 不修改输入数组；
- 记录 `alpha`、`coefficient_norm`、`train_rmse`、`valid_rmse`；
- 多特征时使用整个系数向量的 L2 范数；
- 所有结果是有限数；
- 不能访问 test。

## F. 找错题

```python
model = Ridge(alpha=1.0)
print(model.coef_)
model.fit(X_valid, y_valid)
score = mean_squared_error(y_train, model.predict(X_train))
```

至少指出三处与今天协议不符或会报错的问题。

## G. 解释而不是排名

假想结果：

| alpha | \|coefficient\| | train RMSE | valid RMSE |
|---:|---:|---:|---:|
| 0.01 | 8.2 | 0.20 | 1.40 |
| 1 | 3.4 | 0.35 | 0.90 |
| 100 | 0.2 | 1.80 | 1.95 |

写四句话：

1. 描述系数趋势；
2. 解释 `alpha=0.01` 的 train/valid 差距；
3. 解释 `alpha=100`；
4. 写出证据边界。

## 验收清单

- [ ] 能手算多特征线性预测；
- [ ] 能说明 `coef_` 与 `alpha` 的身份不同；
- [ ] 能解释 L2 惩罚；
- [ ] 能同时看 train 与 valid；
- [ ] 能实现控制变量的候选循环；
- [ ] 没有使用 test 选择 `alpha`；
- [ ] 没有把系数直接解释成因果。

完成后再核对：[折叠参考答案](04_reference_answers.md)。
