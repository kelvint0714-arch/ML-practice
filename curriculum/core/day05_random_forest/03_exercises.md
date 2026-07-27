# Day 5.3：随机森林练习

先完成，再看 [参考答案](04_reference_answers.md)。

## A. Bootstrap 判断

原训练编号 `[0, 1, 2, 3]`。判断下列是否可能是抽取 4 次得到的 Bootstrap 样本，并说明原因：

1. `[0, 0, 2, 3]`
2. `[0, 1, 2, 3]`
3. `[0, 1, 2]`
4. `[4, 1, 2, 3]`
5. `[3, 3, 3, 3]`

## B. 手算平均预测

四棵树对三个验证样本的预测：

| 树 | 样本 A | 样本 B | 样本 C |
|---|---:|---:|---:|
| 1 | 1 | 4 | 7 |
| 2 | 2 | 3 | 8 |
| 3 | 3 | 2 | 9 |
| 4 | 2 | 3 | 8 |

计算森林对 A、B、C 的预测。

## C. 参数配对

把参数与作用配对：

- `n_estimators`
- `max_depth`
- `max_features`
- `bootstrap`
- `random_state`

作用：

1. 是否对训练行做有放回抽样；
2. 每次分裂考虑多少特征；
3. 固定伪随机过程；
4. 森林包含多少棵树；
5. 限制每棵树深度。

## D. 反例解释

回答：

1. 为什么复制 100 棵完全相同的树不会降低它们共同的错误？
2. 为什么“树有差异”仍不能保证森林有效？
3. 为什么 1000 棵树不保证一定优于 100 棵？
4. 随机森林是否完全不会过拟合？

## E. 实现逐树平均检查

已训练的 `forest` 和 `X_valid` 已存在。写代码：

1. 收集每棵树对全部验证样本的预测；
2. 得到 shape 为 `(n_trees, n_valid)` 的数组；
3. 沿树的轴取平均；
4. 与 `forest.predict(X_valid)` 使用 `np.allclose` 核对。

## F. 实现公平对照

写函数比较：

- `DecisionTreeRegressor(max_depth=3)`；
- `RandomForestRegressor(n_estimators=100, max_depth=3)`。

结果必须是长表：

```text
model | split | mae | rmse | r2
```

要求固定种子，不访问 test。

## G. 读假想结果

| model | train RMSE | valid RMSE |
|---|---:|---:|
| one_tree | 0.10 | 1.40 |
| random_forest | 0.35 | 0.95 |

写三句话：

1. 哪个模型训练拟合更强；
2. 哪个模型在这一次验证上更好；
3. 哪些结论不能从表中得出。

## H. 找不公平

```python
tree.fit(X_train_small, y_train_small)
forest.fit(X_train, y_train)

tree_score = rmse(y_valid, tree.predict(X_valid))
forest_score = rmse(y_test, forest.predict(X_test))
```

指出所有无法直接比较的原因。

## 验收清单

- [ ] 能解释有放回抽样；
- [ ] 能手算森林回归平均；
- [ ] 能区分树数量、树深和随机特征；
- [ ] 能从内部树预测复算森林输出；
- [ ] 公平对照使用同一数据和指标；
- [ ] 没有把树数量增加写成必然提升；
- [ ] 没有把重要性写成因果。

完成后再打开：[折叠参考答案](04_reference_answers.md)。
