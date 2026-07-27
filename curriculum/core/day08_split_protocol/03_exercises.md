# Day 8.3：数据划分与泄漏练习

先独立回答，再打开 [参考答案](04_reference_answers.md)。

## A. split 职责

将动作分配到允许使用的最早 split：

1. 拟合 Ridge 系数；
2. 拟合 `StandardScaler` 的均值和标准差；
3. 比较预先设定的三个 `alpha`；
4. 判断是否需要继续改模型；
5. 方案完全冻结后做一次最终评价；
6. 训练神经网络权重；
7. 选择早停轮次。

说明 validation 为什么可以“查看指标”却不能参与 `fit`。

## B. 找泄漏

逐段指出问题并改正。

### 情形 1

```python
scaler.fit(np.vstack([X_train, X_valid]))
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
```

### 情形 2

```python
for alpha in [0.01, 0.1, 1, 10]:
    model.fit(X_train, y_train)
    score = rmse(y_test, model.predict(X_test))
```

### 情形 3

```python
selected_features = choose_features(X_valid, y_valid)
model.fit(X_train[:, selected_features], y_train)
```

## C. shape 契约

某 split：

```text
X.shape = (113, 1024)
y.shape = (113, 1)
ids.shape = (112,)
```

1. 将 `y` 整理为单目标一维后是什么 shape？
2. 哪个契约失败？
3. 为什么不能直接删除 X 的最后一行“让它对上”？

## D. 集合交集

```text
train IDs = {A, B, C, D}
valid IDs = {D, E}
test IDs  = {F, A}
```

计算三组交集大小，并说明应否继续训练。

## E. 能力边界

精确字符串 ID 交集为 0。下列结论哪些仍然不能保证？

1. 没有完全相同字符串 ID；
2. 没有同一分子的不同 SMILES 写法；
3. 没有相似骨架；
4. 没有同一批次的相关测量；
5. 三个集合的样本数相同。

## F. test 暴露

回答：

1. 为什么看过 test 后删除输出不能恢复“未见”？
2. 改一个随机种子重新划分，是否自动得到可信的新 test？
3. 若需要新的最终证据，应在什么时候定义外层 holdout？
4. Day 8 可以读取 test 的样本数和 ID 做完整性检查吗？
5. Day 8 可以计算 test RMSE 吗？

## G. 写政策

不用参考本课原句，自己写一份 4–6 条的 test 使用政策。必须包含：

- 预处理 fit 范围；
- 模型选择数据；
- 方案冻结；
- 已暴露 test 的处理；
- 最终评价次数或条件。

## H. 实现检查函数

写函数：

```python
def validate_splits(split_datasets):
    # 返回 split_table 和 overlap_table
    pass
```

要求：

- 检查每个 X 是二维；
- train、validation 的 y 整理成一维；
- train、validation 检查 X/y/ids 行数一致；
- test 只检查 X/ids 行数，函数不得访问 `test_dataset.y`；
- 特征维数一致；
- 三个 split 的 X 全部有限，train、validation 的 y 全部有限；
- 检查三组精确 ID 交集；
- 发现失败时抛出带 split 名称的错误；
- 不训练模型。

## 验收清单

- [ ] 能区分 train、validation、test；
- [ ] 知道 scaler 也会学习；
- [ ] 能发现三种泄漏；
- [ ] 能写 shape/有限值/ID 检查；
- [ ] 能解释精确交集的能力边界；
- [ ] 能诚实说明 test 已暴露；
- [ ] Day 8 没有产生 test 预测或指标。

完成后再核对：[折叠参考答案](04_reference_answers.md)。
