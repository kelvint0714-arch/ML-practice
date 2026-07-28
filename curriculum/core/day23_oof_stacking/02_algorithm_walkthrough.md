# Day 23 算法走读：实现并审计 StackingRegressor

## 步骤 1：冻结基础模型

```python
tree = RandomForestRegressor(**day07_random_forest_params)

mlp = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(32,),
        alpha=0.001,
        early_stopping=True,
        max_iter=300,
        n_iter_no_change=10,
        random_state=42,
    ),
)
```

预处理放在基础模型 Pipeline 内，因此内部每一折都会重新拟合自己的插补器/缩放器。

## 步骤 2：显式创建 scaffold groups 与 GroupKFold

```python
train_scaffolds = np.array([murcko_group(s) for s in train_ids])
inner_splits = list(
    GroupKFold(n_splits=5).split(
        X_train, y_train, groups=train_scaffolds
    )
)
for fit_idx, hold_idx in inner_splits:
    assert set(train_scaffolds[fit_idx]).isdisjoint(
        set(train_scaffolds[hold_idx])
    )
```

`GroupKFold` 不随机打散 group。这里保存显式 split 列表，是因为
`StackingRegressor` 需要直接接收可迭代划分，而不是在 `fit()` 中另传 groups。

## 步骤 3：创建 stack

```python
stack = StackingRegressor(
    estimators=[("tree", tree), ("mlp", mlp)],
    final_estimator=Ridge(alpha=1.0),
    cv=inner_splits,
    passthrough=False,
    n_jobs=1,
)
```

`passthrough=False` 让第二层只看到两列基础预测，保持最小实现。

## 步骤 4：只拟合外部训练并报告警告

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    stack.fit(X_train, y_train)
stack_pred = stack.predict(X_valid)
```

检查调用路径中没有 `X_valid` 或 `y_valid` 进入 `fit()`。
收敛警告应统计并展示，不能用全局过滤器隐藏。

## 步骤 5：独立拟合两个单模型

为了公平得到单模型与平均预测，使用 `clone()`：

```python
tree_single = clone(tree).fit(X_train, y_train)
mlp_single = clone(mlp).fit(X_train, y_train)
tree_pred = tree_single.predict(X_valid)
mlp_pred = mlp_single.predict(X_valid)
mean_pred = (tree_pred + mlp_pred) / 2
```

不要直接依赖 stack 内部对象的非公开实现细节。

## 步骤 6：同表比较

至少四行：

- tree only；
- MLP only；
- simple mean；
- OOF stacking。

记录 RMSE、MAE、R²、耗时和模型结构。

## 步骤 7：审计

程序检查：

```python
assert stack.cv != "prefit"
assert stack.passthrough is False
```

人工检查：

- 外部 valid 是否只在最终评价出现；
- 每个基础模型是否含完整预处理；
- 内部折是否符合样本独立性假设；
- 外部 train/valid 和每个内部折的 scaffold 集合是否互斥；
- 是否保留简单平均和最强单模型；
- 是否因一次结果临时改变结构。

## 解释模板

> 在当前 ESOL 外部验证上，OOF stacking 相对最强单模型的 RMSE 差为【实际值】，相对简单平均为【实际值】。这是一轮固定协议观察；是否保留复杂结构还需多种子/多划分消融。
