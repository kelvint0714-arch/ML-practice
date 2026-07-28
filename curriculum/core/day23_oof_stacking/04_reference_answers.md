# Day 23 参考答案

## A. 形状题

1. `(902, 2)`；
2. `(902, 1026)`，即两列预测加 1024 列原始特征；
3. 特征数显著增加且尺度/共线性更复杂，第二层更容易过拟合。

## B. 分组覆盖

```python
coverage = np.zeros(len(y_train), dtype=int)
for fit_idx, hold_idx in inner_splits:
    coverage[hold_idx] += 1
    assert set(train_scaffolds[fit_idx]).isdisjoint(
        set(train_scaffolds[hold_idx])
    )
assert np.array_equal(coverage, np.ones(len(y_train), dtype=int))
```

## C. API 审计

- `estimators`：第一层基础模型；
- `final_estimator`：第二层组合模型；
- `cv`：生成 OOF 特征的内部划分；
- `passthrough`：是否把原始 `X` 也传给第二层。

`prefit` 可能让第二层接收基础模型对已见训练样本的预测，违背 OOF 条件。
显式 split 列表把已经使用 `groups=train_scaffolds` 生成的索引交给
`StackingRegressor`；否则普通整数/KFold 配置不会自动知道 scaffold groups。

## D. 代码改错

风险包括：缩放器在内部 CV 外预先拟合；预拟合 MLP 看过全部训练样本；
`cv="prefit"` 产生训练内二层输入；若目标是建立组合基线，只有一个基础模型
也缺少组合意义（Day 24 仅把它作为预注册消融对照）。

正确结构：

```python
mlp = make_pipeline(
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
stack = StackingRegressor(
    estimators=[("tree", tree), ("mlp", mlp)],
    final_estimator=Ridge(),
    cv=inner_splits,
)
stack.fit(X_train, y_train)
```

## E. 结果题

示例中简单平均的当前 RMSE 最低且结构最透明，可作为下一步候选；stack 未超过它。差异来自一次验证，仍需多种子/多划分稳定性与耗时检查。题目数值是教学假设，不是本仓库新实验结果。
