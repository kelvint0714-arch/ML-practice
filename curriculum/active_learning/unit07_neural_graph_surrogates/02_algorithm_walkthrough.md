# Unit 07 算法走读：RF 与 MLP 集成同协议对照

## 建立成员

```python
from sklearn.compose import TransformedTargetRegressor

rf_members = [
    RandomForestRegressor(random_state=seed)
    for seed in seeds
]

mlp_members = [
    make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(24,),
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            ),
            transformer=StandardScaler(),
        ),
    )
    for seed in seeds
]
```

第一个 `StandardScaler` 只用已标注数据缩放 X；
`TransformedTargetRegressor` 内部的第二个 scaler 缩放 y，并把预测变回原单位。
RF 通常不需要缩放。这个流程差异应记录。

训练后检查每个 MLP：

```python
iterations = [
    member.named_steps[
        "transformedtargetregressor"
    ].regressor_.n_iter_
    for member in mlp_members
]
assert max(iterations) < 2000
```

达到 `max_iter` 或出现 `ConvergenceWarning` 时，应调整流程并重新验证，
不能全局隐藏警告后继续比较。

## 统一训练与预测函数

```python
def fit_and_predict(members, X_labeled, y_labeled, X_pool):
    predictions = []
    for member in members:
        member.fit(X_labeled, y_labeled)
        predictions.append(member.predict(X_pool))
    return np.vstack(predictions)
```

输入与输出：

- 输入成员列表、同一已标注集、同一候选池；
- 输出 `(n_members, n_pool)`；
- 函数不接收 `y_pool`。

## 使用同一采集函数

```python
mean = prediction_matrix.mean(axis=0)
std = prediction_matrix.std(axis=0)
ucb = mean + beta * std
```

这样候选差异主要来自代理模型及其分歧，而不是采集公式变化。

## 不能直接得出的结论

若 MLP 本轮选中更好候选，不能立即说明神经网络普遍优于 RF。至少需要多轮、多种子、多个数据集和运行成本。

## 迁移到 GNN

完成 GNN 课程后，可把 `MLPRegressor` 替换为图模型，但接口仍应输出：

```text
candidate_id, prediction_mean, prediction_std
```

采集层不需要知道内部是 RF、MLP 还是 GNN。
