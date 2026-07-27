# Day 25 算法走读：生成、排序和诊断分歧

## 步骤 1：冻结集成成员

```python
SEEDS = [11, 22, 33, 44, 55]
```

成员共享数据、预处理、超参数；唯一计划内差异是随机种子。

## 步骤 2：先在原 train 内建立 scaffold 诊断 holdout

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

fit_idx, diagnostic_idx, unused_idx = dc.splits.ScaffoldSplitter().split(
    train_dataset,
    frac_train=0.8,
    frac_valid=0.2,
    frac_test=0.0,
)
assert len(unused_idx) == 0

scaffold_groups = np.asarray([
    MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=False
    )
    for smiles in train_ids
])
assert set(scaffold_groups[fit_idx]).isdisjoint(
    set(scaffold_groups[diagnostic_idx])
)
```

DeepChem 在原 train 内做确定性的 scaffold split；随后再用标准
Bemis–Murcko 字符串独立复核组互斥。无环分子的空 scaffold 不能按每条
SMILES 拆成“独立组”，否则会虚假通过检查。外部 valid 在 Day 25 完全不读取。

## 步骤 3：逐成员拟合

```python
member_predictions = []
for seed in SEEDS:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=120,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        ),
    )
    model.fit(X_train[fit_idx], y_train[fit_idx])
    member_predictions.append(model.predict(X_train[diagnostic_idx]))
```

所有成员只使用内部拟合部分。

## 步骤 4：堆叠并检查形状

```python
prediction_matrix = np.vstack(member_predictions)
assert prediction_matrix.shape == (len(SEEDS), len(diagnostic_idx))
mean_prediction = prediction_matrix.mean(axis=0)
disagreement = prediction_matrix.std(axis=0, ddof=0)
```

若使用 `column_stack`，矩阵方向会相反，需要相应调整轴；课程统一用 `vstack`。

## 步骤 5：建立逐样本表

```python
diagnosis = pd.DataFrame({
    "sample_id": train_ids[diagnostic_idx],
    "prediction_mean": mean_prediction,
    "disagreement": disagreement,
    "absolute_error": np.abs(y_train[diagnostic_idx] - mean_prediction),
})
```

候选阶段不能创建 `absolute_error`，因为真实标签未知。它只在预先切出的内部诊断 holdout 上用于一次性诊断。

## 步骤 6：稳健地分组检查

```python
rank_fraction = diagnosis["disagreement"].rank(method="first", pct=True)
diagnosis["group"] = pd.cut(
    rank_fraction,
    bins=[0.0, 1 / 3, 2 / 3, 1.0],
    labels=["low", "middle", "high"],
    include_lowest=True,
)
group_summary = diagnosis.groupby("group", observed=True).agg(
    n=("sample_id", "size"),
    mean_disagreement=("disagreement", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
)

from scipy.stats import spearmanr

spearman_rho, spearman_p = spearmanr(
    diagnosis["disagreement"],
    diagnosis["absolute_error"],
)
```

如果高分歧组误差没有更大，应如实保留。主动学习仍需随机对照。

先排名再分箱，即使多个样本的分歧完全相同，也不会因 `qcut` 合并分位点而产生标签数量错误。
Spearman 相关只描述分歧排序与误差排序在这一次内部 holdout 上是否同向；
`p` 值也不把启发式分歧升级为校准区间或策略有效证据。

## 步骤 7：排序

```python
ranked = diagnosis.sort_values(
    ["disagreement", "sample_id"],
    ascending=[False, True],
)
```

第二排序键使并列结果确定性更强。

## 结果解释模板

> 五成员随机森林在 ESOL 原 train 的 scaffold 隔离诊断 holdout 上，分歧范围为【实际值】。高/低分歧组的平均绝对误差为【实际值】/【实际值】。外部 valid 未参与本日规则诊断；该标准差未做覆盖率校准，仅用于启发式排序，不能称为预测区间。

## 失败检查

- 成员使用不同特征或数据清洗；
- 在分歧计算或排序时使用 `y_diagnostic`；
- 把绝对误差放入采集函数；
- 用内部诊断结果反复改同一轮种子集合；
- 把高分歧候选自动发送为实验任务。
