# Unit 05 算法走读：怎样组织多种子结果

## 长表格式

发现任务每一行只代表“一个策略、一个种子、一个轮次”。第 0 轮记录
初始状态；第 1 轮起还必须保存 query 证据：

| strategy | seed | round | candidate_id | prediction_mean | prediction_std | score | observed_y | model_version |
|---|---:|---:|---|---:|---:|---:|---:|---|
| random | 11 | 1 | D07 | 0.51 | 0.32 | NA | 0.44 | fixed_gp_rbf_v1 |
| ucb | 11 | 1 | D18 | 0.63 | 0.71 | 1.48 | 1.02 | fixed_gp_rbf_v1 |

同一 seed 的第 0 轮在所有策略中必须相同，因为初始集相同。

## 共享初始集

```python
initial_rng = np.random.default_rng(run_seed)
initial_indices = initial_rng.choice(
    n_candidates, size=n_initial, replace=False
)

initial_values = oracle(initial_indices)
random_result = run_strategy(
    "random",
    initial_indices.copy(),
    initial_values.copy(),
    run_seed,
    oracle,
)
```

传入函数时应复制索引，避免第一个策略修改列表影响后一个策略。

## 汇总

```python
summary = runs.groupby(["strategy", "round"], as_index=False).agg(
    regret_mean=("simple_regret", "mean"),
    regret_std=("simple_regret", "std"),
)
```

- `groupby` 按策略和轮次分组；
- `agg` 为每组计算多个统计量；
- `as_index=False` 保留普通列，便于画图。

## 预算检查

```python
budget_check = runs.groupby(
    ["strategy", "run_seed"]
)["n_labeled"].max()
assert budget_check.nunique() == 1
```

这只检查最终标签数一致，还要检查 batch size 和轮次数。

## 两种目标不能混用指标

```text
材料发现：best-so-far、simple regret、top-k recovery
全局模型学习：固定测试集 MAE/RMSE/R²、不确定性校准
```

模型学习轨迹可以每轮调用 `evaluator(model)` 记录测试 RMSE，但采集函数不能
接收该返回值或测试标签。Notebook 以同一 GP 比较 Random 与最大不确定性，
避免把“换模型”和“换采集函数”混在一起。

## 结论

不要根据均值的一点小差异直接宣布胜负。还要看波动、曲线、一致性以及差异是否在多个数据集存在。
