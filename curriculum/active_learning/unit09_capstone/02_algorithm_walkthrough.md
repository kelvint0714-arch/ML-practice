# Unit 09 算法走读：研究包怎样组织

## 配置

```python
config = {
    "objective": "maximize",
    "n_initial": 4,
    "n_rounds": 10,
    "batch_size": 1,
    "run_seeds": [11, 22, 33, 44, 55],
    "strategies": ["random", "greedy", "ucb", "ei"],
    "beta": 1.5,
    "xi": 0.01,
}
```

配置应在运行前冻结。修改配置必须产生新的实验编号或版本记录。

## 实验矩阵

```python
records = []
for run_seed in config["run_seeds"]:
    initial_indices = make_initial_set(run_seed)
    for strategy in config["strategies"]:
        strategy_records = run_strategy(
            strategy=strategy,
            initial_indices=initial_indices.copy(),
            run_seed=run_seed,
            config=config,
        )
        records.extend(strategy_records)
```

`.copy()` 防止某策略在原地修改共享初始列表。

## 长表到汇总

```python
query_log = pd.DataFrame(records)
metrics = query_log.groupby(
    ["strategy", "round"], as_index=False
).agg(
    regret_mean=("simple_regret", "mean"),
    regret_std=("simple_regret", "std"),
    n_runs=("run_seed", "nunique"),
)
```

先保存长表，再生成汇总；不要反过来手工填写结果。

逐轮长表还必须包含 `candidate_id`、query 前的预测均值/标准差、采集分数、
Oracle 返回和 `model_version`。Random 没有采集分数时保存 `NaN`，不是伪造 0。

## 最低断言

```python
assert query_log["candidate_id"].notna().all()
assert (query_log["simple_regret"] >= -1e-12).all()
assert metrics["n_runs"].eq(len(config["run_seeds"])).all()
```

还应检查每个策略/种子无重复 query、最终预算一致和初始轮结果一致。

## 报告

报告必须引用配置和原始日志，写清：

- 哪个策略在什么协议下表现如何；
- 差异是否稳定；
- 哪些结果为负；
- 当前数据不是粘合剂实验；
- 下一步怎样接作者论文或真实 Oracle。

若多个策略的最低平均 regret 相同，必须写“并列”，不能因为排序表的第一行
恰好是 EI 就宣布 EI 唯一最好。
