# Day 21 算法走读：建立统一比较表

## 步骤 1：写下协议

在任何 `fit()` 前记录：

```text
数据：ESOL / ECFP 1024
外部划分：固定 scaffold train/valid
目标：原始 logS
主指标：valid RMSE
辅助指标：MAE、R²
测试策略：不生成 test 预测
传统配置来源：experiments/esol/day01_baseline/results/run_config.json
候选：Dummy、Ridge、受限决策树、随机森林、冻结 MLP
```

## 步骤 2：读取冻结候选

配置缺失时应停止并解释，不能静默退回默认参数：

```python
config_path = REPO_ROOT / "experiments/esol/day01_baseline/results/run_config.json"
if not config_path.exists():
    raise FileNotFoundError("缺少 Day 07 冻结配置")
params = json.loads(config_path.read_text())["model_params"]
```

传统模型按保存参数创建；MLP 使用 Day 20 冻结配方：

```python
dummy = DummyRegressor(**params["dummy_mean"])
ridge = Ridge(**params["ridge"])
tree = DecisionTreeRegressor(**params["decision_tree_regularized"])
rf = RandomForestRegressor(**params["random_forest"])
mlp = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(32,), alpha=0.001,
        early_stopping=True, max_iter=300,
        n_iter_no_change=10, random_state=42,
    ),
)
```

Day 07 的随机森林冻结为 `n_jobs=-1`；因此计时需要记录并行设置，
不能与单线程或另一台机器直接比较。

## 步骤 3：每个模型只拟合一次

```python
for name, model in models.items():
    started = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
```

停止计时后才调用 `predict()`。不要在循环中改变 `X_train`，
也不要按模型创建不同验证样本。

## 步骤 4：保存训练与验证指标

训练指标帮助诊断拟合程度，验证指标用于当前比较。结果使用长表更易审计：

| model | split | rmse | mae | r2 | fit_seconds |
|---|---|---:|---:|---:|---:|

训练时间在同一个模型的两行中可重复记录，或单独放模型级表格。

## 步骤 5：程序化检查公平性

```python
assert set(results["split"]) == {"train", "valid"}
assert results.groupby(["model", "split"]).size().eq(1).all()
assert np.isfinite(results[["rmse", "mae", "r2"]]).all().all()
```

再人工检查所有 Pipeline 的参数与输入索引。

## 步骤 6：计算差值

```python
best_traditional = valid_rows.query("family == 'traditional'").iloc[0]
mlp_row = valid_rows.query("model == 'mlp'").iloc[0]
delta = mlp_row["rmse"] - best_traditional["rmse"]
```

`delta < 0` 表示当前 MLP RMSE 更低；它只是当前验证差值，不含统计显著性。

## 步骤 7：冻结进入混合阶段的基线

选择 Day 22–24 的基础模型时，除了验证 RMSE，还记录：

- 是否稳定；
- 是否容易复现；
- 训练成本；
- 与另一个模型是否可能提供不同误差模式。

混合模型需要“互补”，不只是把两个最复杂模型放在一起。

## 失败检查

- 任一模型看过外部 valid 标签进行拟合或调参；
- 一个模型多次试验只报告最好一次；
- 缩放在全数据上预先完成；
- 结果表遗漏 Dummy 或最强简单基线；
- 从当前排名直接推断粘合剂结论。
