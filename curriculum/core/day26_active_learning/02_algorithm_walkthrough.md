# Day 26 算法走读：一轮选择—揭示—更新

## 步骤 1：固定划分与预算

```python
INITIAL_SIZE = 160
BATCH_SIZE = 10
ENSEMBLE_SEEDS = [11, 22, 33, 44, 55]
SPLIT_SEED = 42
RANDOM_QUERY_SEED = 2026
```

从原 train 再分为初始已标注集和候选池；外部 valid 不参与选择，
仅作为两种策略共用的固定对照。它在课程历史中已被查看过，不是严格未见 test。
成员继续使用 Day 25 冻结的 `RandomForestRegressor(n_estimators=120,
max_features="sqrt", n_jobs=1)`；本日不因结果好坏改配方。

## 步骤 2：隔离池标签

```python
X_labeled = X_train[initial_idx]
y_labeled = y_train[initial_idx]
X_pool = X_train[pool_idx]
pool_ids = train_ids[pool_idx]
```

这里故意不创建 `hidden_pool_labels` 或 oracle。更严格的工程会把标签放在
仅“实验返回”服务可访问的位置；教程到 query 固定后才实例化模拟 oracle。

## 步骤 3：query 前只使用允许变量

```python
pool_predictions = []
for seed in ENSEMBLE_SEEDS:
    model = make_member(seed)
    model.fit(X_labeled, y_labeled)
    pool_predictions.append(model.predict(X_pool))

prediction_matrix = np.vstack(pool_predictions)
score = prediction_matrix.std(axis=0)
query_pos = np.lexsort((
    pool_ids.astype(str),
    -score,
))[:BATCH_SIZE]
query_ids = pool_ids[query_pos]

random_rng = np.random.default_rng(RANDOM_QUERY_SEED)
random_pos = random_rng.choice(len(X_pool), size=BATCH_SIZE, replace=False)
```

到这里主动与随机 query 都已固定，而且没有读取任何 pool 标签。

## 步骤 4：保存 query，再揭示

```python
query_table = pd.DataFrame({
    "candidate_id": query_ids,
    "predicted_logS": prediction_matrix.mean(axis=0)[query_pos],
    "acquisition_score": score[query_pos],
    "reason": "top ensemble disagreement",
})

# ===== 标签揭示线：模拟实验返回 =====
oracle = SimulatedLabelOracle(y_train[pool_idx])
selected_y = oracle.reveal(query_pos)
```

公开 ESOL 中的 `selected_y` 只是池模拟。oracle 是模拟实验基础设施，
不会传入采集函数；真实系统中它应是实验完成后的外部返回。

## 步骤 5：更新与评价

```python
X_next = np.concatenate([X_labeled, X_pool[query_pos]], axis=0)
y_next = np.concatenate([y_labeled, selected_y], axis=0)
```

用相同模型定义分别在更新前后训练，并在固定外部 valid 上评价。
该表只验证本轮流程，不是无偏最终性能。

## 步骤 6：核对已在揭示前固定的随机对照

```python
assert len(random_pos) == BATCH_SIZE
assert len(np.unique(random_pos)) == BATCH_SIZE
```

`random_pos` 已在步骤 3、标签揭示线之前生成；这里不重新抽签。
随机策略必须使用相同初始已标注集和预算。主动与随机的模型、评估集和指标也必须一致。

## 步骤 7：正式实验扩展（本教程不假装已经完成）

把整个流程封装为 `run_one_simulation(seed)`，对预先声明的初始化种子与多轮预算运行，
保存完整学习曲线和逐种子指标，不只保存均值或最好一次。本教程止于单轮机制，
因此不能据此宣布主动策略更好。

## 最重要的代码审计

在 query 固定前的 cell 中搜索：

- `hidden_pool_labels` / `y_pool`；
- 候选真实误差；
- 候选真实排名；
- 用池标签调权重的代码。

任何出现都需要解释；正常采集路径中应为零次。

## 解释模板

> 本次仅在已有 ESOL 标签上进行单轮池模拟。分歧策略新增【预算】个已知样本后，固定课程 valid 指标由【值】变为【值】；同预算随机对照为【值】。该 valid 并非严格未见 test，单轮差异不构成策略优越证据，也不代表真实实验节省；本教程未生成或测试下游任务候选。
