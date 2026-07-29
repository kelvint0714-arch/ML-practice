# Unit 04 算法走读：索引怎样移动

## 初始状态

假设候选永久编号为 `C0`–`C9`，初始已标注位置为 `[0, 5]`：

```python
labeled_indices = [0, 5]
pool_indices = [1, 2, 3, 4, 6, 7, 8, 9]
```

## 局部与全局位置

模型对 `X[pool_indices]` 预测。若采集函数返回：

```python
query_local = 2
```

它表示剩余池中的第 3 行，对应：

```python
query_global = pool_indices[2]  # 3
query_id = candidate_ids[3]    # "C3"
```

直接把 `2` 当成全局位置会错误查询 `C2`。

## 安全更新

```python
labeled_indices.append(query_global)
pool_indices.remove(query_global)

assert set(labeled_indices).isdisjoint(pool_indices)
assert len(set(labeled_indices)) == len(labeled_indices)
```

- `append` 把全局位置加入已标注列表；
- `remove` 按值从候选列表删除；
- `isdisjoint` 检查两集合没有交集；
- 第二个断言防止重复查询。

## 日志顺序

```python
record = {
    "round": round_number,
    "candidate_id": query_id,
    "prediction_mean": mean[query_local],
    "prediction_std": std[query_local],
    "acquisition_score": score[query_local],
    "model_version": "fixed_gp_rbf_v1",
}

# query 已固定，下面才加入 observed_y。
record["observed_y"] = oracle(query_global)
```

字典字段顺序不是权限控制；真正的权限来自函数参数和代码执行顺序。
策略函数应持有 `y_labeled`，而不是完整 `y_pool`。全局最优与 regret
由 campaign 结束后的 evaluator 计算。

## 可重复随机对照

Random 应使用单独、预先固定的随机数生成器：

```python
rng = np.random.default_rng(run_seed)
query_local = int(rng.integers(len(pool_indices)))
```

每个策略获得相同初始索引，但 Random 的选点由自己的种子决定。
