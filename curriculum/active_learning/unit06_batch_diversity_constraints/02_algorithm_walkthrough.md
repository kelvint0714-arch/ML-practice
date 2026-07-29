# Unit 06 算法走读：贪心多样性批次

## 可行性掩码

```python
feasible_mask = (
    (temperature >= 20)
    & (temperature <= 120)
    & (component_a + component_b <= 1.0)
)
eligible_positions = np.flatnonzero(feasible_mask)
```

- `&` 表示逐元素“并且”；
- 每个条件要加括号；
- `flatnonzero` 返回为 True 的位置。

## 标准化采集分数

```python
score_min = score[eligible_positions].min()
score_range = np.ptp(score[eligible_positions])
normalized_score = (score - score_min) / max(score_range, 1e-12)
```

`np.ptp` 是最大值减最小值。只使用候选的采集分数，不涉及隐藏标签。

## 多样性贪心

第一项选最高分；之后每次计算候选到已选集合的最小距离：

\[
d(x,S)=\min_{s\in S}\lVert x-s\rVert
\]

综合：

\[
a_{\text{batch}}(x)=\tilde a(x)+\lambda \tilde d(x,S)
\]

`lambda=0` 退化为 top-score，越大越重视覆盖。

## 变量位置

`eligible_positions`、`selected_positions` 都是原 `X_pool` 的位置。若先创建 `X_eligible`，必须清楚局部位置和原池位置的映射。

## 审核状态

推荐表可使用：

| candidate_id | acquisition_score | feasible | review_status | rejection_reason |
|---|---:|---:|---|---|
| C12 | 1.83 | True | proposed | |

算法只创建 `proposed`；人工或受控系统负责 `approved/rejected`。
`proposed` 表中不能提前出现 `observed_y`。只有 approved 候选完成实验后，
Oracle 才返回标签；失败行保留 `failed` 和原因，拒绝行保留 `rejected` 和原因。

## 批内距离

对称距离矩阵只读取上三角（不含对角线）：

```python
distances = pairwise_distances(X_batch)
pairwise_values = distances[
    np.triu_indices(len(X_batch), k=1)
]
minimum_distance = pairwise_values.min()
mean_distance = pairwise_values.mean()
```

不能把所有等于 0 的元素改为无穷：不同候选可能拥有完全相同特征，
这种真实重复的距离 0 必须被保留。
